"""Kraken Derivatives perpetual scanner: BTC SQZMOM phase + 3-bar pivots + 5-bar SMC OB touch.

BTC pivots define the directional phase and BTC itself is exempt from SMC OB.
Altcoin pivots must match the active BTC phase and pass the matching SMC OB.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from indicator_engine import evaluate_signals, latest_confirmed_sqzmom_pivot, squeeze_momentum

REST_BASE = "https://futures.kraken.com/derivatives/api/v3"
CHART_BASE = "https://futures.kraken.com/api/charts/v1"
TIMEFRAME = "30m"
INTERVAL_MS = 30 * 60 * 1000
CANDLE_LIMIT = 1000
MIN_CANDLES = 250
MAX_WORKERS = 5
STATE_FILE = Path("state.json")
PERPETUAL_PREFIXES = ("PF_", "PI_")

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Kraken-Derivatives-SQZMOM-SMC-Scanner/1.1-diagnostics", "Accept": "application/json"})
retry = Retry(total=4, connect=4, read=3, status=3, backoff_factor=0.7,
              status_forcelist=(408, 429, 500, 502, 503, 504),
              allowed_methods=frozenset(["GET"]), respect_retry_after_header=True)
adapter = HTTPAdapter(max_retries=retry, pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS)
SESSION.mount("https://", adapter)


def get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    response = SESSION.get(url, params=params, timeout=(10, 30))
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"Kraken returned non-JSON data: {url}") from exc
    if isinstance(payload, dict) and payload.get("result") == "error":
        raise RuntimeError(f"Kraken API error: {payload.get('errors', payload)}")
    if isinstance(payload, dict) and payload.get("error"):
        raise RuntimeError(f"Kraken API error: {payload['error']}")
    return payload


def telegram_send(message: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        raise RuntimeError("GitHub Secrets TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required")
    response = SESSION.post(f"https://api.telegram.org/bot{token}/sendMessage",
                            data={"chat_id": chat_id, "text": message, "disable_web_page_preview": True},
                            timeout=(10, 25))
    response.raise_for_status()
    payload = response.json()
    if not payload.get("ok"):
        raise RuntimeError(f"Telegram API rejected the message: {payload}")


def load_state() -> set[str]:
    if not STATE_FILE.exists():
        return set()
    try:
        return set(json.loads(STATE_FILE.read_text(encoding="utf-8")).get("sent", []))
    except (OSError, ValueError, TypeError) as exc:
        print(f"WARNING: state.json could not be read; starting with empty history: {exc}")
        return set()


def save_state(sent: set[str]) -> None:
    temp = STATE_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps({"sent": sorted(sent)[-15000:]}, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(STATE_FILE)


def fetch_symbols() -> list[str]:
    """Return listed perpetual futures only; exclude dated futures, spot and options."""
    payload = get_json(f"{REST_BASE}/instruments")
    instruments = payload.get("instruments", []) if isinstance(payload, dict) else []
    symbols: set[str] = set()
    for item in instruments:
        symbol = str(item.get("symbol") or item.get("tradeable") or "").upper()
        if not symbol.startswith(PERPETUAL_PREFIXES):
            continue
        if item.get("tradeable") is False or item.get("isExpired") is True:
            continue
        status = str(item.get("status", "")).lower()
        if status and status not in {"online", "trading", "active"}:
            continue
        symbols.add(symbol)
    if not symbols:
        raise RuntimeError("No Kraken perpetual instruments found; refusing to scan a different market")
    return sorted(symbols)


def fetch_closed_candles(symbol: str, now_ms: int | None = None) -> pd.DataFrame:
    """Fetch Kraken trade candles and discard the still-forming 30m candle."""
    now_ms = int(now_ms if now_ms is not None else time.time() * 1000)
    url = f"{CHART_BASE}/trade/{quote(symbol, safe='_')}/{TIMEFRAME}"
    end_s = now_ms // 1000
    start_s = end_s - (CANDLE_LIMIT + 5) * (INTERVAL_MS // 1000)
    payload = get_json(url, {"from": start_s, "to": end_s, "count": CANDLE_LIMIT})
    rows = payload.get("candles", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame()
    normalized = []
    for row in rows:
        try:
            stamp = int(row["time"] if isinstance(row, dict) else row[0])
            if stamp < 10**12:
                stamp *= 1000
            if stamp + INTERVAL_MS > now_ms:
                continue
            if isinstance(row, dict):
                opn, high, low, close = row["open"], row["high"], row["low"], row["close"]
                volume = row.get("volume", 0)
            else:
                opn, high, low, close, volume = row[1:6]
            normalized.append((stamp, float(opn), float(high), float(low), float(close), float(volume)))
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    if len(normalized) < MIN_CANDLES:
        return pd.DataFrame()
    frame = pd.DataFrame(normalized, columns=["timestamp", "open", "high", "low", "close", "volume"])
    frame = frame.drop_duplicates("timestamp").sort_values("timestamp")
    frame["timestamp"] = pd.to_datetime(frame.timestamp, unit="ms", utc=True)
    return frame.set_index("timestamp")


def evaluate_symbol(symbol: str, now_ms: int, btc_bias: str, btc_candle_time_ms: int) -> dict[str, Any]:
    """Evaluate one alt contract; it must pass BTC phase and matching SMC OB."""
    candles = fetch_closed_candles(symbol, now_ms)
    if candles.empty:
        return {"symbol": symbol, "status": "NO_DATA_OR_LT_250_CANDLES"}

    result = evaluate_signals(candles)
    if result.empty:
        return {"symbol": symbol, "status": "ENGINE_EMPTY"}

    last = result.iloc[-1]
    signal_time_ms = int(result.index[-1].timestamp() * 1000)
    lag_bars = max(0, int((btc_candle_time_ms - signal_time_ms) // INTERVAL_MS))
    aligned = signal_time_ms == btc_candle_time_ms

    buy_pivot = bool(last.get("buy_trough", False))
    sell_pivot = bool(last.get("sell_peak", False))
    buy_ob = bool(last.get("buy_ob_confirm", False))
    sell_ob = bool(last.get("sell_ob_confirm", False))
    buy_signal = bool(last.get("BUY", False))
    sell_signal = bool(last.get("SELL", False))

    # Stage classification is diagnostic only; no trading condition is changed.
    if buy_pivot and sell_pivot:
        pivot_kind = "BOTH_PIVOTS_CONFLICT"
    elif buy_pivot:
        pivot_kind = "BUY_PIVOT"
    elif sell_pivot:
        pivot_kind = "SELL_PIVOT"
    else:
        pivot_kind = "NO_PIVOT_ON_LATEST_CANDLE"

    if buy_pivot and not buy_ob:
        ob_rejection = "BUY_PIVOT_WITHOUT_GREEN_OB"
    elif sell_pivot and not sell_ob:
        ob_rejection = "SELL_PIVOT_WITHOUT_RED_OB"
    elif not (buy_pivot or sell_pivot):
        ob_rejection = "NOT_APPLICABLE"
    else:
        ob_rejection = "OB_CONFIRMED"

    raw_signal = "BUY" if buy_signal else "SELL" if sell_signal else ""
    if not aligned:
        status = "CANDLE_TIME_MISMATCH"
    elif raw_signal and raw_signal != btc_bias:
        status = f"{raw_signal}_BLOCKED_BY_BTC_BIAS"
    elif raw_signal:
        status = "SIGNAL_READY"
    elif buy_pivot and buy_ob and btc_bias != "BUY":
        status = "BUY_BLOCKED_BY_BTC_BIAS"
    elif sell_pivot and sell_ob and btc_bias != "SELL":
        status = "SELL_BLOCKED_BY_BTC_BIAS"
    elif buy_pivot or sell_pivot:
        status = ob_rejection
    else:
        status = "NO_LATEST_PIVOT"

    return {
        "symbol": symbol,
        "status": status,
        "pivot_kind": pivot_kind,
        "buy_pivot": buy_pivot,
        "sell_pivot": sell_pivot,
        "buy_ob_confirm": buy_ob,
        "sell_ob_confirm": sell_ob,
        "buy_ob_inside_now": bool(last.get("bull_ob_inside", False)),
        "sell_ob_inside_now": bool(last.get("bear_ob_inside", False)),
        "buy_ob_touch_now": bool(last.get("bull_ob_touch", False)),
        "sell_ob_touch_now": bool(last.get("bear_ob_touch", False)),
        "buy_ob_touch_5": bool(last.get("bull_ob_touch_5", False)),
        "sell_ob_touch_5": bool(last.get("bear_ob_touch_5", False)),
        "buy_ob_count_now": int(last.get("bull_ob_count", 0)),
        "sell_ob_count_now": int(last.get("bear_ob_count", 0)),
        "raw_signal": raw_signal,
        "aligned": aligned,
        "lag_bars": lag_bars,
        "candle_time": result.index[-1].strftime("%Y-%m-%d %H:%M UTC"),
        "timestamp_ms": signal_time_ms,
        "close": float(last.close),
        "sqzmom": float(last.sqzmom),
        "peak_trough": float(last.sqzmom_peak_trough),
        "btc_bias": btc_bias,
    }


def make_alert(item: dict[str, Any]) -> str:
    title = "🟢 BUY / LONG" if item["raw_signal"] == "BUY" else "🔴 SELL / SHORT"
    ob_name = "باکس سبز Bullish OB" if item["raw_signal"] == "BUY" else "باکس قرمز Bearish OB"
    return "\n".join([
        title,
        f"قرارداد: {item['symbol']} — Kraken Derivatives Perpetual",
        f"تایم‌فریم: {TIMEFRAME}",
        f"قیمت بسته‌شدن: {item['close']:.10g}",
        f"SQZMOM میله میانی قله/دره: {item['peak_trough']:.10g}",
        f"SQZMOM آخرین میله: {item['sqzmom']:.10g}",
        f"جهت فعال BTC بر اساس آخرین قله/دره سه‌میله‌ای SQZMOM: {item['btc_bias']}",
        ("تأیید SMC لازم نیست؛ بیت‌کوین مرجع جهت بازار است." if item.get("is_btc") else
         f"تأیید SMC: تاچ شدن باکس فعال هم‌رنگ {ob_name} توسط حداقل یکی از پنج کندل آخر (تماس سایه یا بدنه کافی است)"),
        f"زمان کندل بسته‌شده: {item['candle_time']}",
        "فقط هشدار تحلیلی است؛ هیچ سفارش یا معامله‌ای ثبت نمی‌شود.",
    ])



def build_btc_signal(btc_candles: pd.DataFrame, btc_pivot: dict, btc_symbol: str) -> dict[str, Any] | None:
    """BTC emits its own pivot signal without SMC; only a newly confirmed last-bar pivot qualifies."""
    if int(btc_pivot.get("confirm_index", -1)) != len(btc_candles) - 1:
        return None
    sqz = squeeze_momentum(btc_candles)
    last = btc_candles.iloc[-1]
    hist = sqz["sqzmom"]
    signal_time = btc_candles.index[-1]
    return {
        "symbol": btc_symbol,
        "status": "SIGNAL_READY",
        "raw_signal": str(btc_pivot["bias"]),
        "is_btc": True,
        "btc_bias": str(btc_pivot["bias"]),
        "close": float(last["close"]),
        "sqzmom": float(hist.iloc[-1]),
        "peak_trough": float(btc_pivot["pivot_value"]),
        "candle_time": signal_time.strftime("%Y-%m-%d %H:%M UTC"),
        "timestamp_ms": int(signal_time.timestamp() * 1000),
    }


def print_diagnostics(records: list[dict[str, Any]], btc_bias: str, btc_symbol: str,
                      btc_pivot: dict, btc_candle_time: pd.Timestamp, total_alts: int) -> None:
    counts = Counter(r.get("status", "UNKNOWN") for r in records)
    counts.update({})
    buy_pivots = sum(bool(r.get("buy_pivot")) for r in records)
    sell_pivots = sum(bool(r.get("sell_pivot")) for r in records)
    buy_ob = sum(bool(r.get("buy_pivot")) and bool(r.get("buy_ob_confirm")) for r in records)
    sell_ob = sum(bool(r.get("sell_pivot")) and bool(r.get("sell_ob_confirm")) for r in records)
    evaluated = sum("aligned" in r for r in records)
    aligned = sum(bool(r.get("aligned")) for r in records)
    mismatched = evaluated - aligned

    print("\n========== SIGNAL DIAGNOSTICS ==========")
    print(f"BTC symbol={btc_symbol} | active_phase={btc_bias} | last_phase_change={btc_pivot['confirm_time']}")
    print(f"BTC last closed candle={btc_candle_time} | latest SQZMOM={btc_pivot.get('latest_histogram')}")
    print(f"ALT universe={total_alts} | returned_records={len(records)} | aligned_to_BTC_candle={aligned} | time_mismatch={mismatched}")
    print(f"ALT current-candle 3-bar pivots: BUY_TROUGH={buy_pivots} | SELL_PEAK={sell_pivots}")
    print(f"Pivots with matching active colored OB touch in 5-bar window: BUY+GREEN_OB={buy_ob} | SELL+RED_OB={sell_ob}")
    print("Stage outcomes:")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")

    # Print every pivot candidate and every nontrivial rejection, keeping log readable.
    notable = [r for r in records if r.get("buy_pivot") or r.get("sell_pivot") or
               r.get("status") in {"CANDLE_TIME_MISMATCH", "NO_DATA_OR_LT_250_CANDLES", "ENGINE_EMPTY"}]
    if notable:
        print("Per-symbol candidates / data exclusions:")
        for r in sorted(notable, key=lambda x: (x.get("status", ""), x.get("symbol", ""))):
                print(
                "  {symbol}: status={status} pivot={pivot_kind} buyOBtouch5={buy_ob_confirm} "
                "sellOBtouch5={sell_ob_confirm} closeInsideGreenOBs={buy_ob_count_now} "
                "closeInsideRedOBs={sell_ob_count_now} aligned={aligned} lagBars={lag_bars}".format(
                    symbol=r.get("symbol"), status=r.get("status"),
                    pivot_kind=r.get("pivot_kind", "-"),
                    buy_ob_confirm=r.get("buy_ob_confirm", "-"),
                    sell_ob_confirm=r.get("sell_ob_confirm", "-"),
                    buy_ob_count_now=r.get("buy_ob_count_now", "-"),
                    sell_ob_count_now=r.get("sell_ob_count_now", "-"),
                    aligned=r.get("aligned", "-"), lag_bars=r.get("lag_bars", "-")))
    else:
        print("Per-symbol candidates: none on the latest closed candle.")
    print("=========================================================\n")


def main() -> None:
    now_ms = int(time.time() * 1000)
    print(f"Starting Kraken Derivatives perpetual scanner | {TIMEFRAME} | BTC SQZMOM phase + 3-bar pivots + 5-bar SMC OB touch")
    symbols = fetch_symbols()
    btc_symbol = next((s for s in ("PF_XBTUSD", "PI_XBTUSD") if s in symbols), None)
    if btc_symbol is None:
        raise RuntimeError("No BTC perpetual contract found in Kraken instrument catalogue")
    btc_candles = fetch_closed_candles(btc_symbol, now_ms)
    if btc_candles.empty:
        raise RuntimeError("BTC candles unavailable; refusing to scan alts without a confirmed BTC phase")
    btc_pivot = latest_confirmed_sqzmom_pivot(btc_candles)
    if btc_pivot is None:
        raise RuntimeError("No confirmed BTC SQZMOM 3-closed-bar peak/trough found; no altcoin scan this cycle")
    btc_bias = str(btc_pivot["bias"])
    btc_signal = build_btc_signal(btc_candles, btc_pivot, btc_symbol)
    btc_candle_time_ms = int(btc_candles.index[-1].timestamp() * 1000)
    btc_sqz = squeeze_momentum(btc_candles)
    btc_last3 = [None if pd.isna(v) else float(v) for v in btc_sqz["sqzmom"].tail(3).tolist()]
    btc_candle_time = btc_candles.index[-1]
    print(f"BTC reference: {btc_symbol} | active phase={btc_bias} | last phase change={btc_pivot['confirm_time']} | last closed candle={btc_candle_time}")
    print(f"BTC last 3 closed SQZMOM bars (oldest->newest): {btc_last3}")
    print("BTC SMC exemption: ON | BTC pivot alert: " + (btc_signal["raw_signal"] if btc_signal else "none on latest closed candle"))

    sent = load_state()
    alt_symbols = [s for s in symbols if s not in {"PF_XBTUSD", "PI_XBTUSD"}]
    print(f"Active Kraken perpetual contracts: {len(symbols)} | BTC reference excluded | alt contracts queued={len(alt_symbols)}")

    records: list[dict[str, Any]] = []
    errors = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(evaluate_symbol, symbol, now_ms, btc_bias, btc_candle_time_ms): symbol
                   for symbol in alt_symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                records.append(future.result())
            except requests.HTTPError as exc:
                errors += 1
                status = exc.response.status_code if exc.response is not None else "unknown"
                records.append({"symbol": symbol, "status": f"HTTP_ERROR_{status}"})
                print(f"HTTP error for {symbol} (status {status}): {exc}")
                if status == 429:
                    raise
            except Exception as exc:
                errors += 1
                records.append({"symbol": symbol, "status": f"ENGINE_OR_FETCH_ERROR_{type(exc).__name__}"})
                print(f"Skipped {symbol}: {type(exc).__name__}: {exc}")

    print_diagnostics(records, btc_bias, btc_symbol, btc_pivot, btc_candle_time, len(alt_symbols))

    found: list[dict[str, Any]] = []
    # BTC is the directional reference and is the only instrument exempt from SMC OB.
    if btc_signal is not None:
        btc_key = f"{btc_signal['symbol']}|{TIMEFRAME}|{btc_signal['timestamp_ms']}|{btc_signal['raw_signal']}|BTC_SELF"
        if btc_key not in sent:
            btc_signal["dedupe_key"] = btc_key
            found.append(btc_signal)
    for item in records:
        if item.get("status") != "SIGNAL_READY":
            continue
        key = f"{item['symbol']}|{TIMEFRAME}|{item['timestamp_ms']}|{item['raw_signal']}|BTC:{btc_bias}"
        if key not in sent:
            item["dedupe_key"] = key
            found.append(item)

    found.sort(key=lambda x: (x["raw_signal"], not x.get("is_btc", False), x["symbol"]))
    alerts = 0
    for item in found:
        try:
            telegram_send(make_alert(item))
            sent.add(item["dedupe_key"])
            save_state(sent)
            alerts += 1
            suffix = " SMC_EXEMPT(BTC)" if item.get("is_btc") else " SMC_CONFIRMED"
            print(f"ALERT {item['symbol']} {item['raw_signal']} {item['candle_time']} BTC={btc_bias}{suffix}")
        except Exception as exc:
            print(f"Telegram send failed for {item['symbol']}: {type(exc).__name__}: {exc}")
    save_state(sent)
    print(f"Finished: checked={len(records)}/{len(alt_symbols)} BTC_bias={btc_bias} new_alerts={alerts} errors={errors}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
