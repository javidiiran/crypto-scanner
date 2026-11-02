"""Kraken Derivatives perpetual scanner: closed 30m candles, SQZMOM trigger + SMC confirmation."""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from indicator_engine import evaluate_signals, latest_confirmed_sqzmom_pivot

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
SESSION.headers.update({"User-Agent": "Kraken-Derivatives-SQZMOM-SMC-Scanner/1.0", "Accept": "application/json"})
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
        # Kraken's instrument catalogue uses tradeable for live tradability on
        # some contract records; do not reject records where that field is absent.
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
                # The documented Kraken charts response is an object array;
                # this positional fallback is only for compatible proxy responses.
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


def evaluate_symbol(symbol: str, now_ms: int, btc_bias: str, btc_candle_time_ms: int) -> dict[str, Any] | None:
    candles = fetch_closed_candles(symbol, now_ms)
    if candles.empty:
        return None
    # Avoid evaluating an altcoin against a BTC phase from a different 30m bar.
    latest_time_ms = int(candles.index[-1].timestamp() * 1000)
    if latest_time_ms != btc_candle_time_ms:
        return None
    result = evaluate_signals(candles)
    if result.empty:
        return None
    last = result.iloc[-1]
    signal = str(last.signal)
    # BTC phase is a hard directional gate: no counter-trend altcoin alerts.
    if signal not in ("BUY", "SELL") or signal != btc_bias:
        return None
    candle_time = result.index[-1]
    return {
        "symbol": symbol, "signal": signal,
        "timestamp_ms": int(candle_time.timestamp() * 1000),
        "candle_time": candle_time.strftime("%Y-%m-%d %H:%M UTC"),
        "close": float(last.close), "sqzmom": float(last.sqzmom),
        "peak_trough": float(last.sqzmom_peak_trough),
        "bull_ob_inside": bool(last.bull_ob_inside),
        "bear_ob_inside": bool(last.bear_ob_inside),
        "ob_confirmed_by_three": bool(last.buy_ob_confirm if signal == "BUY" else last.sell_ob_confirm),
        "btc_bias": btc_bias,
    }


def make_alert(item: dict[str, Any]) -> str:
    title = "🟢 BUY / LONG" if item["signal"] == "BUY" else "🔴 SELL / SHORT"
    ob_name = "باکس سبز Bullish OB" if item["signal"] == "BUY" else "باکس قرمز Bearish OB"
    return "\n".join([
        title,
        f"قرارداد: {item['symbol']} — Kraken Derivatives Perpetual",
        f"تایم‌فریم: {TIMEFRAME}",
        f"قیمت بسته‌شدن: {item['close']:.10g}",
        f"SQZMOM میله میانی قله/دره: {item['peak_trough']:.10g}",
        f"SQZMOM آخرین میله: {item['sqzmom']:.10g}",
        f"جهت فعال BTC بر اساس آخرین قله/دره سه‌کندلی: {item['btc_bias']}",
        f"تأیید SMC: بسته‌شدن حداقل یکی از سه کندل الگوی SQZMOM داخل {ob_name}",
        f"زمان کندل بسته‌شده: {item['candle_time']}",
        "فقط هشدار تحلیلی است؛ هیچ سفارش یا معامله‌ای ثبت نمی‌شود.",
    ])


def main() -> None:
    now_ms = int(time.time() * 1000)
    print(f"Starting Kraken Derivatives perpetual scanner | {TIMEFRAME} | BTC SQZMOM phase + 3-bar alt pivots + colored OB")
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
    btc_candle_time_ms = int(btc_candles.index[-1].timestamp() * 1000)
    print(f"BTC reference: {btc_symbol} | active bias={btc_bias} | latest confirmed pivot={btc_pivot['confirm_time']} | last closed candle={btc_candles.index[-1]}")

    sent = load_state()
    print(f"Active Kraken perpetual contracts: {len(symbols)} | BTC reference excluded from alt scan")
    found: list[dict[str, Any]] = []
    checked = errors = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(evaluate_symbol, symbol, now_ms, btc_bias, btc_candle_time_ms): symbol
                   for symbol in symbols if symbol not in {"PF_XBTUSD", "PI_XBTUSD"}}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                item = future.result()
                checked += 1
                if item:
                    key = f"{item['symbol']}|{TIMEFRAME}|{item['timestamp_ms']}|{item['signal']}|BTC:{btc_bias}"
                    if key not in sent:
                        item["dedupe_key"] = key
                        found.append(item)
            except requests.HTTPError as exc:
                errors += 1
                status = exc.response.status_code if exc.response is not None else "unknown"
                print(f"HTTP error for {symbol} (status {status}): {exc}")
                if status == 429:
                    raise
            except Exception as exc:
                errors += 1
                print(f"Skipped {symbol}: {type(exc).__name__}: {exc}")

    found.sort(key=lambda x: (x["signal"], x["symbol"]))
    alt_symbols = [s for s in symbols if s not in {"PF_XBTUSD", "PI_XBTUSD"}]
    alerts = 0
    for item in found:
        try:
            telegram_send(make_alert(item))
            sent.add(item["dedupe_key"])
            save_state(sent)
            alerts += 1
            print(f"ALERT {item['symbol']} {item['signal']} {item['candle_time']} BTC={btc_bias}")
        except Exception as exc:
            print(f"Telegram send failed for {item['symbol']}: {type(exc).__name__}: {exc}")
    save_state(sent)
    print(f"Finished: checked={checked}/{len(alt_symbols)} BTC_bias={btc_bias} new_alerts={alerts} errors={errors}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
