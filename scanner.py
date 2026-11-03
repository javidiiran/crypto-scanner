"""Kraken Futures Wave Trend + Order Blocks/FVG scanner (30-minute, closed bars)."""
from __future__ import annotations

import json
import math
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

from indicator_engine import analyze_market

REST_BASE = "https://futures.kraken.com/derivatives/api/v3"
CHART_BASE = "https://futures.kraken.com/api/charts/v1"
TIMEFRAME = "30m"
INTERVAL_MS = 30 * 60 * 1000
CANDLE_LIMIT = 1000
MIN_CANDLES = 60
MAX_WORKERS = 5
STATE_FILE = Path("state.json")
PERPETUAL_PREFIXES = ("PF_", "PI_")
BTC_SYMBOLS = ("PF_XBTUSD", "PI_XBTUSD")

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Kraken-Futures-WaveTrend-OB-FVG-Scanner/1.1", "Accept": "application/json"})
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
    response = SESSION.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data={"chat_id": chat_id, "text": message, "disable_web_page_preview": True},
        timeout=(10, 25),
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("ok"):
        raise RuntimeError(f"Telegram API rejected the message: {payload}")


def load_state() -> dict[str, set[str]]:
    if not STATE_FILE.exists():
        return {"sent": set(), "cancelled": set()}
    try:
        raw = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        # Upgrade old list-only/old scanner state safely.
        if isinstance(raw, dict):
            return {
                "sent": set(raw.get("sent", [])),
                "cancelled": set(raw.get("cancelled", [])),
            }
    except (OSError, ValueError, TypeError) as exc:
        print(f"WARNING: state.json could not be read; starting empty: {exc}")
    return {"sent": set(), "cancelled": set()}


def save_state(state: dict[str, set[str]]) -> None:
    payload = {
        "sent": sorted(state["sent"])[-20000:],
        "cancelled": sorted(state["cancelled"])[-20000:],
        "scanner": "Kraken WaveTrend OB-FVG 1.1",
    }
    temp = STATE_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(STATE_FILE)


def fetch_symbols() -> list[str]:
    """Only currently tradable Kraken perpetual contracts; no spot/dated contracts."""
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
        raise RuntimeError("No Kraken perpetual instruments found; refusing to scan another exchange")
    return sorted(symbols)


def fetch_closed_candles(symbol: str, now_ms: int | None = None) -> pd.DataFrame:
    """Fetch 30m trade candles and discard any candle that has not closed."""
    now_ms = int(now_ms if now_ms is not None else time.time() * 1000)
    url = f"{CHART_BASE}/trade/{quote(symbol, safe='_')}/{TIMEFRAME}"
    end_s = now_ms // 1000
    start_s = end_s - (CANDLE_LIMIT + 5) * (INTERVAL_MS // 1000)
    payload = get_json(url, {"from": start_s, "to": end_s, "count": CANDLE_LIMIT})
    rows = payload.get("candles", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame()
    normalized: list[tuple[int, float, float, float, float, float]] = []
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


def _fmt_rsi(value: float) -> str:
    return f"{value:.1f}" if math.isfinite(value) else "N/A"


def _signal_key(symbol: str, candidate: dict[str, Any]) -> str:
    return f"{symbol}|{TIMEFRAME}|{candidate['timestamp_ms']}|{candidate['direction']}"


def _btc_context(btc_symbol: str, btc_analysis: dict[str, Any]) -> dict[str, Any]:
    confirmed = [c for c in btc_analysis["candidates"] if c["status"] == "CONFIRMED"]
    latest = max(confirmed, key=lambda c: c["timestamp_ms"]) if confirmed else None
    return {
        "symbol": btc_symbol,
        "direction": latest["direction"] if latest else "NO_CONFIRMED_SIGNAL",
        "signal_time": latest["timestamp"].strftime("%Y-%m-%d %H:%M UTC") if latest else "—",
        "signal_rsi": latest["rsi"] if latest else float("nan"),
        "current_rsi": btc_analysis["latest_rsi"],
    }


def evaluate_symbol(symbol: str, now_ms: int) -> dict[str, Any]:
    candles = fetch_closed_candles(symbol, now_ms)
    if candles.empty:
        return {"symbol": symbol, "status": "NO_DATA_OR_LT_100_CANDLES"}
    analysis = analyze_market(candles)
    return {"symbol": symbol, "status": "OK", "analysis": analysis}


def make_alert(item: dict[str, Any], btc: dict[str, Any]) -> str:
    buy = item["direction"] == "BUY"
    title = "🟢 BUY / LONG" if buy else "🔴 SELL / SHORT"
    priority = "اولویت اول — RSI هم‌جهت (خرید ≥ ۵۰ / فروش < ۵۰)" if item["priority"] == 1 else "اولویت دوم — RSI خلاف جهت (خرید < ۵۰ / فروش ≥ ۵۰)"
    btc_alignment = item.get("btc_alignment", "UNKNOWN")
    btc_alignment_fa = {"ALIGNED": "هم‌جهت با بیت‌کوین", "OPPOSITE": "خلاف جهت بیت‌کوین", "UNKNOWN": "جهت BTC نامشخص"}.get(btc_alignment, btc_alignment)
    match_kind = item.get("match_kind", "OB/FVG")
    return "\n".join([
        title,
        f"قرارداد: {item['symbol']} — Kraken Futures Perpetual",
        f"تایم‌فریم: {TIMEFRAME}",
        f"قیمت کندل سیگنال: {item['price']:.10g}",
        f"RSI(14) ارز هنگام تأیید: {_fmt_rsi(item['rsi'])}",
        f"جهت نسبت به BTC: {btc_alignment_fa}",
        f"اولویت RSI: {priority}",
        f"تأیید: برخورد به باکس هم‌رنگ {match_kind}",
        f"زمان سیگنال Wave Trend: {item['signal_time']}",
        "",
        f"آخرین سیگنال تأییدشده BTC ({btc['symbol']}): {btc['direction']}",
        f"زمان سیگنال BTC: {btc['signal_time']}",
        f"RSI(14) بیت‌کوین هنگام آخرین سیگنال: {_fmt_rsi(btc['signal_rsi'])}",
        f"RSI(14) فعلی بیت‌کوین در آخرین کندل بسته: {_fmt_rsi(btc['current_rsi'])}",
    ])


def make_cancellation(item: dict[str, Any]) -> str:
    return "\n".join([
        "⚠️ لغو سیگنال قبلی",
        f"قرارداد: {item['symbol']} — Kraken Futures",
        f"سیگنال قبلی: {item['direction']} | {item['signal_time']}",
        "علت: در پنجره تأیید، قیمت با باکس مخالف برخورد کرد.",
        "طبق قانون اسکنر، وجود حتی یک برخورد مخالف باعث رد کامل سیگنال است.",
    ])


def _candidate_for_current_window(symbol: str, analysis: dict[str, Any], last_index: int) -> list[dict[str, Any]]:
    """Only newly relevant candidates (signal candle age 0..2) are actionable."""
    out = []
    candles = analysis["candles"]
    touches = analysis["touches"]
    for candidate in analysis["candidates"]:
        age = last_index - int(candidate["signal_index"])
        if age < 0 or age > 2:
            continue
        item = dict(candidate)
        item["symbol"] = symbol
        item["signal_time"] = candidate["timestamp"].strftime("%Y-%m-%d %H:%M UTC")
        # First matching zone class at the first same-direction touch.
        match_idx = candidate.get("first_match_index")
        if match_idx is not None:
            if item["direction"] == "BUY":
                ob = bool(touches["bull_ob_touch"].iloc[match_idx])
                fvg = bool(touches["bull_fvg_touch"].iloc[match_idx])
            else:
                ob = bool(touches["bear_ob_touch"].iloc[match_idx])
                fvg = bool(touches["bear_fvg_touch"].iloc[match_idx])
            item["match_kind"] = "Order Block و FVG" if ob and fvg else "Order Block" if ob else "FVG"
        else:
            item["match_kind"] = "OB/FVG"
        out.append(item)
    return out


def main() -> None:
    now_ms = int(time.time() * 1000)
    print(f"Starting Kraken Futures scanner | {TIMEFRAME} | Wave Trend + Order Blocks/FVG + RSI")
    symbols = fetch_symbols()
    btc_symbol = next((s for s in BTC_SYMBOLS if s in symbols), None)
    if btc_symbol is None:
        raise RuntimeError("No BTC perpetual contract found in Kraken instrument catalogue")

    btc_candles = fetch_closed_candles(btc_symbol, now_ms)
    if btc_candles.empty:
        raise RuntimeError("BTC 30m closed candles unavailable; cannot build BTC context")
    btc_analysis = analyze_market(btc_candles)
    btc = _btc_context(btc_symbol, btc_analysis)
    print(f"BTC context: latest_confirmed={btc['direction']} | signal_time={btc['signal_time']} | current_RSI={_fmt_rsi(btc['current_rsi'])}")

    state = load_state()
    alt_symbols = [s for s in symbols if s not in BTC_SYMBOLS]
    print(f"Tradable perpetual contracts={len(symbols)} | BTC reference={btc_symbol} | other contracts={len(alt_symbols)}")

    records: list[dict[str, Any]] = []
    errors = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(evaluate_symbol, symbol, now_ms): symbol for symbol in alt_symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                records.append(future.result())
            except requests.HTTPError as exc:
                errors += 1
                status = exc.response.status_code if exc.response is not None else "unknown"
                records.append({"symbol": symbol, "status": f"HTTP_ERROR_{status}"})
                print(f"HTTP error {symbol}: {exc}")
                if status == 429:
                    raise
            except Exception as exc:
                errors += 1
                records.append({"symbol": symbol, "status": f"ENGINE_ERROR_{type(exc).__name__}"})
                print(f"Skipped {symbol}: {type(exc).__name__}: {exc}")

    new_alerts: list[dict[str, Any]] = []
    cancellations: list[dict[str, Any]] = []
    candidate_count = 0
    for record in records:
        if record.get("status") != "OK":
            continue
        symbol = record["symbol"]
        analysis = record["analysis"]
        last_index = len(analysis["candles"]) - 1
        candidates = _candidate_for_current_window(symbol, analysis, last_index)
        candidate_count += len(candidates)
        for item in candidates:
            key = _signal_key(symbol, item)
            if item["status"] == "REJECTED_OPPOSITE_TOUCH":
                if key in state["sent"] and key not in state["cancelled"]:
                    item["dedupe_key"] = key
                    cancellations.append(item)
                continue
            if item["status"] == "CONFIRMED" and key not in state["sent"]:
                item["dedupe_key"] = key
                new_alerts.append(item)

    # Required order: BTC-aligned first, then BTC-opposite; within each,
    # RSI priority 1 before priority 2. Unknown BTC direction is last.
    for item in new_alerts:
        if btc["direction"] in {"BUY", "SELL"}:
            item["btc_alignment"] = "ALIGNED" if item["direction"] == btc["direction"] else "OPPOSITE"
        else:
            item["btc_alignment"] = "UNKNOWN"
    alignment_rank = {"ALIGNED": 0, "OPPOSITE": 1, "UNKNOWN": 2}
    new_alerts.sort(key=lambda x: (alignment_rank[x["btc_alignment"]], x["priority"], x["timestamp_ms"], x["symbol"]))
    cancellations.sort(key=lambda x: (x["timestamp_ms"], x["symbol"]))

    sent_count = 0
    for item in cancellations:
        try:
            telegram_send(make_cancellation(item))
            state["cancelled"].add(item["dedupe_key"])
            save_state(state)
            print(f"CANCELLED {item['symbol']} {item['direction']} {item['signal_time']} due to opposite box touch")
        except Exception as exc:
            print(f"Cancellation notice failed {item['symbol']}: {type(exc).__name__}: {exc}")

    for item in new_alerts:
        try:
            telegram_send(make_alert(item, btc))
            state["sent"].add(item["dedupe_key"])
            save_state(state)
            sent_count += 1
            print(f"ALERT priority={item['priority']} {item['symbol']} {item['direction']} RSI={_fmt_rsi(item['rsi'])} {item['signal_time']}")
        except Exception as exc:
            print(f"Telegram send failed {item['symbol']}: {type(exc).__name__}: {exc}")

    save_state(state)
    ok_count = sum(r.get("status") == "OK" for r in records)
    print(f"Finished: fetched={ok_count}/{len(alt_symbols)} candidates_in_window={candidate_count} alerts={sent_count} cancellations={len(cancellations)} errors={errors}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
