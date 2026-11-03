"""Kraken Futures Scanner 2 — CNV signals + OB/FVG confirmation, 1h only."""
from __future__ import annotations

import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ob_sqzmom_engine import analyze_market, confirm_with_latest_zone

REST_BASE = "https://futures.kraken.com/derivatives/api/v3"
CHART_BASE = "https://futures.kraken.com/api/charts/v1"
TIMEFRAME = "1h"
INTERVAL_MS = 60 * 60 * 1000
CANDLE_LIMIT = 1000
MIN_CANDLES = 45
MAX_CANDLE_STALENESS_MS = 2 * INTERVAL_MS
MAX_WORKERS = 5
STATE_FILE = Path("state.json")
PERPETUAL_PREFIXES = ("PF_", "PI_")
BTC_SYMBOLS = ("PF_XBTUSD", "PI_XBTUSD")
VERSION = "2.2.0"

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": f"Kraken-Futures-Scanner-2-CNV-OB-FVG/{VERSION}",
    "Accept": "application/json",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
})
_retry = Retry(
    total=4, connect=4, read=3, status=3, backoff_factor=0.7,
    status_forcelist=(408, 429, 500, 502, 503, 504),
    allowed_methods=frozenset(["GET"]), respect_retry_after_header=True,
)
SESSION.mount("https://", HTTPAdapter(
    max_retries=_retry, pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS
))


def get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    response = SESSION.get(url, params=params, timeout=(10, 30))
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and (payload.get("result") == "error" or payload.get("error")):
        raise RuntimeError(f"Kraken API error: {payload}")
    return payload


def telegram_send(message: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in GitHub Secrets")
    response = SESSION.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data={"chat_id": chat_id, "text": message, "disable_web_page_preview": True},
        timeout=(10, 25),
    )
    response.raise_for_status()
    if not response.json().get("ok"):
        raise RuntimeError(f"Telegram rejected message: {response.text}")


def load_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "sent": set(),
        "btc_direction": None,
        "btc_signal_timestamp_ms": 0,
        "btc_signal_time": "—",
        "btc_signal_rsi": None,
    }
    try:
        if STATE_FILE.exists():
            raw = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            state.update({k: v for k, v in raw.items() if k != "sent"})
            state["sent"] = set(raw.get("sent", []))
    except Exception as exc:
        print(f"WARNING state reset: {type(exc).__name__}: {exc}")
    if not isinstance(state.get("sent"), set):
        state["sent"] = set(state.get("sent") or [])
    return state


def save_state(state: dict[str, Any]) -> None:
    payload = dict(state)
    payload["sent"] = sorted(state["sent"])[-20000:]
    payload["scanner"] = f"Kraken Futures Scanner 2 CNV OB-FVG RSI {VERSION}"
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)


def fetch_symbols() -> list[str]:
    payload = get_json(f"{REST_BASE}/instruments")
    instruments = payload.get("instruments", []) if isinstance(payload, dict) else []
    symbols: set[str] = set()
    for instrument in instruments:
        symbol = str(instrument.get("symbol") or instrument.get("tradeable") or "").upper()
        if not symbol.startswith(PERPETUAL_PREFIXES):
            continue
        if instrument.get("tradeable") is False or instrument.get("isExpired") is True:
            continue
        status = str(instrument.get("status", "")).lower()
        if status and status not in {"online", "trading", "active"}:
            continue
        symbols.add(symbol)
    if not symbols:
        raise RuntimeError("No tradable Kraken perpetual contracts found")
    return sorted(symbols)


def fetch_closed_candles(symbol: str, now_ms: int) -> pd.DataFrame:
    url = f"{CHART_BASE}/trade/{quote(symbol, safe='_')}/{TIMEFRAME}"
    end = now_ms // 1000
    start = end - 21 * 86400
    modes = [
        ("range", {"from": start, "to": end, "count": CANDLE_LIMIT}),
        ("latest", {"count": CANDLE_LIMIT}),
        ("default_latest", None),
    ]
    last_stamp: int | None = None
    for attempt in range(1, 4):
        for mode, params in modes:
            payload = get_json(url, params)
            rows = payload.get("candles", []) if isinstance(payload, dict) else []
            normalized: list[tuple[int, float, float, float, float, float]] = []
            for row in rows if isinstance(rows, list) else []:
                try:
                    stamp = int(row["time"] if isinstance(row, dict) else row[0])
                    stamp = stamp * 1000 if stamp < 10**12 else stamp
                    # Exclude the still-forming candle; only closed 1h bars enter any calculation.
                    if stamp + INTERVAL_MS > now_ms:
                        continue
                    if isinstance(row, dict):
                        op, hi, low, close = row["open"], row["high"], row["low"], row["close"]
                        volume = row.get("volume", 0)
                    else:
                        op, hi, low, close, volume = row[1:6]
                    normalized.append((stamp, float(op), float(hi), float(low), float(close), float(volume)))
                except (KeyError, TypeError, ValueError, IndexError):
                    continue
            if not normalized:
                continue
            data = pd.DataFrame(
                normalized, columns=["timestamp", "open", "high", "low", "close", "volume"]
            ).drop_duplicates("timestamp").sort_values("timestamp")
            last_stamp = int(data.timestamp.iloc[-1])
            stale_ms = now_ms - (last_stamp + INTERVAL_MS)
            if stale_ms <= MAX_CANDLE_STALENESS_MS:
                data["timestamp"] = pd.to_datetime(data.timestamp, unit="ms", utc=True)
                data = data.set_index("timestamp")
                if len(data) < MIN_CANDLES:
                    return pd.DataFrame()
                print(f"DATA_OK {symbol}: {mode}, candles={len(data)}, last_closed={data.index[-1]}")
                return data
            print(f"DATA_STALE {symbol}: {mode}, stale_min={stale_ms / 60000:.1f}")
        if attempt < 3:
            time.sleep(attempt)
    raise RuntimeError(f"No fresh closed Kraken {TIMEFRAME} candles for {symbol}; last={last_stamp}")


def evaluate_symbol(symbol: str, now_ms: int) -> dict[str, Any]:
    data = fetch_closed_candles(symbol, now_ms)
    if data.empty:
        return {"symbol": symbol, "status": "INSUFFICIENT_DATA"}
    analysis = analyze_market(data)
    last_index = len(data) - 1
    # A pattern is actionable only when its third candle is the latest closed 1h candle.
    latest_signal = next(
        (s for s in reversed(analysis["signals"]) if int(s["signal_index"]) == last_index),
        None,
    )
    if latest_signal is None:
        return {"symbol": symbol, "status": "NO_SIGNAL", "analysis": analysis}
    confirmation = confirm_with_latest_zone(data, latest_signal["direction"], lookback=5)
    if not confirmation["confirmed"]:
        zone = confirmation.get("zone")
        desc = (
            f"{zone['kind']} {zone['direction']} {zone['bottom']:.10g}-{zone['top']:.10g}"
            if zone else "none"
        )
        print(f"REJECT {symbol} {latest_signal['direction']}: {confirmation['reason']} latest_zone={desc}")
        return {"symbol": symbol, "status": "UNCONFIRMED", "analysis": analysis,
                "signal": latest_signal, "confirmation": confirmation}
    item = dict(latest_signal)
    item.update({
        "symbol": symbol,
        "current_rsi": analysis["latest_rsi"],
        "change_24h_pct": percent_change_24h(data),
        "confirmation": confirmation,
        "cnv_current_color": analysis["latest_cnv_color"],
    })
    return {"symbol": symbol, "status": "CONFIRMED", "analysis": analysis, "item": item}


def percent_change_24h(data: pd.DataFrame) -> float | None:
    """24h return from the latest closed 1h close versus 24 closed bars earlier."""
    if len(data) < 25:
        return None
    previous = float(data.close.iloc[-25])
    latest = float(data.close.iloc[-1])
    if not math.isfinite(previous) or not math.isfinite(latest) or previous == 0:
        return None
    return (latest / previous - 1.0) * 100.0


def sort_signals_by_priority(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """BUY first (largest positive 24h move), then SELL (largest decline)."""
    def key(item: dict[str, Any]) -> tuple[int, int, float, str]:
        direction = item.get("direction")
        change = item.get("change_24h_pct")
        valid = change is not None and math.isfinite(float(change))
        if direction == "BUY":
            return (0, 0 if valid else 1, -float(change) if valid else 0.0, str(item.get("symbol", "")))
        return (1, 0 if valid else 1, float(change) if valid else 0.0, str(item.get("symbol", "")))
    return sorted(items, key=key)


def fmt(value: Any) -> str:
    try:
        return f"{float(value):.1f}" if value is not None and math.isfinite(float(value)) else "N/A"
    except (TypeError, ValueError):
        return "N/A"


def zone_description(confirmation: dict[str, Any]) -> str:
    zone = confirmation.get("zone")
    if not zone:
        return "باکس فعال پیدا نشد"
    if zone["kind"] == "OB":
        color = "سبز" if zone["direction"] == "BUY" else "قرمز"
        kind = "Order Block"
    else:
        color = "آبی" if zone["direction"] == "BUY" else "نارنجی/زرد"
        kind = "FVG"
    return f"{kind} {color} | {zone['bottom']:.10g} تا {zone['top']:.10g}"


def make_alert(item: dict[str, Any], btc_context: dict[str, Any]) -> str:
    buy = item["direction"] == "BUY"
    title = "🟢 BUY / LONG" if buy else "🔴 SELL / SHORT"
    zone = zone_description(item["confirmation"])
    btc_signal = btc_context.get("direction") or "بدون سیگنال تأییدشده"
    btc_signal_time = btc_context.get("time", "—")
    btc_signal_rsi = fmt(btc_context.get("signal_rsi"))
    btc_colors = btc_context.get("last_two_cnv_colors") or ["N/A", "N/A"]
    color_names = {"BLUE": "آبی", "PINK": "صورتی", "N/A": "نامشخص"}
    btc_color_text = "، ".join(color_names.get(str(color), str(color)) for color in btc_colors)
    lines = [
        title,
        f"قرارداد: {item['symbol']} — Kraken Futures Perpetual",
        f"تایم‌فریم تمام محاسبات: {TIMEFRAME}",
        f"تغییر قیمت ۲۴ ساعت نماد: {fmt(item.get('change_24h_pct'))}%",
        f"تغییر قیمت ۲۴ ساعت بیت‌کوین: {fmt(btc_context.get('change_24h_pct'))}%",
        f"قیمت بسته‌شدن آخرین کندل سیگنال: {item['price']:.10g}",
        f"الگوی CNV سیگنال: {'صورتی ← آبی ← آبی' if buy else 'آبی ← صورتی ← صورتی'}",
        f"RSI(14) فعلی نماد در 1H: {fmt(item.get('current_rsi'))} (صرفاً اطلاع‌رسانی؛ بدون فیلتر)",
        f"RSI(14) فعلی بیت‌کوین در 1H: {fmt(btc_context.get('current_rsi'))} (صرفاً اطلاع‌رسانی؛ بدون فیلتر)",
        f"دو کندل بسته‌شده آخر CNV بیت‌کوین در 1H: {btc_color_text}",
        f"تأیید OB/FVG: موفق | آخرین باکس فعال: {zone}",
        "شرط تماس: حداقل یکی از پنج کندل بسته‌شده اخیر با سایه یا بدنه باکس را لمس کرده است.",
        f"آخرین سیگنال تأییدشده BTC: {btc_signal} | زمان: {btc_signal_time} | RSI زمان سیگنال: {btc_signal_rsi}",
        "جهت BTC فیلتر اجباری برای سیگنال آلت‌کوین‌ها نیست.",
    ]
    return "\n".join(lines)

def _send_confirmed_btc(btc_result: dict[str, Any], state: dict[str, Any], btc_current_rsi: float | None) -> None:
    item = btc_result["item"]
    state.update({
        "btc_direction": item["direction"],
        "btc_signal_timestamp_ms": item["timestamp_ms"],
        "btc_signal_time": item["signal_time"],
        "btc_signal_rsi": item.get("current_rsi"),
    })
    key = f"{item['symbol']}|{item['timestamp_ms']}|{item['direction']}"
    context = {
        "direction": item["direction"], "time": item["signal_time"],
        "signal_rsi": item.get("current_rsi"), "current_rsi": btc_current_rsi,
    }
    if key not in state["sent"]:
        try:
            telegram_send(make_alert(item, context))
            state["sent"].add(key)
            print(f"BTC_ALERT_SENT {item['direction']} {item['signal_time']}")
        except Exception as exc:
            print(f"BTC_TELEGRAM_ERROR: {type(exc).__name__}: {exc}")


def main() -> None:
    now_ms = int(time.time() * 1000)
    print(f"Start Kraken Futures Scanner 2 | CNV + OB/FVG + RSI info | {TIMEFRAME} | v{VERSION}")
    symbols = fetch_symbols()
    btc_symbol = next((symbol for symbol in BTC_SYMBOLS if symbol in symbols), None)
    if not btc_symbol:
        raise RuntimeError("BTC perpetual symbol missing from Kraken instruments")

    # BTC is always fetched and evaluated first. It is confirmed by the same OB/FVG rule.
    btc_result = evaluate_symbol(btc_symbol, now_ms)
    if btc_result["status"] == "INSUFFICIENT_DATA":
        raise RuntimeError("BTC closed candles unavailable; altcoin scan aborted")
    btc_analysis = btc_result["analysis"]
    btc_current_rsi = btc_analysis["latest_rsi"]
    state = load_state()
    btc_change_24h = percent_change_24h(btc_analysis["candles"])
    btc_last_two_colors = btc_analysis.get("last_two_cnv_colors", ["N/A", "N/A"])
    if btc_result["status"] == "CONFIRMED":
        btc_item = btc_result["item"]
        state.update({
            "btc_direction": btc_item["direction"],
            "btc_signal_timestamp_ms": btc_item["timestamp_ms"],
            "btc_signal_time": btc_item["signal_time"],
            "btc_signal_rsi": btc_item.get("current_rsi"),
        })
        btc_item["change_24h_pct"] = btc_change_24h
    elif btc_result["status"] == "UNCONFIRMED":
        print("BTC_SIGNAL_NOT_CONFIRMED: CNV pattern exists but latest OB/FVG rule rejected it")
        btc_item = None
    else:
        print("BTC_NO_SIGNAL: no CNV pattern on latest closed 1h candle")
        btc_item = None

    # BTC direction is context only; it does not block or redirect altcoin signals.
    btc_context = {
        "direction": state.get("btc_direction"),
        "time": state.get("btc_signal_time", "—"),
        "signal_rsi": state.get("btc_signal_rsi"),
        "current_rsi": btc_current_rsi,
        "change_24h_pct": btc_change_24h,
        "last_two_cnv_colors": btc_last_two_colors,
    }

    alt_symbols = [symbol for symbol in symbols if symbol not in BTC_SYMBOLS]
    found: list[dict[str, Any]] = []
    errors = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(evaluate_symbol, symbol, now_ms): symbol for symbol in alt_symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                result = future.result()
                if result.get("status") == "CONFIRMED":
                    found.append(result["item"])
            except requests.HTTPError as exc:
                errors += 1
                status = exc.response.status_code if exc.response is not None else "unknown"
                print(f"HTTP_ERROR {symbol} status={status}: {exc}")
                if status in (418, 429):
                    raise
            except Exception as exc:
                errors += 1
                print(f"SKIP {symbol}: {type(exc).__name__}: {exc}")

    # Combine BTC and altcoin alerts, then enforce BUY-first / SELL-second priority.
    if btc_item is not None:
        found.append(btc_item)
    ordered = sort_signals_by_priority(found)
    sent_count = 0
    for item in ordered:
        key = f"{item['symbol']}|{item['timestamp_ms']}|{item['direction']}"
        if key in state["sent"]:
            continue
        try:
            telegram_send(make_alert(item, btc_context))
            state["sent"].add(key)
            save_state(state)
            sent_count += 1
            print(
                f"ALERT {item['symbol']} {item['direction']} "
                f"24h={fmt(item.get('change_24h_pct'))}% "
                f"RSI={fmt(item.get('current_rsi'))} {item['signal_time']}"
            )
        except Exception as exc:
            print(f"TELEGRAM_ERROR {item['symbol']}: {type(exc).__name__}: {exc}")

    save_state(state)
    print(
        f"Finished: perpetuals={len(symbols)} altcoins={len(alt_symbols)} "
        f"confirmed_alerts_sent={sent_count} errors={errors}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
