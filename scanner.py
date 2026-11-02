"""24/7 Binance USDⓈ-M USDT perpetual scanner; closed 30m candles only."""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from indicator_engine import evaluate_signals

BASE_URL = "https://fapi.binance.com"
TIMEFRAME = "30m"
INTERVAL_MS = 30 * 60 * 1000
CANDLE_LIMIT = 1000
MAX_WORKERS = 12
STATE_FILE = Path("state.json")

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Binance-USDM-SQZMOM-Reversal-SMC-Scanner/3.0"})
retry = Retry(
    total=4, connect=4, read=3, status=3, backoff_factor=0.5,
    status_forcelist=(418, 429, 500, 502, 503, 504),
    allowed_methods=frozenset(["GET"]), respect_retry_after_header=True,
)
adapter = HTTPAdapter(max_retries=retry, pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS)
SESSION.mount("https://", adapter)


def get_json(path: str, params: dict[str, Any] | None = None) -> Any:
    response = SESSION.get(BASE_URL + path, params=params, timeout=(10, 25))
    response.raise_for_status()
    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(f"Binance returned non-JSON data for {path}") from exc


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


def load_state() -> set[str]:
    if not STATE_FILE.exists():
        return set()
    try:
        return set(json.loads(STATE_FILE.read_text(encoding="utf-8")).get("sent", []))
    except (OSError, ValueError, TypeError) as exc:
        print(f"WARNING: state.json could not be read; starting with empty history: {exc}")
        return set()


def save_state(sent: set[str]) -> None:
    payload = {"sent": sorted(sent)[-15000:]}
    temp = STATE_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(STATE_FILE)


def fetch_symbols() -> list[str]:
    info = get_json("/fapi/v1/exchangeInfo")
    symbols = []
    for item in info.get("symbols", []):
        if (
            item.get("status") == "TRADING"
            and item.get("contractType") == "PERPETUAL"
            and item.get("quoteAsset") == "USDT"
        ):
            symbols.append(str(item["symbol"]))
    return sorted(set(symbols))


def fetch_closed_candles(symbol: str, server_time_ms: int) -> pd.DataFrame:
    rows = get_json("/fapi/v1/klines", {"symbol": symbol, "interval": TIMEFRAME, "limit": CANDLE_LIMIT})
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame()
    closed = [row for row in rows if int(row[6]) < server_time_ms]
    if len(closed) < 250:
        return pd.DataFrame()
    frame = pd.DataFrame({
        "timestamp": [int(row[0]) for row in closed],
        "open": [float(row[1]) for row in closed],
        "high": [float(row[2]) for row in closed],
        "low": [float(row[3]) for row in closed],
        "close": [float(row[4]) for row in closed],
        "volume": [float(row[5]) for row in closed],
    })
    frame["timestamp"] = pd.to_datetime(frame.timestamp, unit="ms", utc=True)
    return frame.set_index("timestamp").sort_index()


def evaluate_symbol(symbol: str, server_time_ms: int) -> dict[str, Any] | None:
    candles = fetch_closed_candles(symbol, server_time_ms)
    if candles.empty:
        return None
    result = evaluate_signals(candles)
    if result.empty:
        return None
    last = result.iloc[-1]
    signal = str(last.signal)
    if signal not in ("BUY", "SELL"):
        return None
    candle_time = result.index[-1]
    return {
        "symbol": symbol,
        "signal": signal,
        "timestamp_ms": int(candle_time.timestamp() * 1000),
        "candle_time": candle_time.strftime("%Y-%m-%d %H:%M UTC"),
        "close": float(last.close),
        "sqzmom": float(last.sqzmom),
        "peak_trough": float(last.sqzmom_peak_trough),
        "reversal_bottom": float(last.reversal_zone_bottom),
        "reversal_top": float(last.reversal_zone_top),
        "smc_matches": str(last.smc_matches),
        "smc_match_age": int(last.smc_match_age),
    }


def make_alert(item: dict[str, Any]) -> str:
    signal = item["signal"]
    title = "🟢 BUY / LONG" if signal == "BUY" else "🔴 SELL / SHORT"
    lines = [
        title,
        f"قرارداد: {item['symbol']} — Binance USDⓈ-M USDT-M Perpetual",
        f"تایم‌فریم: {TIMEFRAME}",
        f"قیمت بسته‌شدن: {item['close']:.10g}",
        f"SQZMOM میله میانی قله/دره: {item['peak_trough']:.10g}",
        f"SQZMOM آخرین میله: {item['sqzmom']:.10g}",
        f"Reversal Probability ({'سبز' if signal == 'BUY' else 'قرمز'}): {item['reversal_bottom']:.10g} تا {item['reversal_top']:.10g}",
        f"تأیید SMC در ۵ کندل آخر: {item['smc_matches']}",
        f"زمان کندل بسته‌شده: {item['candle_time']}",
        "فقط هشدار تحلیلی است؛ هیچ سفارش یا معامله‌ای ثبت نمی‌شود.",
    ]
    return "\n".join(lines)


def main() -> None:
    print(f"Starting Binance USDⓈ-M Futures scanner | {TIMEFRAME} | SQZMOM + Reversal Probability + SMC")
    server_time_ms = int(get_json("/fapi/v1/time")["serverTime"])
    symbols = fetch_symbols()
    sent = load_state()
    print(f"Active USDT-M perpetual contracts: {len(symbols)}")

    found: list[dict[str, Any]] = []
    checked = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(evaluate_symbol, symbol, server_time_ms): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                item = future.result()
                checked += 1
                if item:
                    key = f"{item['symbol']}|{TIMEFRAME}|{item['timestamp_ms']}|{item['signal']}"
                    if key not in sent:
                        item["dedupe_key"] = key
                        found.append(item)
            except requests.HTTPError as exc:
                errors += 1
                status = exc.response.status_code if exc.response is not None else "unknown"
                print(f"HTTP error for {symbol} (status {status}): {exc}")
                if status in (418, 429):
                    raise
            except Exception as exc:
                errors += 1
                print(f"Skipped {symbol}: {type(exc).__name__}: {exc}")

    found.sort(key=lambda x: (x["signal"], x["symbol"]))
    alerts = 0
    for item in found:
        try:
            telegram_send(make_alert(item))
            sent.add(item["dedupe_key"])
            save_state(sent)
            alerts += 1
            print(f"ALERT {item['symbol']} {item['signal']} {item['candle_time']}")
        except Exception as exc:
            print(f"Telegram send failed for {item['symbol']}: {type(exc).__name__}: {exc}")

    save_state(sent)
    print(f"Finished: checked={checked}/{len(symbols)} new_alerts={alerts} errors={errors}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
