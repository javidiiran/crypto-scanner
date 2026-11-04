#!/usr/bin/env python3
"""Scanner 4 - Kraken Futures, 15m B-Xtrender + Power Order Blocks."""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

BASE = "https://futures.kraken.com"
CHARTS = f"{BASE}/api/charts/v1"
INSTRUMENTS_URL = f"{BASE}/derivatives/api/v3/instruments"
TICKERS_URL = f"{BASE}/derivatives/api/v3/tickers"
INTERVAL_MS = 15 * 60 * 1000
CANDLE_LIMIT = 220
STATE_PATH = Path("state.json")

# Locked strategy settings
SHORT_L1, SHORT_L2, SHORT_L3 = 5, 20, 15
RSI_LENGTH = 14
OB_DISPLACEMENT = 0.5
OB_MAX_PER_SIDE = 10
OB_LOOKBACK = 100
OB_CONFIRM_BARS = 4
BTC_SCAN_BARS = 4

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S UTC",
)
LOG = logging.getLogger("scanner4")

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Scanner4/1.0", "Accept": "application/json"})


@dataclass
class Candle:
    time: int  # candle open time, milliseconds
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


def request_json(url: str, params: dict | None = None, attempts: int = 4) -> dict:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = SESSION.get(url, params=params, timeout=25)
            if response.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {response.status_code}")
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and payload.get("result") == "error":
                raise RuntimeError(str(payload))
            return payload
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(1.0 + attempt * 1.5)
    raise RuntimeError(f"Request failed: {url}: {last_error}")


def get_instruments() -> list[dict]:
    payload = request_json(INSTRUMENTS_URL)
    instruments = payload.get("instruments", [])
    tradable = []
    for item in instruments:
        if not item.get("tradeable", False):
            continue
        symbol = str(item.get("symbol", ""))
        typ = str(item.get("type", "")).lower()
        if not symbol or typ not in ("futures_inverse", "futures_vanilla", "flexible_futures"):
            continue
        if item.get("suspended") is True:
            continue
        tradable.append(item)
    if not tradable:
        raise RuntimeError("Kraken returned no tradable futures instruments; API schema or market status may have changed.")
    return tradable


def get_tickers() -> dict[str, dict]:
    payload = request_json(TICKERS_URL)
    rows = payload.get("tickers", [])
    return {str(row.get("symbol", "")).upper(): row for row in rows if row.get("symbol")}


def pick_btc_symbol(instruments: list[dict]) -> str:
    symbols = {str(x["symbol"]).upper() for x in instruments}
    for preferred in ("PF_XBTUSD", "PI_XBTUSD", "FI_XBTUSD", "FF_XBTUSD"):
        if preferred in symbols:
            return preferred
    candidates = [s for s in symbols if "XBT" in s and (s.startswith("PF_") or s.startswith("PI_") or s.startswith("FI_") or s.startswith("FF_"))]
    if not candidates:
        candidates = [s for s in symbols if "BTC" in s]
    if not candidates:
        raise RuntimeError("Could not identify a tradable BTC futures contract on Kraken.")
    return sorted(candidates)[0]


def fetch_candles(symbol: str) -> list[Candle]:
    url = f"{CHARTS}/trade/{symbol}/15m"
    payload = request_json(url, params={"count": CANDLE_LIMIT})
    rows = payload.get("candles", [])
    candles: list[Candle] = []
    for row in rows:
        try:
            candles.append(Candle(
                time=int(row["time"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 0) or 0),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    candles.sort(key=lambda c: c.time)
    # Exclude the currently forming 15-minute candle. All decisions use closed bars only.
    now_ms = int(time.time() * 1000)
    candles = [c for c in candles if c.time + INTERVAL_MS <= now_ms]
    # Remove duplicate timestamps, keeping the latest row.
    unique: dict[int, Candle] = {c.time: c for c in candles}
    return [unique[k] for k in sorted(unique)]


def ema(values: list[float], length: int) -> list[float]:
    if not values:
        return []
    alpha = 2.0 / (length + 1.0)
    out = [values[0]]
    for value in values[1:]:
        out.append(alpha * value + (1.0 - alpha) * out[-1])
    return out


def ema_nullable(values: list[float | None], length: int) -> list[float | None]:
    if not values:
        return []
    alpha = 2.0 / (length + 1.0)
    out: list[float | None] = [None] * len(values)
    previous: float | None = None
    for i, value in enumerate(values):
        if value is None:
            continue
        previous = float(value) if previous is None else alpha * float(value) + (1.0 - alpha) * previous
        out[i] = previous
    return out


def rma(values: list[float | None], length: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    valid = [i for i, v in enumerate(values) if v is not None]
    if len(valid) < length:
        return out
    # Wilder's RMA is seeded by the SMA of the first length non-na values.
    seed_indices = valid[:length]
    seed_i = seed_indices[-1]
    prev = sum(float(values[i]) for i in seed_indices) / length
    out[seed_i] = prev
    valid_set = set(valid)
    for i in range(seed_i + 1, len(values)):
        if i in valid_set:
            prev = (prev * (length - 1) + float(values[i])) / length
            out[i] = prev
        else:
            out[i] = prev
    return out


def rsi(values: list[float], length: int = RSI_LENGTH) -> list[float | None]:
    if len(values) < 2:
        return [None] * len(values)
    changes: list[float | None] = [None]
    gains: list[float | None] = [None]
    losses: list[float | None] = [None]
    for i in range(1, len(values)):
        change = values[i] - values[i - 1]
        changes.append(change)
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    avg_gain = rma(gains, length)
    avg_loss = rma(losses, length)
    out: list[float | None] = [None] * len(values)
    for i, (g, l) in enumerate(zip(avg_gain, avg_loss)):
        if g is None or l is None:
            continue
        if l == 0:
            out[i] = 100.0 if g > 0 else 50.0
        elif g == 0:
            out[i] = 0.0
        else:
            out[i] = 100.0 - (100.0 / (1.0 + g / l))
    return out


def bxt_color_series(candles: list[Candle]) -> list[str | None]:
    closes = [c.close for c in candles]
    if not closes:
        return []
    ema_fast = ema(closes, SHORT_L1)
    ema_slow = ema(closes, SHORT_L2)
    spread = [a - b for a, b in zip(ema_fast, ema_slow)]
    spread_rsi = rsi(spread, SHORT_L3)
    short_x = [None if v is None else v - 50.0 for v in spread_rsi]
    # Pine t3(src, 5), b=0.7: six cascaded EMAs and the original coefficients.
    stages = []
    current: list[float | None] = list(short_x)
    for _ in range(6):
        current = ema_nullable(current, 5)
        stages.append(current)
    b = 0.7
    c1 = -b * b * b
    c2 = 3 * b * b + 3 * b * b * b
    c3 = -6 * b * b - 3 * b - 3 * b * b * b
    c4 = 1 + 3 * b + b * b * b + 3 * b * b
    # Pine uses xe6, xe5, xe4, xe3 in this order.
    t3_values: list[float | None] = []
    for i in range(len(candles)):
        vals = [stages[j][i] for j in (5, 4, 3, 2)]
        if any(v is None for v in vals):
            t3_values.append(None)
        else:
            t3_values.append(c1 * float(vals[0]) + c2 * float(vals[1]) + c3 * float(vals[2]) + c4 * float(vals[3]))
    colors: list[str | None] = [None] * len(candles)
    for i in range(1, len(candles)):
        if t3_values[i] is None or t3_values[i - 1] is None:
            continue
        colors[i] = "GREEN" if t3_values[i] > t3_values[i - 1] else "RED"
    return colors


def bxt_signal(candles: list[Candle], colors: list[str | None]) -> str | None:
    """Three latest closed bars: prior opposite, then two bars in new direction."""
    if len(colors) < 3:
        return None
    a, b, c = colors[-3:]
    if a == "RED" and b == "GREEN" and c == "GREEN":
        return "BUY"
    if a == "GREEN" and b == "RED" and c == "RED":
        return "SELL"
    return None


@dataclass
class OB:
    top: float
    bottom: float
    birth_index: int
    direction: str


def active_order_blocks(candles: list[Candle]) -> tuple[list[OB], list[OB]]:
    """Reproduce Power Order Blocks creation, overlap replacement and invalidation."""
    bull: list[OB] = []
    bear: list[OB] = []
    for i in range(1, len(candles)):
        cur, prev = candles[i], candles[i - 1]
        # Existing blocks are invalidated on candle close beyond their far edge.
        bull = [ob for ob in bull if cur.close >= ob.bottom]
        bear = [ob for ob in bear if cur.close <= ob.top]

        bull_ob = (prev.close < prev.open and cur.close > cur.open and
                   cur.close > prev.high and (cur.close - cur.open) > (prev.high - prev.low) * OB_DISPLACEMENT)
        bear_ob = (prev.close > prev.open and cur.close < cur.open and
                   cur.close < prev.low and (cur.open - cur.close) > (prev.high - prev.low) * OB_DISPLACEMENT)

        if bull_ob:
            top, bottom = prev.high, prev.low
            bull = [ob for ob in bull if not (top >= ob.bottom and bottom <= ob.top)]
            bull.append(OB(top, bottom, i - 1, "BUY"))
            if len(bull) > OB_MAX_PER_SIDE:
                bull = bull[len(bull) - OB_MAX_PER_SIDE:]
        if bear_ob:
            top, bottom = prev.high, prev.low
            bear = [ob for ob in bear if not (top >= ob.bottom and bottom <= ob.top)]
            bear.append(OB(top, bottom, i - 1, "SELL"))
            if len(bear) > OB_MAX_PER_SIDE:
                bear = bear[len(bear) - OB_MAX_PER_SIDE:]
    return bull, bear


def ob_confirms(candles: list[Candle], direction: str) -> bool:
    if len(candles) < OB_CONFIRM_BARS:
        return False
    bull, bear = active_order_blocks(candles)
    blocks = bull if direction == "BUY" else bear
    recent = candles[-OB_CONFIRM_BARS:]
    # A still-valid zone confirms when at least one of the last four closed candles
    # traded into/overlapped the zone. No separate retest-label/gap filter is used.
    for ob in blocks:
        for candle in recent:
            if candle.low <= ob.top and candle.high >= ob.bottom:
                return True
    return False


def ticker_change(ticker: dict | None) -> float | None:
    if not ticker:
        return None
    value = ticker.get("change24h")
    try:
        if value is not None:
            return float(value)
        last, open24 = ticker.get("last"), ticker.get("open24h")
        if last is not None and open24 not in (None, 0):
            return (float(last) / float(open24) - 1.0) * 100.0
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    return None


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {
            "last_btc_candle_time": None,
            "active_direction": None,
            "bars_remaining": 0,
            "previous_btc_signal": None,
            "previous_btc_signal_time": None,
            "last_btc_signal": None,
            "prior_btc_signal": None,
            "last_signal_candle_by_symbol": {},
            "schema_version": 1,
        }
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        state.setdefault("last_signal_candle_by_symbol", {})
        return state
    except Exception:
        LOG.exception("Could not read state.json; refusing to overwrite corrupted state.")
        raise


def save_state(state: dict) -> None:
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATE_PATH)


def fmt_pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value:+.2f}%"


def fmt_rsi(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1f}"


def fmt_color(color: str | None) -> str:
    return {"GREEN": "سبز", "RED": "قرمز", None: "نامشخص"}.get(color, "نامشخص")


def send_telegram(text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        LOG.warning("Telegram secrets are not set; alert will only appear in the Actions log.")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    # Telegram message limit is 4096 characters. Keep whole signal lines together.
    chunks: list[str] = []
    current = ""
    for line in text.splitlines():
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > 3900 and current:
            chunks.append(current)
            current = line
        else:
            current = candidate
    if current:
        chunks.append(current)
    for chunk in chunks:
        response = SESSION.post(url, json={"chat_id": chat_id, "text": chunk, "disable_web_page_preview": True}, timeout=25)
        response.raise_for_status()
        data = response.json()
        if not data.get("ok"):
            raise RuntimeError(f"Telegram rejected message: {data}")


def previous_signal_distance(state: dict, candle_time: int) -> tuple[str | None, int | None]:
    previous_direction = state.get("previous_btc_signal")
    previous_time = state.get("previous_btc_signal_time")
    if previous_direction is None or previous_time is None:
        return None, None
    gap = max(0, int((candle_time - int(previous_time)) // INTERVAL_MS))
    return str(previous_direction), gap


def process_new_btc_bar(state: dict, btc_candles: list[Candle], btc_colors: list[str | None], btc_rsi: list[float | None], btc_ticker: dict | None) -> tuple[bool, str | None, str | None, str]:
    """Return (new_bar, new_valid_signal, status_text, two latest B-Xtrender colors)."""
    if not btc_candles:
        raise RuntimeError("No closed BTC 15m candles received.")
    latest = btc_candles[-1]
    latest_time = latest.time
    last_done = state.get("last_btc_candle_time")
    last_two = btc_colors[-2:] if len(btc_colors) >= 2 else btc_colors
    two_text = " | ".join(fmt_color(x) for x in last_two)
    if last_done is not None and latest_time <= int(last_done):
        return False, None, None, two_text

    candidate = bxt_signal(btc_candles, btc_colors)
    valid_signal = candidate if candidate and ob_confirms(btc_candles, candidate) else None
    latest_color = btc_colors[-1] if btc_colors else None

    if valid_signal:
        old_signal = state.get("last_btc_signal")
        if old_signal:
            state["prior_btc_signal"] = old_signal
        state["last_btc_signal"] = {"direction": valid_signal, "time": latest_time}
        # Retain legacy fields for readable state files and backwards compatibility.
        state["previous_btc_signal"] = valid_signal
        state["previous_btc_signal_time"] = latest_time
        state["active_direction"] = valid_signal
        state["bars_remaining"] = BTC_SCAN_BARS
        status = "NEW_BTC_SIGNAL"
    else:
        active = state.get("active_direction")
        if active and latest_color and latest_color != ("GREEN" if active == "BUY" else "RED"):
            state["active_direction"] = None
            state["bars_remaining"] = 0
            status = "STOPPED_OPPOSITE_BTC_COLOR"
        elif active and int(state.get("bars_remaining", 0)) > 0:
            state["bars_remaining"] = int(state["bars_remaining"]) - 1
            if state["bars_remaining"] <= 0:
                state["active_direction"] = None
                status = "FOUR_BAR_WINDOW_EXPIRED"
            else:
                status = "ACTIVE_WINDOW"
        else:
            state["active_direction"] = None
            state["bars_remaining"] = 0
            status = "WAITING_FOR_BTC_SIGNAL"

    state["last_btc_candle_time"] = latest_time
    return True, valid_signal, status, two_text


def analyze_market(symbol: str) -> tuple[str, dict] | None:
    try:
        candles = fetch_candles(symbol)
        if len(candles) < 80:
            return None
        colors = bxt_color_series(candles)
        direction = bxt_signal(candles, colors)
        if not direction or not ob_confirms(candles, direction):
            return None
        rsis = rsi([c.close for c in candles], RSI_LENGTH)
        return symbol, {
            "direction": direction,
            "candle_time": candles[-1].time,
            "rsi": rsis[-1],
        }
    except Exception as exc:
        LOG.warning("Market %s skipped: %s", symbol, exc)
        return None


def run() -> int:
    state = load_state()
    instruments = get_instruments()
    tickers = get_tickers()
    btc_symbol = pick_btc_symbol(instruments)
    btc_candles = fetch_candles(btc_symbol)
    if len(btc_candles) < 80:
        raise RuntimeError(f"Not enough BTC history for B-Xtrender/RSI: {len(btc_candles)} candles")
    btc_closes = [c.close for c in btc_candles]
    btc_colors = bxt_color_series(btc_candles)
    btc_rsis = rsi(btc_closes, RSI_LENGTH)
    btc_ticker = tickers.get(btc_symbol.upper())

    new_bar, btc_signal, status, two_colors = process_new_btc_bar(state, btc_candles, btc_colors, btc_rsis, btc_ticker)
    if not new_bar:
        LOG.info("No new closed BTC 15m candle. Nothing to scan.")
        return 0

    latest_btc = btc_candles[-1]
    btc_rsi = btc_rsis[-1]
    btc_change = ticker_change(btc_ticker)
    latest_two_colors = btc_colors[-2:] if len(btc_colors) >= 2 else btc_colors
    direction = state.get("active_direction")
    messages: list[str] = []
    btc_line: str | None = None
    last_signal_info = state.get("last_btc_signal")
    prior_signal_info = state.get("prior_btc_signal")
    if btc_signal or direction:
        if last_signal_info:
            current_direction = str(last_signal_info.get("direction", direction or ""))
            current_time = int(last_signal_info.get("time", latest_btc.time))
            if prior_signal_info:
                prior_direction = str(prior_signal_info.get("direction", ""))
                prior_time = int(prior_signal_info.get("time", current_time))
                gap = max(0, int((current_time - prior_time) // INTERVAL_MS))
                btc_line = (
                    f"BTC | سیگنال فعلی: {current_direction} | سیگنال قبلی: {prior_direction} | فاصله: {gap} کندل | "
                    f"تغییر ۲۴ساعته: {fmt_pct(btc_change)} | RSI بیت‌کوین: {fmt_rsi(btc_rsi)} | "
                    f"رنگ دو کندل آخر B-Xtrender: {two_colors}"
                )
            else:
                btc_line = (
                    f"BTC | سیگنال فعلی: {current_direction} | تغییر ۲۴ساعته: {fmt_pct(btc_change)} | "
                    f"RSI بیت‌کوین: {fmt_rsi(btc_rsi)} | رنگ دو کندل آخر B-Xtrender: {two_colors}"
                )

    if direction and int(state.get("bars_remaining", 0)) > 0:
        symbols = []
        for item in instruments:
            symbol = str(item.get("symbol", "")).upper()
            if symbol and symbol != btc_symbol.upper():
                symbols.append(symbol)
        # Keep unique market symbols and scan all tradable futures contracts.
        symbols = sorted(set(symbols))
        found: list[dict] = []
        workers = max(2, min(8, int(os.getenv("SCAN_WORKERS", "6"))))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(analyze_market, symbol): symbol for symbol in symbols}
            for future in as_completed(futures):
                result = future.result()
                if not result:
                    continue
                symbol, data = result
                if data["direction"] != direction:
                    continue
                if data["candle_time"] != latest_btc.time:
                    # Avoid mixing signals from markets whose latest closed candle
                    # is not aligned with the BTC scan candle.
                    continue
                previous_seen = state["last_signal_candle_by_symbol"].get(symbol)
                if previous_seen == data["candle_time"]:
                    continue
                state["last_signal_candle_by_symbol"][symbol] = data["candle_time"]
                change = ticker_change(tickers.get(symbol))
                found.append({"symbol": symbol, **data, "change24h": change})

        # Highest absolute 24h percentage change first, as requested.
        found.sort(key=lambda item: abs(item["change24h"]) if item["change24h"] is not None else -1, reverse=True)
        for item in found:
            messages.append(
                f"{item['symbol']} | {direction} | تغییر ۲۴ساعته: {fmt_pct(item['change24h'])} | "
                f"RSI ارز: {fmt_rsi(item['rsi'])} | RSI بیت‌کوین: {fmt_rsi(btc_rsi)} | "
                f"رنگ دو کندل آخر BTC: {two_colors}"
            )
        LOG.info("BTC window %s; remaining scan bars=%s; matching signals=%s", direction, state.get("bars_remaining"), len(found))
    else:
        LOG.info("No active BTC permission window. Status=%s", status)

    # Every batch of market signals starts with BTC. On later BTC signals,
    # the BTC line includes its previous signal and the candle gap.
    if btc_line and (btc_signal or len(messages) > 0):
        messages.insert(0, btc_line)

    if messages:
        header = "اسکنر ۴ | Kraken Futures | 15m"
        if btc_signal:
            header += "\nسیگنال معتبر بیت‌کوین؛ شروع اسکن هم‌جهت"
        elif direction:
            header += f"\nادامه مجوز بیت‌کوین: {direction} | کندل‌های باقی‌مانده: {state.get('bars_remaining', 0)}"
        else:
            header += "\nدر انتظار سیگنال معتبر بیت‌کوین"
        send_telegram("\n".join([header] + messages))
    else:
        LOG.info("No alert lines for this closed candle.")

    save_state(state)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(run())
    except Exception:
        LOG.exception("Scanner 4 failed")
        sys.exit(1)
