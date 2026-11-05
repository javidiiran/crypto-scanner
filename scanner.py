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
OB_CONFIRM_BARS = 6
ALT_LOOKBACK_BARS = 3
BTC_SCAN_BARS = 4
MAX_CANDLE_LAG_MS = INTERVAL_MS

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


def bxt_color_change(colors: list[str | None]) -> str | None:
    """A signal is a color transition on the latest closed candle."""
    if len(colors) < 2:
        return None
    previous, latest = colors[-2], colors[-1]
    # Locked mapping: dark green starts SELL; dark red starts BUY.
    if previous == "GREEN" and latest == "RED":
        return "BUY"
    if previous == "RED" and latest == "GREEN":
        return "SELL"
    return None


def recent_bxt_change(candles: list[Candle], colors: list[str | None], direction: str) -> int | None:
    """Return the index of the latest matching color-change candle in the last 3 bars."""
    if len(candles) < ALT_LOOKBACK_BARS or len(colors) != len(candles):
        return None
    # BUY starts when the curve turns dark red; SELL starts when it turns dark green.
    desired = "RED" if direction == "BUY" else "GREEN"
    opposite = "GREEN" if direction == "BUY" else "RED"
    first = max(1, len(colors) - ALT_LOOKBACK_BARS)
    for i in range(len(colors) - 1, first - 1, -1):
        if colors[i - 1] == opposite and colors[i] == desired:
            return i
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
            # Pine creates the box on candle i, but its retest/update loop has
            # already run for that candle. It first becomes touch-eligible on i+1.
            bull.append(OB(top, bottom, i, "BUY"))
            if len(bull) > OB_MAX_PER_SIDE:
                bull = bull[len(bull) - OB_MAX_PER_SIDE:]
        if bear_ob:
            top, bottom = prev.high, prev.low
            bear = [ob for ob in bear if not (top >= ob.bottom and bottom <= ob.top)]
            # Same lifecycle as the bullish block: creation candle cannot retest itself.
            bear.append(OB(top, bottom, i, "SELL"))
            if len(bear) > OB_MAX_PER_SIDE:
                bear = bear[len(bear) - OB_MAX_PER_SIDE:]
    return bull, bear


def candle_touches_ob(candle: Candle, ob: OB) -> bool:
    return candle.low <= ob.top and candle.high >= ob.bottom


def ob_confirms(candles: list[Candle], direction: str, lookback: int = OB_CONFIRM_BARS) -> bool:
    """Replay the ChartPrime OB lifecycle and detect scanner-rule overlap.

    On each closed candle, existing blocks are invalidated first, then the
    candle is checked against blocks that were already active, and only then
    are new blocks created. This mirrors the order of operations in the Pine
    source. The scanner's six-bar range-overlap rule is intentionally distinct
    from the indicator's optional directional-crossing Retest/Alert condition.
    """
    if len(candles) < lookback:
        return False
    bull: list[OB] = []
    bear: list[OB] = []
    start = len(candles) - lookback

    for i in range(1, len(candles)):
        cur, prev = candles[i], candles[i - 1]
        bull = [ob for ob in bull if cur.close >= ob.bottom]
        bear = [ob for ob in bear if cur.close <= ob.top]

        if i >= start:
            blocks = bull if direction == "BUY" else bear
            if any(candle_touches_ob(cur, ob) for ob in blocks):
                return True

        bull_ob = (prev.close < prev.open and cur.close > cur.open and
                   cur.close > prev.high and (cur.close - cur.open) >
                   (prev.high - prev.low) * OB_DISPLACEMENT)
        bear_ob = (prev.close > prev.open and cur.close < cur.open and
                   cur.close < prev.low and (cur.open - cur.close) >
                   (prev.high - prev.low) * OB_DISPLACEMENT)

        if bull_ob:
            top, bottom = prev.high, prev.low
            bull = [ob for ob in bull if not (top >= ob.bottom and bottom <= ob.top)]
            bull.append(OB(top, bottom, i, "BUY"))
            if len(bull) > OB_MAX_PER_SIDE:
                bull = bull[-OB_MAX_PER_SIDE:]
        if bear_ob:
            top, bottom = prev.high, prev.low
            bear = [ob for ob in bear if not (top >= ob.bottom and bottom <= ob.top)]
            bear.append(OB(top, bottom, i, "SELL"))
            if len(bear) > OB_MAX_PER_SIDE:
                bear = bear[-OB_MAX_PER_SIDE:]
    return False


def alt_signal_confirms(candles: list[Candle], colors: list[str | None], direction: str) -> bool:
    """A recent matching B-Xtrender transition plus any valid OB touch in last 6 bars."""
    if recent_bxt_change(candles, colors, direction) is None:
        return False
    return ob_confirms(candles, direction, lookback=OB_CONFIRM_BARS)


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

    candidate = bxt_color_change(btc_colors)
    btc_ob_confirmed = bool(candidate and ob_confirms(btc_candles, candidate, OB_CONFIRM_BARS))
    valid_signal = candidate if btc_ob_confirmed else None
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
        if active and latest_color and latest_color != ("RED" if active == "BUY" else "GREEN"):
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
            status = "BTC_OB_NOT_CONFIRMED" if candidate else "NO_BTC_COLOR_CHANGE"

    LOG.info(
        "BTC_CHECK candle=%s colors=%s candidate=%s ob_confirmed=%s status=%s",
        latest_time, "/".join(fmt_color(x) for x in (btc_colors[-3:] if len(btc_colors) >= 3 else btc_colors)),
        candidate or "NONE", btc_ob_confirmed, status
    )
    state["last_btc_candle_time"] = latest_time
    return True, valid_signal, status, two_text


def analyze_market(symbol: str, direction: str) -> tuple[str, dict]:
    """Analyze one market and return a reason even when it has no signal."""
    try:
        candles = fetch_candles(symbol)
        if len(candles) < 80:
            return symbol, {"reason": "insufficient_history", "candle_count": len(candles)}
        colors = bxt_color_series(candles)
        changed_index = recent_bxt_change(candles, colors, direction)
        if changed_index is None:
            return symbol, {"reason": "no_matching_color_change"}
        if not ob_confirms(candles, direction, lookback=OB_CONFIRM_BARS):
            return symbol, {"reason": "no_valid_matching_ob_touch"}
        rsis = rsi([c.close for c in candles], RSI_LENGTH)
        return symbol, {
            "reason": "signal",
            "direction": direction,
            "candle_time": candles[-1].time,
            "change_candle_time": candles[changed_index].time,
            "rsi": rsis[-1],
        }
    except Exception as exc:
        LOG.warning("Market %s skipped: %s", symbol, exc)
        return symbol, {"reason": "fetch_or_analysis_error", "error": str(exc)}


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
        symbols = sorted(set(symbols))
        found: list[dict] = []
        diagnostics = {
            "total_markets": len(symbols), "signals": 0, "insufficient_history": 0,
            "no_matching_color_change": 0, "no_valid_matching_ob_touch": 0,
            "stale_candles": 0, "duplicates": 0, "fetch_or_analysis_error": 0,
            "wrong_direction": 0,
        }
        workers = max(2, min(12, int(os.getenv("SCAN_WORKERS", "8"))))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(analyze_market, symbol, direction): symbol for symbol in symbols}
            for future in as_completed(futures):
                symbol, data = future.result()
                reason = data.get("reason", "fetch_or_analysis_error")
                if reason != "signal":
                    diagnostics[reason] = diagnostics.get(reason, 0) + 1
                    continue
                # Do not require exact timestamp equality: some instruments can publish
                # their latest fully closed candle a little later. Allow at most one bar lag.
                if abs(int(data["candle_time"]) - int(latest_btc.time)) > MAX_CANDLE_LAG_MS:
                    diagnostics["stale_candles"] += 1
                    continue
                if data["direction"] != direction:
                    diagnostics["wrong_direction"] += 1
                    continue
                previous_seen = state["last_signal_candle_by_symbol"].get(symbol)
                if previous_seen == data["candle_time"]:
                    diagnostics["duplicates"] += 1
                    continue
                state["last_signal_candle_by_symbol"][symbol] = data["candle_time"]
                change = ticker_change(tickers.get(symbol))
                found.append({"symbol": symbol, **data, "change24h": change})

        found.sort(key=lambda item: abs(item["change24h"]) if item["change24h"] is not None else -1, reverse=True)
        for item in found:
            messages.append(
                f"{item['symbol']} | {direction} | تغییر ۲۴ساعته: {fmt_pct(item['change24h'])} | "
                f"RSI ارز: {fmt_rsi(item['rsi'])} | RSI بیت‌کوین: {fmt_rsi(btc_rsi)} | "
                f"رنگ دو کندل آخر BTC: {two_colors}"
            )
        diagnostics["signals"] = len(found)
        LOG.info(
            "SCAN_DIAGNOSTICS direction=%s remaining_bars=%s workers=%s %s",
            direction, state.get("bars_remaining"), workers,
            " ".join(f"{key}={value}" for key, value in diagnostics.items())
        )
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
