#!/usr/bin/env python3
"""Scanner 4 - Kraken Futures, 15m B-Xtrender + Delta Reaction Zones."""
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
DELTA_SMOOTH = 3
DELTA_PIVOT_LENGTH = 12
DELTA_ROC_LENGTH = 12
DELTA_ATR_LENGTH = 14
DELTA_ATR_MULTIPLIER = 0.35
DELTA_IMPULSE_WINDOW = 100
DELTA_MAX_ZONES = 8
DELTA_MERGE_GAP_TICKS = 20.0
DELTA_CONFIRM_BARS = 6
ALT_LOOKBACK_BARS = 1
MAX_CANDLE_AGE_MS = INTERVAL_MS

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


def candle_is_stale(candle: Candle, now_ms: int | None = None) -> bool:
    """A closed candle is stale only after it has been closed for over 15 minutes."""
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    closed_at = candle.time + INTERVAL_MS
    return now_ms - closed_at > MAX_CANDLE_AGE_MS or closed_at > now_ms


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


def rsi(values: list[float], length: int) -> list[float | None]:
    """TradingView-style RSI using Wilder RMA; returns None before warm-up."""
    n = len(values)
    out: list[float | None] = [None] * n
    if n <= length:
        return out
    gains = [0.0] * n
    losses = [0.0] * n
    for i in range(1, n):
        change = float(values[i]) - float(values[i - 1])
        gains[i] = max(change, 0.0)
        losses[i] = max(-change, 0.0)
    avg_gain = sum(gains[1:length + 1]) / length
    avg_loss = sum(losses[1:length + 1]) / length
    def value(g: float, loss: float) -> float:
        if loss == 0:
            return 100.0 if g > 0 else 50.0
        if g == 0:
            return 0.0
        rs = g / loss
        return 100.0 - 100.0 / (1.0 + rs)
    out[length] = value(avg_gain, avg_loss)
    for i in range(length + 1, n):
        avg_gain = (avg_gain * (length - 1) + gains[i]) / length
        avg_loss = (avg_loss * (length - 1) + losses[i]) / length
        out[i] = value(avg_gain, avg_loss)
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
    """Find a transition on either of the last two steps in a 3-bar window.

    The transition on the most recent step takes precedence if both steps
    changed. RED->GREEN is BUY; GREEN->RED is SELL.
    """
    if len(colors) < 3:
        return None
    a, b, c = colors[-3], colors[-2], colors[-1]
    for previous, latest in ((b, c), (a, b)):
        if previous == "RED" and latest == "GREEN":
            return "BUY"
        if previous == "GREEN" and latest == "RED":
            return "SELL"
    return None


@dataclass
class DeltaZone:
    top: float
    bottom: float
    is_support: bool
    created_index: int
    last_touch_index: int = -1


def _ema_series(values: list[float], length: int) -> list[float]:
    if not values:
        return []
    alpha = 2.0 / (length + 1.0)
    result = [float(values[0])]
    for value in values[1:]:
        result.append(alpha * float(value) + (1.0 - alpha) * result[-1])
    return result


def _rma_series(values: list[float], length: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    if len(values) < length:
        return out
    prev = sum(values[:length]) / length
    out[length - 1] = prev
    for i in range(length, len(values)):
        prev = (prev * (length - 1) + values[i]) / length
        out[i] = prev
    return out


def _atr_series(candles: list[Candle], length: int) -> list[float | None]:
    tr: list[float] = []
    for i, c in enumerate(candles):
        if i == 0:
            tr.append(c.high - c.low)
        else:
            prev_close = candles[i - 1].close
            tr.append(max(c.high - c.low, abs(c.high - prev_close), abs(c.low - prev_close)))
    return _rma_series(tr, length)


def delta_zone_confirmation(candles: list[Candle], direction: str,
                            lookback: int = DELTA_CONFIRM_BARS,
                            tick_size: float = 0.01) -> bool:
    """Replay Delta Reaction Zones [BOSWaves] using its Pine construction rules.

    Green lower-pivot support confirms BUY; red upper-pivot resistance confirms
    SELL. Text labels such as BUY FLOW / SELL FLOW are intentionally ignored.
    Only an active zone at the end of the replay can confirm, and at least one
    of the last `lookback` closed candles must overlap it.
    """
    n = len(candles)
    if n < max(DELTA_PIVOT_LENGTH * 2 + 3, DELTA_ATR_LENGTH + lookback):
        return False

    raw_delta = [((1.0 if c.close > c.open else -1.0 if c.close < c.open else 0.0) * c.volume)
                 for c in candles]
    smooth_delta = _ema_series(raw_delta, DELTA_SMOOTH)
    cumulative: list[float] = []
    running = 0.0
    for value in smooth_delta:
        running += value
        cumulative.append(running)

    # The supplied source defaults to Cumulative Delta (not RoC).
    pivot_values = cumulative
    atr = _atr_series(candles, DELTA_ATR_LENGTH)
    positive_flow = [max(x, 0.0) for x in smooth_delta]
    negative_flow = [max(-x, 0.0) for x in smooth_delta]
    impulse_start = max(0, n - DELTA_IMPULSE_WINDOW)
    # Flow statistics are calculated at the pivot candle, matching Pine's [pivot_length] use.

    zones: list[DeltaZone] = []
    first_recent = max(0, n - lookback)
    p = DELTA_PIVOT_LENGTH
    gap = DELTA_MERGE_GAP_TICKS * max(tick_size, 0.0)

    for i in range(n):
        # A pivot centered at i-p becomes knowable only on candle i.
        center = i - p
        if center >= p and i < n:
            left = pivot_values[center - p:center]
            right = pivot_values[center + 1:center + p + 1]
            if len(left) == p and len(right) == p:
                val = pivot_values[center]
                is_upper = all(val > x for x in left) and all(val >= x for x in right)
                is_lower = all(val < x for x in left) and all(val <= x for x in right)
                for support, found in ((False, is_upper), (True, is_lower)):
                    if not found or atr[center] is None:
                        continue
                    anchor = candles[center].low if support else candles[center].high
                    half = float(atr[center]) * DELTA_ATR_MULTIPLIER
                    new_top, new_bottom = anchor + half, anchor - half
                    # Same-type zones merge when overlapping or within configured tick gap.
                    merged = False
                    for z in reversed(zones):
                        if z.is_support != support:
                            continue
                        if (z.bottom - gap <= new_top + gap) and (z.top + gap >= new_bottom - gap):
                            z.top = max(z.top, new_top)
                            z.bottom = min(z.bottom, new_bottom)
                            z.created_index = min(z.created_index, center)
                            merged = True
                            break
                    if not merged:
                        zones.append(DeltaZone(new_top, new_bottom, support, center))
                    while len(zones) > DELTA_MAX_ZONES:
                        zones.pop(0)

        # Pine updates zones on every candle after creating/merging pivot zones.
        survivors: list[DeltaZone] = []
        for z in zones:
            c = candles[i]
            touched = c.high >= z.bottom and c.low <= z.top
            breached = (c.close < z.bottom) if z.is_support else (c.close > z.top)
            if breached:
                continue
            if touched:
                z.last_touch_index = i
            survivors.append(z)
        zones = survivors

    for z in zones:
        if z.is_support != (direction == "BUY"):
            continue
        if z.last_touch_index >= first_recent:
            return True
    return False


def delta_reclaim_markers(candles: list[Candle], tick_size: float = 0.01,
                          lookback: int = 3) -> tuple[bool, bool]:
    """Return original Delta Reaction Zones Support Reclaim / Resistance Re-enter markers.

    This mirrors the Pine marker conditions (supReclaim/resReenter), not the
    colored-zone touch confirmation used for altcoin signals. The zone replay
    follows the indicator order: create/merge, cap oldest zones, remove breached
    zones, then evaluate marker crossings against each surviving zone_mid.
    """
    n = len(candles)
    if n < max(DELTA_PIVOT_LENGTH * 2 + 3, DELTA_ATR_LENGTH + lookback, 2):
        return False, False

    raw_delta = [((1.0 if c.close > c.open else -1.0 if c.close < c.open else 0.0) * c.volume)
                 for c in candles]
    smooth_delta = _ema_series(raw_delta, DELTA_SMOOTH)
    cumulative: list[float] = []
    running = 0.0
    for value in smooth_delta:
        running += value
        cumulative.append(running)

    pivot_values = cumulative
    atr = _atr_series(candles, DELTA_ATR_LENGTH)
    zones: list[DeltaZone] = []
    p = DELTA_PIVOT_LENGTH
    gap = DELTA_MERGE_GAP_TICKS * max(tick_size, 0.0)
    first_recent = max(0, n - lookback)
    support_reclaim = False
    resistance_reenter = False

    for i in range(n):
        # Preserve existing zone_mid values for Pine's historical crossover tests.
        prior_midpoints = {id(z): (z.top + z.bottom) / 2.0 for z in zones}

        # A pivot centered at i-p becomes knowable only on candle i.
        center = i - p
        if center >= p:
            left = pivot_values[center - p:center]
            right = pivot_values[center + 1:center + p + 1]
            if len(left) == p and len(right) == p:
                val = pivot_values[center]
                is_upper = all(val > x for x in left) and all(val >= x for x in right)
                is_lower = all(val < x for x in left) and all(val <= x for x in right)
                for support, found in ((False, is_upper), (True, is_lower)):
                    if not found or atr[center] is None:
                        continue
                    anchor = candles[center].low if support else candles[center].high
                    half = float(atr[center]) * DELTA_ATR_MULTIPLIER
                    new_top, new_bottom = anchor + half, anchor - half
                    merged = False
                    for z in reversed(zones):
                        if z.is_support != support:
                            continue
                        if (z.bottom - gap <= new_top + gap) and (z.top + gap >= new_bottom - gap):
                            z.top = max(z.top, new_top)
                            z.bottom = min(z.bottom, new_bottom)
                            z.created_index = min(z.created_index, center)
                            merged = True
                            break
                    if not merged:
                        zones.append(DeltaZone(new_top, new_bottom, support, center))
                    while len(zones) > DELTA_MAX_ZONES:
                        zones.pop(0)

        # Match Pine's zone-update/breach loop before its marker loop.
        survivors: list[DeltaZone] = []
        candle = candles[i]
        for z in zones:
            breached = (candle.close < z.bottom) if z.is_support else (candle.close > z.top)
            if not breached:
                survivors.append(z)
        zones = survivors

        if i < first_recent or i == 0:
            continue
        previous_close = candles[i - 1].close
        current_close = candle.close
        for z in zones:
            zone_mid = (z.top + z.bottom) / 2.0
            prior_mid = prior_midpoints.get(id(z))
            if prior_mid is None:
                continue  # a zone created on this candle has no prior series value
            if z.is_support:
                # Pine: wasBelow AND ta.crossover(close, z.zone_mid).
                if (previous_close < zone_mid and previous_close <= prior_mid
                        and current_close > zone_mid):
                    support_reclaim = True
            else:
                # Pine: wasAbove AND ta.crossunder(close, z.zone_mid).
                if (previous_close > zone_mid and previous_close >= prior_mid
                        and current_close < zone_mid):
                    resistance_reenter = True
        if support_reclaim and resistance_reenter:
            break

    return support_reclaim, resistance_reenter


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


def ticker_change_1h(candles: list[Candle]) -> float | None:
    """Closed-candle 1h change: latest closed 15m close vs close four bars earlier."""
    if len(candles) < 5:
        return None
    reference = candles[-5].close
    latest = candles[-1].close
    if reference == 0:
        return None
    return (latest / reference - 1.0) * 100.0


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {"last_signal_candle_by_symbol": {}, "schema_version": 4}
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        state.setdefault("last_signal_candle_by_symbol", {})
        state["schema_version"] = 4
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


def fmt_color(color: str | None) -> str:
    return color if color in ("GREEN", "RED") else "N/A"


def build_report_messages(
    btc_signal: dict | None,
    btc_colors: list[str | None],
    btc_markers: tuple[bool, bool],
    signals: list[dict],
) -> list[str]:
    """Build the English report; BTC is informational and never gates altcoins."""
    messages: list[str] = []

    # A confirmed BTC signal, when present, is the first report line.
    if btc_signal is not None:
        dot = "🟢" if btc_signal["direction"] == "BUY" else "🔴"
        messages.append(
            f"{dot} {btc_signal['symbol']} | 24h: {fmt_pct(btc_signal['change24h'])} | "
            f"1h: {fmt_pct(btc_signal['change1h'])}"
        )

    # Always report BTC's two most recent CLOSED 15m B-Xtrender Color readings.
    last_two = list(btc_colors[-2:])
    if len(last_two) < 2:
        last_two = [None] * (2 - len(last_two)) + last_two
    messages.append(
        "BTC B-Xtrender Color (last 2 closed 15m candles): "
        f"{fmt_color(last_two[0])} → {fmt_color(last_two[1])}"
    )

    # Display only the original Delta Reaction Zones markers from BTC's last
    # three CLOSED 15m candles. S/R are never derived from colored box touches.
    support_reclaim, resistance_reenter = btc_markers
    marker_letters = ("S" if support_reclaim else "") + (" R" if resistance_reenter else "")
    if marker_letters:
        messages.append(
            "BTC Delta Reaction Zones (last 3 closed 15m candles): " + marker_letters
        )

    # Display-only ordering: strongest 24h BUY gain first, then the
    # largest 24h SELL decline first. Missing 24h values stay at the end.
    buys = sorted(
        (item for item in signals if item["direction"] == "BUY"),
        key=lambda item: (item["change24h"] is None,
                          -(item["change24h"] if item["change24h"] is not None else 0.0)),
    )
    sells = sorted(
        (item for item in signals if item["direction"] == "SELL"),
        key=lambda item: (item["change24h"] is None,
                          item["change24h"] if item["change24h"] is not None else 0.0),
    )
    for item in buys + sells:
        dot = "🟢" if item["direction"] == "BUY" else "🔴"
        messages.append(
            f"{dot} {item['symbol']} | 24h: {fmt_pct(item['change24h'])} | "
            f"1h: {fmt_pct(item['change1h'])}"
        )
    return messages


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


def analyze_market(symbol: str, tick_size: float = 0.01, is_bitcoin: bool = False) -> tuple[str, dict]:
    """Evaluate a single contract independently; BTC never gates other symbols."""
    try:
        candles = fetch_candles(symbol)
        colors = bxt_color_series(candles)
        latest_two_colors = colors[-2:] if len(colors) >= 2 else [None, None]
        btc_markers = delta_reclaim_markers(candles, tick_size=tick_size, lookback=3) if is_bitcoin else (False, False)
        marker_data = {"btc_markers": btc_markers} if is_bitcoin else {}
        if len(candles) < 80:
            return symbol, {"reason": "insufficient_history", "candle_count": len(candles),
                            "latest_colors": latest_two_colors, **marker_data}
        if candle_is_stale(candles[-1]):
            return symbol, {"reason": "stale_candles", "candle_time": candles[-1].time,
                            "latest_colors": latest_two_colors, **marker_data}
        direction = bxt_color_change(colors)
        if direction is None:
            return symbol, {"reason": "no_matching_color_change", "latest_colors": latest_two_colors, **marker_data}
        # BTC bypasses Delta confirmation; every other market keeps the original rule.
        if not is_bitcoin and not delta_zone_confirmation(
            candles, direction, lookback=DELTA_CONFIRM_BARS, tick_size=tick_size
        ):
            return symbol, {"reason": "no_valid_matching_delta_zone_touch", "direction": direction,
                            "latest_colors": latest_two_colors, **marker_data}
        return symbol, {
            "reason": "signal",
            "direction": direction,
            "candle_time": candles[-1].time,
            "change1h": ticker_change_1h(candles),
            "latest_colors": latest_two_colors,
            **marker_data,
        }
    except Exception as exc:
        LOG.warning("Market %s skipped: %s", symbol, exc)
        return symbol, {"reason": "fetch_or_analysis_error", "error": str(exc)}


def run() -> int:
    state = load_state()
    instruments = get_instruments()
    tickers = get_tickers()
    symbols = sorted({str(x.get("symbol", "")).upper() for x in instruments if x.get("symbol")})
    btc_symbol = pick_btc_symbol(instruments)
    instrument_by_symbol = {str(x.get("symbol", "")).upper(): x for x in instruments}

    def market_tick_size(symbol: str) -> float:
        item = instrument_by_symbol.get(symbol.upper(), {})
        for key in ("tickSize", "tick_size", "ticksize", "priceIncrement"):
            try:
                value = float(item.get(key))
                if value > 0:
                    return value
            except (TypeError, ValueError):
                pass
        return 0.01

    diagnostics = {
        "total_markets": len(symbols), "signals": 0,
        "insufficient_history": 0, "no_matching_color_change": 0,
        "no_valid_matching_delta_zone_touch": 0, "stale_candles": 0,
        "duplicates": 0, "fetch_or_analysis_error": 0,
    }
    found: list[dict] = []
    btc_colors: list[str | None] = [None, None]
    btc_markers: tuple[bool, bool] = (False, False)
    btc_signal: dict | None = None
    workers = max(2, min(12, int(os.getenv("SCAN_WORKERS", "8"))))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                analyze_market,
                symbol,
                market_tick_size(symbol),
                symbol == btc_symbol,
            ): symbol
            for symbol in symbols
        }
        for future in as_completed(futures):
            symbol, data = future.result()
            if symbol == btc_symbol:
                colors = data.get("latest_colors", [None, None])
                if isinstance(colors, list) and len(colors) >= 2:
                    btc_colors = colors[-2:]
                markers = data.get("btc_markers", (False, False))
                if isinstance(markers, (list, tuple)) and len(markers) == 2:
                    btc_markers = (bool(markers[0]), bool(markers[1]))
            reason = data.get("reason", "fetch_or_analysis_error")
            if reason != "signal":
                diagnostics[reason] = diagnostics.get(reason, 0) + 1
                continue
            candle_time = int(data["candle_time"])
            collection_now_ms = int(time.time() * 1000)
            if collection_now_ms - (candle_time + INTERVAL_MS) > MAX_CANDLE_AGE_MS or candle_time + INTERVAL_MS > collection_now_ms:
                diagnostics["stale_candles"] += 1
                continue
            previous_seen = state["last_signal_candle_by_symbol"].get(symbol)
            if previous_seen == candle_time:
                diagnostics["duplicates"] += 1
                continue
            signal_item = {"symbol": symbol, **data, "change24h": ticker_change(tickers.get(symbol))}
            if symbol == btc_symbol:
                btc_signal = signal_item
            else:
                found.append(signal_item)

    def within_direction(items: list[dict], direction: str) -> list[dict]:
        selected = [x for x in items if x["direction"] == direction]
        if direction == "BUY":
            selected.sort(key=lambda x: x["change24h"] if x["change24h"] is not None else float("-inf"), reverse=True)
        else:
            selected.sort(key=lambda x: x["change24h"] if x["change24h"] is not None else float("inf"))
        return selected

    # BTC is a report-only lead item: show its confirmed signal first when present,
    # but never use BTC to authorize, block, or reorder the independent altcoin scan.
    ordered = within_direction(found, "BUY") + within_direction(found, "SELL")
    report_ordered = ([btc_signal] if btc_signal is not None else []) + ordered
    for item in report_ordered:
        state["last_signal_candle_by_symbol"][item["symbol"]] = int(item["candle_time"])

    messages = build_report_messages(btc_signal, btc_colors, btc_markers, found)

    diagnostics["signals"] = len(report_ordered)
    LOG.info("SCAN_DIAGNOSTICS workers=%s %s", workers,
             " ".join(f"{key}={value}" for key, value in diagnostics.items()))
    if not report_ordered:
        messages.append("No confirmed signals in this scan.")
    header = "Scanner 4 | Kraken Futures | 15m"
    send_telegram("\n".join([header] + messages))
    save_state(state)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(run())
    except Exception:
        LOG.exception("Scanner 4 failed")
        sys.exit(1)
