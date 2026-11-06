"""Scanner 4 — Kraken Futures, 15-minute closed-candle signals.

Signals are generated independently for every tradable futures contract.
B-Xtrender Color is reproduced from the supplied Pine source; Delta Reaction
Zones is reproduced from its supplied Pine source/default inputs.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

# ---------------------------- Configuration ----------------------------
BASE = "https://futures.kraken.com"
CHARTS = f"{BASE}/api/charts/v1"
REST = f"{BASE}/derivatives/api/v3"
TIMEFRAME = "15m"
INTERVAL_MS = 15 * 60 * 1000
CANDLE_LIMIT = 500
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))
STATE_FILE = Path(os.getenv("STATE_FILE", "scanner4_state.json"))
REQUEST_TIMEOUT = 20

# Delta Reaction Zones supplied Pine defaults
DELTA_SMOOTH = 3
PIVOT_LENGTH = 12
IMPULSE_WINDOW = 100
ATR_LENGTH = 14
ATR_MULTIPLIER = 0.35
MAXIMUM_ZONES = 8
MERGE_GAP_TICKS = 20
CONFIRM_CANDLES = 6

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Scanner4/1.0", "Accept": "application/json"})
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("scanner4")


def get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and payload.get("result") == "error":
                raise RuntimeError(str(payload.get("error") or payload.get("errors")))
            return payload
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1.0 + attempt)
    raise RuntimeError(f"GET {url} failed: {last_error}")


def fetch_tickers() -> dict[str, dict[str, Any]]:
    payload = get_json(f"{REST}/tickers")
    return {str(x.get("symbol", "")).upper(): x for x in payload.get("tickers", []) if x.get("symbol")}


def fetch_instruments() -> list[dict[str, Any]]:
    payload = get_json(f"{REST}/instruments")
    items = payload.get("instruments", []) if isinstance(payload, dict) else []
    out = []
    for item in items:
        symbol = str(item.get("symbol", "")).upper()
        if not symbol:
            continue
        # Kraken instrument metadata uses tradeable/suspended flags. Exclude
        # indices and options; retain tradable futures contracts.
        kind = str(item.get("type", item.get("contractType", ""))).lower()
        if "option" in kind or "index" in kind:
            continue
        if item.get("tradeable") is False or item.get("suspended") is True:
            continue
        out.append(item)
    return out


def fetch_candles(symbol: str) -> pd.DataFrame:
    now_ms = int(time.time() * 1000)
    # Request a generous history for EMA/RMA warm-up and confirmed pivots.
    start = int((now_ms - CANDLE_LIMIT * INTERVAL_MS * 2) / 1000)
    payload = get_json(f"{CHARTS}/trade/{symbol}/{TIMEFRAME}", {"from": start, "to": int(now_ms / 1000)})
    rows = payload.get("candles", []) if isinstance(payload, dict) else []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in ("time", "open", "high", "low", "close", "volume"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["time", "open", "high", "low", "close", "volume"])
    df["time"] = df["time"].astype("int64")
    # Only fully closed 15-minute candles. The live candle is never evaluated.
    df = df[df["time"] + INTERVAL_MS <= now_ms].sort_values("time").drop_duplicates("time")
    if len(df) > CANDLE_LIMIT:
        df = df.iloc[-CANDLE_LIMIT:]
    if df.empty:
        return pd.DataFrame()
    return df.reset_index(drop=True)


# ------------------------- Pine-compatible math -------------------------
def pine_ema(series: pd.Series, length: int) -> pd.Series:
    """TradingView ta.ema recurrence, seeded from the first non-NaN value."""
    x = pd.to_numeric(series, errors="coerce").astype(float)
    out = pd.Series(np.nan, index=x.index, dtype=float)
    alpha = 2.0 / (length + 1.0)
    prev = np.nan
    for i, value in enumerate(x.to_numpy()):
        if not np.isfinite(value):
            continue
        prev = value if not np.isfinite(prev) else alpha * value + (1.0 - alpha) * prev
        out.iloc[i] = prev
    return out


def pine_rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder RMA used by Pine RSI (initial seed is SMA of first length values)."""
    x = pd.to_numeric(series, errors="coerce").astype(float)
    out = pd.Series(np.nan, index=x.index, dtype=float)
    vals = x.to_numpy()
    seed_sum = 0.0
    seed_count = 0
    prev = np.nan
    seeded = False
    for i, value in enumerate(vals):
        if not np.isfinite(value):
            continue
        if not seeded:
            seed_sum += value
            seed_count += 1
            if seed_count == length:
                prev = seed_sum / length
                out.iloc[i] = prev
                seeded = True
        else:
            prev = (prev * (length - 1) + value) / length
            out.iloc[i] = prev
    return out


def pine_rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.astype(float).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pine_rma(gain, length)
    avg_loss = pine_rma(loss, length)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100.0 - (100.0 / (1.0 + rs))
    result = result.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    result = result.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    result = result.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    return result


def pine_t3(series: pd.Series, length: int = 5, b: float = 0.7) -> pd.Series:
    e1 = pine_ema(series, length)
    e2 = pine_ema(e1, length)
    e3 = pine_ema(e2, length)
    e4 = pine_ema(e3, length)
    e5 = pine_ema(e4, length)
    e6 = pine_ema(e5, length)
    c1 = -(b ** 3)
    c2 = 3 * b * b + 3 * b ** 3
    c3 = -6 * b * b - 3 * b - 3 * b ** 3
    c4 = 1 + 3 * b + b ** 3 + 3 * b * b
    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


def bxtrender_colors(df: pd.DataFrame) -> pd.Series:
    """Color of ONLY Pine plot 'B-Xtrender Color' (not histogram/shadow/long line)."""
    close = df["close"].astype(float)
    short_l1 = pine_ema(close, 5)
    short_l2 = pine_ema(close, 20)
    short_term = pine_rsi(short_l1 - short_l2, 15) - 50.0
    color_line = pine_t3(short_term, 5, 0.7)
    rising = color_line > color_line.shift(1)
    # Pine source: rising -> lime/green, otherwise red.
    return pd.Series(np.where(rising, "GREEN", "RED"), index=df.index, dtype="object").where(color_line.notna())


def detect_bxtrender_signal(colors: pd.Series) -> tuple[str | None, int | None]:
    """Check transitions on candle 2 OR candle 3 of the latest 3 closed bars.

    Return the most recent transition in that two-transition window. This
    avoids issuing an older direction if both transitions occur in the window.
    """
    if len(colors) < 4:
        return None, None
    c = colors.iloc[-4:].tolist()  # predecessor + the latest three closed candles
    transitions: list[tuple[str, int]] = []
    for j in (2, 3):  # second and third candle of the three-candle window
        previous, current = c[j - 1], c[j]
        if previous in ("GREEN", "RED") and current in ("GREEN", "RED") and previous != current:
            transitions.append(("SELL" if previous == "GREEN" else "BUY", j - 3))
    if not transitions:
        return None, None
    return transitions[-1]


# ------------------------- Delta Reaction Zones -------------------------
@dataclass
class Zone:
    top: float
    bottom: float
    lower: bool  # True=green support; False=red resistance
    created_index: int


def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    # Pine ta.atr uses Wilder RMA.
    return pine_rma(tr, length)


def _confirmed_pivots(values: pd.Series, left: int, right: int) -> tuple[set[int], set[int]]:
    """Return pivot-high and pivot-low source indices, confirmed at i+right."""
    highs: set[int] = set()
    lows: set[int] = set()
    arr = values.to_numpy(dtype=float)
    for i in range(left, len(arr) - right):
        window = arr[i-left:i+right+1]
        if not np.isfinite(window).all():
            continue
        center = arr[i]
        # Pine pivots require the center to be the extremum; tie behavior is
        # handled conservatively to avoid inventing duplicate zones.
        if center == np.max(window) and np.count_nonzero(window == center) == 1:
            highs.add(i)
        if center == np.min(window) and np.count_nonzero(window == center) == 1:
            lows.add(i)
    return highs, lows


def build_active_zones(df: pd.DataFrame, tick_size: float) -> list[Zone]:
    """Rebuild zones chronologically; pivots appear only after 12 right bars."""
    n = len(df)
    if n < max(ATR_LENGTH + PIVOT_LENGTH * 2 + 5, 60):
        return []
    signed_position = np.sign(df["close"] - df["open"])
    raw_delta = signed_position * df["volume"]
    smoothed_delta = pine_ema(raw_delta, DELTA_SMOOTH)
    cumulative_delta = smoothed_delta.fillna(0).cumsum()
    pivot_highs, pivot_lows = _confirmed_pivots(cumulative_delta, PIVOT_LENGTH, PIVOT_LENGTH)
    atr = _atr(df, ATR_LENGTH)
    zones: list[Zone] = []
    merge_gap = max(0.0, float(tick_size)) * MERGE_GAP_TICKS

    for current in range(n):
        # Remove breached active zones using this closed candle before signal test.
        hi = float(df.iloc[current].high)
        lo = float(df.iloc[current].low)
        cl = float(df.iloc[current].close)
        survivors: list[Zone] = []
        for z in zones:
            breached = (cl < z.bottom) if z.lower else (cl > z.top)
            if not breached:
                survivors.append(z)
        zones = survivors

        source_idx = current - PIVOT_LENGTH
        if source_idx >= 0 and np.isfinite(atr.iloc[source_idx]):
            half_width = float(atr.iloc[source_idx]) * ATR_MULTIPLIER
            if source_idx in pivot_lows:
                anchor = float(df.iloc[source_idx].low)
                candidate = Zone(anchor + half_width, anchor - half_width, True, source_idx)
                zones = _merge_zone(zones, candidate, merge_gap)
            if source_idx in pivot_highs:
                anchor = float(df.iloc[source_idx].high)
                candidate = Zone(anchor + half_width, anchor - half_width, False, source_idx)
                zones = _merge_zone(zones, candidate, merge_gap)
        if len(zones) > MAXIMUM_ZONES:
            zones = sorted(zones, key=lambda z: z.created_index)[-MAXIMUM_ZONES:]
    return zones


def _merge_zone(zones: list[Zone], candidate: Zone, merge_gap: float) -> list[Zone]:
    if merge_gap > 0:
        for existing in zones:
            if existing.lower != candidate.lower:
                continue
            gap = max(0.0, max(existing.bottom, candidate.bottom) - min(existing.top, candidate.top))
            overlaps = candidate.bottom <= existing.top and candidate.top >= existing.bottom
            if overlaps or gap <= merge_gap:
                existing.top = max(existing.top, candidate.top)
                existing.bottom = min(existing.bottom, candidate.bottom)
                existing.created_index = min(existing.created_index, candidate.created_index)
                return zones
    zones.append(candidate)
    return zones


def delta_zone_confirmation(df: pd.DataFrame, tick_size: float, direction: str) -> bool:
    zones = build_active_zones(df, tick_size)
    if not zones:
        return False
    # If any active opposite-color zones overlap, suppress signals on this asset.
    for i, a in enumerate(zones):
        for b in zones[i + 1:]:
            if a.lower != b.lower and max(a.bottom, b.bottom) <= min(a.top, b.top):
                return False
    wanted_lower = direction == "BUY"
    matching = [z for z in zones if z.lower == wanted_lower]
    if not matching:
        return False
    recent = df.iloc[-CONFIRM_CANDLES:]
    for z in matching:
        for row in recent.itertuples(index=False):
            if float(row.high) >= z.bottom and float(row.low) <= z.top:
                return True
    return False


# ------------------------------- Signals -------------------------------
def tick_size_for(instrument: dict[str, Any]) -> float:
    for key in ("tickSize", "tick_size", "tickSizeIncrement", "tick_size_increment"):
        value = instrument.get(key)
        try:
            number = float(value)
            if number > 0:
                return number
        except (TypeError, ValueError):
            pass
    return 0.0


def pct_1h(df: pd.DataFrame) -> float:
    if len(df) < 5:
        return float("nan")
    base = float(df.iloc[-5].close)  # close four 15m bars earlier
    last = float(df.iloc[-1].close)
    return ((last / base) - 1.0) * 100.0 if base else float("nan")


def evaluate_symbol(instrument: dict[str, Any], ticker: dict[str, Any]) -> dict[str, Any] | None:
    symbol = str(instrument.get("symbol", "")).upper()
    try:
        df = fetch_candles(symbol)
        if len(df) < 100:
            return None
        now_ms = int(time.time() * 1000)
        last_close_time = int(df.iloc[-1].time) + INTERVAL_MS
        if now_ms - last_close_time > INTERVAL_MS:
            LOG.warning("Stale candles: %s (last closed candle age %.1f min)", symbol, (now_ms-last_close_time)/60000)
            return None
        colors = bxtrender_colors(df)
        signal, relative_position = detect_bxtrender_signal(colors)
        if signal is None:
            return None
        if not delta_zone_confirmation(df, tick_size_for(instrument), signal):
            return None
        change24 = ticker.get("change24h")
        if change24 is None:
            return None
        change24 = float(change24)
        return {
            "symbol": symbol,
            "signal": signal,
            "change24": change24,
            "change1h": pct_1h(df),
            "signal_candle_ms": int(df.iloc[-1 + int(relative_position or 0)].time) + INTERVAL_MS,
            "last_closed_color": str(colors.iloc[-1]),
        }
    except Exception as exc:
        LOG.warning("%s skipped: %s", symbol, exc)
        return None


def load_state() -> set[str]:
    if not STATE_FILE.exists():
        return set()
    try:
        return set(json.loads(STATE_FILE.read_text(encoding="utf-8")).get("sent", []))
    except Exception as exc:
        LOG.warning("State file unreadable; starting with empty state: %s", exc)
        return set()


def save_state(sent: set[str]) -> None:
    STATE_FILE.write_text(json.dumps({"sent": sorted(sent)[-20000:]}, ensure_ascii=False, indent=2), encoding="utf-8")


def format_signal_line(item: dict[str, Any]) -> str:
    circle = "🟢" if item["signal"] == "BUY" else "🔴"
    one_hour = item["change1h"]
    one_hour_text = f"{one_hour:+.2f}%" if math.isfinite(one_hour) else "N/A"
    return f"{circle} {item['symbol']} | {item['change24']:+.2f}% | {one_hour_text}"


def format_report(signals: list[dict[str, Any]], btc_colors: tuple[str, str] | None) -> str:
    if btc_colors:
        btc_line = f"BTC B-Xtrender Color (2 closed candles): {btc_colors[0]} → {btc_colors[1]}"
    else:
        btc_line = "BTC B-Xtrender Color (2 closed candles): N/A"
    buys = sorted((x for x in signals if x["signal"] == "BUY"), key=lambda x: x["change24"], reverse=True)
    sells = sorted((x for x in signals if x["signal"] == "SELL"), key=lambda x: x["change24"])
    lines = ["Scanner 4 | Kraken Futures | 15m", btc_line]
    lines.extend(format_signal_line(x) for x in buys)
    lines.extend(format_signal_line(x) for x in sells)
    if not buys and not sells:
        lines.append("No confirmed signals")
    return "\n".join(lines)


def telegram_send(message: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        LOG.info("Telegram credentials absent; report printed locally only.")
        return
    response = SESSION.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data={"chat_id": chat_id, "text": message, "disable_web_page_preview": True},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()


def main() -> None:
    LOG.info("Starting Scanner 4: Kraken Futures, %s", TIMEFRAME)
    instruments = fetch_instruments()
    tickers = fetch_tickers()
    candidates = [x for x in instruments if str(x.get("symbol", "")).upper() in tickers]
    LOG.info("Tradable instruments with tickers: %d", len(candidates))
    signals: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(evaluate_symbol, item, tickers[str(item.get("symbol", "")).upper()]): item for item in candidates}
        for future in as_completed(futures):
            result = future.result()
            if result:
                signals.append(result)
    # BTC status is informational only; it never gates or changes any market signal.
    btc_symbol = next((str(x.get("symbol", "")).upper() for x in candidates
                       if str(x.get("symbol", "")).upper() in ("PF_XBTUSD", "PI_XBTUSD")), None)
    btc_colors: tuple[str, str] | None = None
    if btc_symbol:
        try:
            btc_df = fetch_candles(btc_symbol)
            if len(btc_df) >= 100:
                btc_series = bxtrender_colors(btc_df).dropna()
                if len(btc_series) >= 2:
                    btc_colors = (str(btc_series.iloc[-2]), str(btc_series.iloc[-1]))
        except Exception as exc:
            LOG.warning("Could not retrieve BTC display status: %s", exc)
    # One notification per symbol/direction/signal candle; state prevents repeats.
    sent = load_state()
    fresh: list[dict[str, Any]] = []
    for item in signals:
        key = f"{item['symbol']}|{item['signal']}|{item['signal_candle_ms']}"
        if key not in sent:
            fresh.append(item)
            sent.add(key)
    save_state(sent)
    report = format_report(fresh, btc_colors)
    print(report, flush=True)
    telegram_send(report)


if __name__ == "__main__":
    main()
