"""Signal engine: SQZMOM LazyBear trigger + SMC confirmation.

All inputs are fully closed OHLCV candles. Reversal Probability is calculated
for informational color/zone reporting only; it never blocks a signal. SMC
confirms when ANY of the five most recent closed candles wick-touches a
directionally compatible SMC zone.

The Reversal Probability pivot/percentile logic follows the supplied Pine
source's core calculations. SMC is a Python reconstruction of the supplied
LuxAlgo SMC defaults and zone families; chart drawings/labels are not rendered.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np
import pandas as pd

HIGH = 1
LOW = -1


@dataclass(frozen=True)
class Settings:
    # SQZMOM LazyBear defaults
    bb_length: int = 20
    bb_mult: float = 2.0  # kept for source compatibility; source uses kc_mult below
    kc_length: int = 20
    kc_mult: float = 1.5
    use_true_range: bool = True

    # LuxAlgo Reversal Probability defaults
    reversal_swing_length: int = 20
    reversal_max_pivots: int = 1000
    reversal_normalize: bool = False
    reversal_percentile: int = 90

    # LuxAlgo SMC defaults / all-zone scan
    internal_pivot_length: int = 5
    swing_pivot_length: int = 50
    ob_atr_length: int = 200
    ob_atr_filter: bool = True
    ob_max_per_side_per_type: int = 5
    smc_lookback_candles: int = 5
    premium_discount_swing_length: int = 50
    premium_width_pct: float = 0.05
    equilibrium_half_width_pct: float = 0.025


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [str(c).lower() for c in d.columns]
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    for col in required:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=list(required)).sort_index()
    return d


def _linreg_last(values: np.ndarray) -> float:
    n = len(values)
    if n < 2 or not np.isfinite(values).all():
        return np.nan
    x = np.arange(n, dtype=float)
    xm = x.mean()
    denom = ((x - xm) ** 2).sum()
    if denom == 0:
        return float(values[-1])
    ym = values.mean()
    slope = ((x - xm) * (values - ym)).sum() / denom
    intercept = ym - slope * xm
    return float(intercept + slope * (n - 1))


def squeeze_momentum(df: pd.DataFrame, s: Settings = Settings()) -> pd.Series:
    d = validate_ohlcv(df)
    close, high, low = d.close, d.high, d.low
    basis = close.rolling(s.bb_length, min_periods=s.bb_length).mean()
    # Preserve the supplied LazyBear source behavior: multKC is used here.
    dev = s.kc_mult * close.rolling(s.bb_length, min_periods=s.bb_length).std(ddof=0)
    upper_bb, lower_bb = basis + dev, basis - dev

    previous_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    range_source = true_range if s.use_true_range else high - low
    range_ma = range_source.rolling(s.kc_length, min_periods=s.kc_length).mean()
    kc_ma = close.rolling(s.kc_length, min_periods=s.kc_length).mean()
    upper_kc = kc_ma + range_ma * s.kc_mult
    lower_kc = kc_ma - range_ma * s.kc_mult

    highest = high.rolling(s.kc_length, min_periods=s.kc_length).max()
    lowest = low.rolling(s.kc_length, min_periods=s.kc_length).min()
    source = close - (((highest + lowest) / 2.0 + kc_ma) / 2.0)
    hist = source.rolling(s.kc_length, min_periods=s.kc_length).apply(_linreg_last, raw=True)
    hist.name = "sqzmom"
    return hist


def _nearest_rank(values: list[float] | list[int], percentile: int) -> float | int | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(ceil(percentile / 100.0 * len(ordered)))))
    return ordered[rank - 1]


def reversal_probability_zones(df: pd.DataFrame, s: Settings = Settings()) -> list[dict[str, Any] | None]:
    """Return the active LuxAlgo-style forecast zone at each closed candle."""
    d = validate_ohlcv(df)
    o = d.open.to_numpy(float)
    c = d.close.to_numpy(float)
    n = len(d)
    length = s.reversal_swing_length

    bullish_prices: list[float] = []
    bullish_bars: list[int] = []
    bearish_prices: list[float] = []
    bearish_bars: list[int] = []

    bias: int | None = None
    current_price = float(c[0]) if n else 0.0
    current_bar = 0
    last_pivot_price = float(c[0]) if n else 0.0
    last_pivot_bar = 0
    active_zone: dict[str, Any] | None = None
    zones: list[dict[str, Any] | None] = [None] * n

    body_high = np.maximum(o, c)
    body_low = np.minimum(o, c)
    for t in range(n):
        start = max(0, t - length + 1)
        upper = float(np.max(body_high[start:t + 1]))
        lower = float(np.min(body_low[start:t + 1]))
        max_now = float(body_high[t])
        min_now = float(body_low[t])
        new_bias = HIGH if max_now == upper else LOW if min_now == lower else bias
        new_pivot = new_bias is not None and new_bias != bias

        if new_pivot:
            pivot_price = current_price
            pivot_bar = current_bar
            pivot_bias = HIGH if new_bias == LOW else LOW

            bullish_move = pivot_price > last_pivot_price
            price_delta = abs(pivot_price - last_pivot_price)
            if s.reversal_normalize:
                price_delta = price_delta / max(abs(last_pivot_price), 1e-12)
            bars_delta = pivot_bar - last_pivot_bar
            if bars_delta != 0 and np.isfinite(price_delta):
                if bullish_move:
                    bullish_prices.append(float(price_delta)); bullish_bars.append(int(bars_delta))
                    if len(bullish_prices) > s.reversal_max_pivots:
                        bullish_prices.pop(0); bullish_bars.pop(0)
                else:
                    bearish_prices.append(float(price_delta)); bearish_bars.append(int(bars_delta))
                    if len(bearish_prices) > s.reversal_max_pivots:
                        bearish_prices.pop(0); bearish_bars.pop(0)
                last_pivot_price = pivot_price
                last_pivot_bar = pivot_bar

            # The Pine forecast uses bearish samples for a pivot high and
            # bullish samples for a pivot low, then the maximum enabled rank (90th).
            p_prices = bearish_prices if pivot_bias == HIGH else bullish_prices
            p_bars = bearish_bars if pivot_bias == HIGH else bullish_bars
            price_pct = _nearest_rank(p_prices, s.reversal_percentile)
            bars_pct = _nearest_rank(p_bars, s.reversal_percentile)
            if price_pct is not None and bars_pct is not None and bars_pct > 0:
                if pivot_bias == HIGH:
                    other = pivot_price - float(price_pct)
                    direction = "SELL"
                else:
                    other = pivot_price + float(price_pct)
                    direction = "BUY"
                active_zone = {
                    "direction": direction,
                    "top": max(pivot_price, other),
                    "bottom": min(pivot_price, other),
                    "start_bar": int(pivot_bar),
                    "end_bar": int(pivot_bar + bars_pct),
                }

            current_price = upper if new_bias == HIGH else lower
            current_bar = t
        else:
            if bias is not None:
                next_price = max(upper, current_price) if bias == HIGH else min(lower, current_price)
                if next_price != current_price:
                    current_price = next_price
                    current_bar = t

        bias = new_bias
        if active_zone is not None and t <= active_zone["end_bar"]:
            zones[t] = dict(active_zone)

    return zones


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev = df.close.shift(1)
    tr = pd.concat(
        [df.high - df.low, (df.high - prev).abs(), (df.low - prev).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def _confirmed_pivot(values: np.ndarray, center: int, length: int, is_high: bool) -> bool:
    left = center - length
    right = center + length + 1
    if left < 0 or right > len(values):
        return False
    window = values[left:right]
    target = values[center]
    if is_high:
        return bool(target == np.max(window) and np.argmax(window) == length)
    return bool(target == np.min(window) and np.argmin(window) == length)


def smart_money_zones(df: pd.DataFrame, s: Settings = Settings()) -> list[list[dict[str, Any]]]:
    """Reconstruct internal/swing OB and Premium/Discount/Equilibrium zones.

    Bullish OB + Discount are BUY zones; bearish OB + Premium are SELL zones.
    Equilibrium remains neutral and is not used as directional confirmation.
    Confirmation is evaluated later as a wick-range touch of a matching zone.
    """
    d = validate_ohlcv(df)
    n = len(d)
    o, h, l, c = (d[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    atr = _atr(d, s.ob_atr_length).to_numpy(float)
    zones_by_bar: list[list[dict[str, Any]]] = [[] for _ in range(n)]
    order_blocks: list[dict[str, Any]] = []
    structures: dict[str, dict[str, Any]] = {
        "internal": {"last_high": None, "last_low": None, "broken_high": None, "broken_low": None},
        "swing": {"last_high": None, "last_low": None, "broken_high": None, "broken_low": None},
    }

    def add_ob(kind: str, bullish: bool, pivot_idx: int, break_idx: int) -> None:
        if break_idx <= pivot_idx:
            return
        candidates = range(max(0, pivot_idx), break_idx)
        source_idx = None
        for k in reversed(list(candidates)):
            is_bearish_candle = c[k] < o[k]
            if (bullish and is_bearish_candle) or ((not bullish) and (not is_bearish_candle)):
                source_idx = k
                break
        if source_idx is None:
            return
        top, bottom = float(h[source_idx]), float(l[source_idx])
        if s.ob_atr_filter and (top - bottom) > 2.5 * max(float(atr[break_idx]), 1e-12):
            return
        order_blocks.append({
            "kind": f"{kind}_{'bullish' if bullish else 'bearish'}_ob",
            "direction": "BUY" if bullish else "SELL",
            "top": top,
            "bottom": bottom,
            "born": break_idx,
            "active": True,
        })
        same = [z for z in order_blocks if z["kind"] == f"{kind}_{'bullish' if bullish else 'bearish'}_ob" and z["active"]]
        if len(same) > s.ob_max_per_side_per_type:
            oldest = min(same, key=lambda z: z["born"])
            oldest["active"] = False

    for t in range(n):
        # Detect confirmed internal (5/5) and swing (50/50) pivots.
        for kind, length in (("internal", s.internal_pivot_length), ("swing", s.swing_pivot_length)):
            center = t - length
            if center >= length:
                if _confirmed_pivot(h, center, length, True):
                    structures[kind]["last_high"] = (float(h[center]), center)
                if _confirmed_pivot(l, center, length, False):
                    structures[kind]["last_low"] = (float(l[center]), center)

            state = structures[kind]
            last_high = state["last_high"]
            last_low = state["last_low"]
            if last_high and state["broken_high"] != last_high[1] and c[t] > last_high[0]:
                add_ob(kind, True, last_high[1], t)
                state["broken_high"] = last_high[1]
            if last_low and state["broken_low"] != last_low[1] and c[t] < last_low[0]:
                add_ob(kind, False, last_low[1], t)
                state["broken_low"] = last_low[1]

        # Apply SMC high/low mitigation to retained OBs.
        for z in order_blocks:
            if not z["active"]:
                continue
            if t - z["born"] > 500:
                z["active"] = False
            elif z["direction"] == "BUY" and l[t] < z["bottom"]:
                z["active"] = False
            elif z["direction"] == "SELL" and h[t] > z["top"]:
                z["active"] = False

        current: list[dict[str, Any]] = []
        for z in order_blocks:
            if z["active"] and z["born"] <= t:
                current.append(z)

        # Premium / Discount / Equilibrium zones from trailing swing extremes.
        sw = s.premium_discount_swing_length
        start = max(0, t - sw + 1)
        swing_high = float(np.max(h[start:t + 1]))
        swing_low = float(np.min(l[start:t + 1]))
        span = max(swing_high - swing_low, 1e-12)
        premium_bottom = swing_high - span * s.premium_width_pct
        discount_top = swing_low + span * s.premium_width_pct
        eq_mid = (swing_high + swing_low) / 2.0
        eq_half = span * s.equilibrium_half_width_pct
        current.extend([
            {"kind": "premium", "direction": "SELL", "top": swing_high, "bottom": premium_bottom},
            {"kind": "discount", "direction": "BUY", "top": discount_top, "bottom": swing_low},
            {"kind": "equilibrium", "direction": "BOTH", "top": eq_mid + eq_half, "bottom": eq_mid - eq_half},
        ])
        zones_by_bar[t] = [dict(z) for z in current]

    return zones_by_bar


def _zone_contains(close: float, zone: dict[str, Any]) -> bool:
    """Whether a close is inside a forecast price interval."""
    return float(zone["bottom"]) <= close <= float(zone["top"])


def _candle_touches_zone(low: float, high: float, zone: dict[str, Any]) -> bool:
    """True when a candle's full wick range overlaps a zone."""
    return float(low) <= float(zone["top"]) and float(high) >= float(zone["bottom"])


def evaluate_signals(df: pd.DataFrame, settings: Settings = Settings()) -> pd.DataFrame:
    """SQZMOM creates signals; SMC is the only confirmation filter.

    Reversal Probability is retained as context (direction and box bounds),
    but no Reversal condition is allowed to reject or change a signal.
    """
    d = validate_ohlcv(df)
    out = d.copy()
    out["sqzmom"] = squeeze_momentum(d, settings)
    reversal_zones = reversal_probability_zones(d, settings)
    smc_zones = smart_money_zones(d, settings)

    n = len(out)
    signals: list[str] = [""] * n
    pattern_mid: list[float] = [np.nan] * n
    reversal_type: list[str] = [""] * n
    reversal_color: list[str] = ["بدون باکس / خنثی"] * n
    reversal_top: list[float] = [np.nan] * n
    reversal_bottom: list[float] = [np.nan] * n
    smc_match_text: list[str] = [""] * n
    smc_match_age: list[int] = [-1] * n

    hist = out.sqzmom.to_numpy(float)
    lows = out.low.to_numpy(float)
    highs = out.high.to_numpy(float)

    for i in range(n):
        # Reversal is informational only, including on candles without signals.
        rz = reversal_zones[i]
        if rz:
            reversal_type[i] = str(rz.get("direction", ""))
            reversal_color[i] = "سبز" if rz.get("direction") == "BUY" else "قرمز" if rz.get("direction") == "SELL" else "بدون باکس / خنثی"
            reversal_top[i] = float(rz["top"])
            reversal_bottom[i] = float(rz["bottom"])

        if i < 2 or not np.isfinite(hist[i - 2:i + 1]).all():
            continue
        left, middle, right = hist[i - 2], hist[i - 1], hist[i]
        signal = ""
        # All three closed bars must remain on the same side of zero;
        # the center bar is the strict local extreme.
        if left > 0 and middle > 0 and right > 0 and abs(middle) > abs(left) and abs(middle) > abs(right):
            signal = "SELL"
        elif left < 0 and middle < 0 and right < 0 and abs(middle) > abs(left) and abs(middle) > abs(right):
            signal = "BUY"
        if not signal:
            continue

        # SMC is the sole confirmation: wick-touch by any of the last five
        # closed candles, using only zones known at each candle (no lookahead).
        found: list[tuple[int, str]] = []
        first = max(0, i - settings.smc_lookback_candles + 1)
        for j in range(i, first - 1, -1):
            for z in smc_zones[j]:
                if z["direction"] == signal and _candle_touches_zone(lows[j], highs[j], z):
                    found.append((i - j, str(z["kind"])))
        if not found:
            continue
        found.sort(key=lambda x: (x[0], x[1]))

        signals[i] = signal
        pattern_mid[i] = middle
        smc_match_text[i] = ", ".join(f"{name} ({age} candle ago)" for age, name in found[:6])
        smc_match_age[i] = int(found[0][0])

    out["signal"] = signals
    out["sqzmom_peak_trough"] = pattern_mid
    out["reversal_direction"] = reversal_type
    out["reversal_color"] = reversal_color
    out["reversal_zone_top"] = reversal_top
    out["reversal_zone_bottom"] = reversal_bottom
    out["smc_matches"] = smc_match_text
    out["smc_match_age"] = smc_match_age
    return out


def latest_confirmed_btc_pivot(hist: pd.Series) -> dict[str, Any] | None:
    """Return the latest confirmed positive peak (SELL) or negative trough (BUY).

    Uses two closed candles on each side of the pivot. The two right-side
    candles confirm the turn; no amplitude multiplier or height threshold is
    applied. Only finite histogram values are considered.
    """
    values = pd.to_numeric(hist, errors="coerce").to_numpy(dtype=float)
    if len(values) < 5:
        return None

    latest: dict[str, Any] | None = None
    for center in range(2, len(values) - 2):
        window = values[center - 2:center + 3]
        if not np.isfinite(window).all():
            continue
        left2, left1, pivot, right1, right2 = map(float, window)
        event: str | None = None
        if pivot > 0 and pivot > left1 and pivot > left2 and pivot > right1 > right2:
            event = "SELL"
        elif pivot < 0 and pivot < left1 and pivot < left2 and pivot < right1 < right2:
            event = "BUY"
        if event:
            latest = {
                "signal": event,
                "pivot_position": center,
                "pivot_time": hist.index[center],
                "pivot_value": pivot,
                "confirmation_time": hist.index[center + 2],
            }
    return latest
