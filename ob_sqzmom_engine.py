"""AB engine for Kraken Futures Scanner 2.

CNV supplies the first-stage three-candle signal. RSI(14) is informational
only. The supplied Order Blocks Zones with Signals logic is replayed on closed
1h candles; the latest still-active OB or FVG zone must match signal direction
and be touched by at least one of the latest five closed candles.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ZoneSettings:
    pivot_length: int = 10
    use_wicks: bool = True
    candle_size_period: int = 20
    fvg_size_multiplier: float = 0.5
    ob_size_multiplier: float = 0.1
    show_fvg: bool = True
    fvg_max_bars: int = 10
    max_age_bars: int = 500
    max_zones_per_type: int = 500


def validate_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    d.columns = [str(c).lower() for c in d.columns]
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    for col in required:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=list(required))
    d = d[~d.index.duplicated(keep="last")].sort_index()
    return d


def rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    close = close.astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    def rma(values: pd.Series) -> pd.Series:
        a = values.to_numpy(float)
        out = np.full(len(a), np.nan)
        if len(a) <= length:
            return pd.Series(out, index=values.index)
        seed = a[1:length + 1]
        if not np.isfinite(seed).all():
            return pd.Series(out, index=values.index)
        out[length] = seed.mean()
        for i in range(length + 1, len(a)):
            if np.isfinite(a[i]) and np.isfinite(out[i - 1]):
                out[i] = (out[i - 1] * (length - 1) + a[i]) / length
        return pd.Series(out, index=values.index)

    avg_gain = rma(gain)
    avg_loss = rma(loss)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    value = 100 - 100 / (1 + rs)
    value = value.mask((avg_loss == 0) & (avg_gain > 0), 100)
    value = value.mask((avg_gain == 0) & (avg_loss > 0), 0)
    value = value.mask((avg_gain == 0) & (avg_loss == 0), 50)
    return value


def cnv(frame: pd.DataFrame) -> pd.DataFrame:
    """CNV cumulative signed-volume trend used by the first indicator."""
    d = validate_ohlcv(frame)
    change = d.close.diff()
    signed_volume = pd.Series(
        np.where(change > 0, d.volume, np.where(change < 0, -d.volume, 0.0)),
        index=d.index,
    )
    cumulative = signed_volume.cumsum()
    average = cumulative.rolling(20, min_periods=20).mean()
    trend = cumulative - average
    blue = trend > 0
    return pd.DataFrame(
        {
            "cnv": cumulative,
            "cnv_sma20": average,
            "cnv_tb": trend,
            "blue": blue.fillna(False).astype(bool),
            "pink": (~blue).astype(bool),
        },
        index=d.index,
    )


def _is_pivot(values: np.ndarray, pivot_pos: int, want_low: bool) -> bool:
    """Approximate Pine ta.pivotlow/high tie handling (latest equal pivot wins)."""
    segment = values
    extreme = np.nanmin(segment) if want_low else np.nanmax(segment)
    if not np.isfinite(extreme) or values[pivot_pos] != extreme:
        return False
    positions = np.flatnonzero(values == extreme)
    return bool(len(positions) and positions[-1] == pivot_pos)


def build_zones(frame: pd.DataFrame, settings: ZoneSettings = ZoneSettings()) -> list[dict[str, Any]]:
    """Chronologically replay OB/FVG creation, invalidation and 500-bar cleanup.

    Zones are returned in creation order. Only still-active zones are returned.
    Pine source defaults are preserved: pivot=10, wicks, OB multiplier=.1,
    FVG multiplier=.5, 20-bar average body, FVG enabled, and FVG right edge
    at left edge + 10 + 1 bars.
    """
    d = validate_ohlcv(frame)
    n = len(d)
    if n == 0:
        return []

    o = d.open.to_numpy(float)
    h = d.high.to_numpy(float)
    lo = d.low.to_numpy(float)
    c = d.close.to_numpy(float)
    body = np.abs(c - o)
    avg_body = pd.Series(body).shift(1).rolling(settings.candle_size_period).mean().to_numpy()
    green = c > o
    zones: list[dict[str, Any]] = []
    serial = 0
    p = settings.pivot_length

    def add_zone(kind: str, direction: str, top: float, bottom: float,
                 origin: int, created: int, right_index: int | None = None) -> None:
        nonlocal serial, zones
        if not np.isfinite(top) or not np.isfinite(bottom) or top <= bottom:
            return
        serial += 1
        zones.append({
            "kind": kind,
            "direction": direction,
            "top": float(top),
            "bottom": float(bottom),
            "origin_index": int(origin),
            "created_index": int(created),
            "created_order": serial,
            "right_index": right_index,
            "active": True,
        })
        # The Pine arrays keep invalidated/deleted box IDs until age cleanup;
        # capacity therefore counts every retained entry, not only active zones.
        same_type = [z for z in zones if z["kind"] == kind]
        if len(same_type) > settings.max_zones_per_type:
            oldest = same_type[0]
            zones.remove(oldest)

    for t in range(n):
        # OB creation from confirmed pivots; Pine evaluates pivot at t-p.
        if t >= 2 * p and np.isfinite(avg_body[t]):
            window_start = t - 2 * p
            pivot_pos = p
            pivot_low = _is_pivot(lo[window_start:t + 1], pivot_pos, True)
            pivot_high = _is_pivot(h[window_start:t + 1], pivot_pos, False)
            idx = p
            pivot_green = bool(green[t - idx])
            prev_green = bool(green[t - (idx + 1)])
            next_green = bool(green[t - (idx - 1)])

            if pivot_low:
                source_idx = None
                is_bull = False
                if not pivot_green and next_green:
                    source_idx, is_bull = idx, True
                elif pivot_green and not prev_green:
                    source_idx, is_bull = idx + 1, True
                elif pivot_green and prev_green and next_green:
                    source_idx, is_bull = idx + 1, False
                if source_idx is not None:
                    s = t - source_idx
                    top = h[s] if settings.use_wicks else max(o[s], c[s])
                    bottom = lo[s] if settings.use_wicks else min(o[s], c[s])
                    if top - bottom >= avg_body[t] * settings.ob_size_multiplier:
                        add_zone("OB", "BUY" if is_bull else "SELL", top, bottom, s, t)

            if pivot_high:
                source_idx = None
                is_bull = False
                if pivot_green and not next_green:
                    source_idx, is_bull = idx, False
                elif not pivot_green and prev_green:
                    source_idx, is_bull = idx + 1, False
                elif not pivot_green and not prev_green and not next_green:
                    source_idx, is_bull = idx + 1, True
                if source_idx is not None:
                    s = t - source_idx
                    top = h[s] if settings.use_wicks else max(o[s], c[s])
                    bottom = lo[s] if settings.use_wicks else min(o[s], c[s])
                    if top - bottom >= avg_body[t] * settings.ob_size_multiplier:
                        add_zone("OB", "BUY" if is_bull else "SELL", top, bottom, s, t)

        # FVG creation occurs after OB creation in the supplied Pine script.
        if settings.show_fvg and t >= 2 and np.isfinite(avg_body[t]):
            if lo[t] > h[t - 2]:
                top, bottom = lo[t], h[t - 2]
                if top - bottom >= avg_body[t] * settings.fvg_size_multiplier:
                    add_zone("FVG", "BUY", top, bottom, t - 2, t,
                             right_index=(t - 2) + settings.fvg_max_bars + 1)
            if h[t] < lo[t - 2]:
                top, bottom = lo[t - 2], h[t]
                if top - bottom >= avg_body[t] * settings.fvg_size_multiplier:
                    add_zone("FVG", "SELL", top, bottom, t - 2, t,
                             right_index=(t - 2) + settings.fvg_max_bars + 1)

        # Pine invalidates FVGs after FVG creation and before OB signal loop.
        for z in zones:
            if not z["active"]:
                continue
            if z["kind"] == "FVG":
                if z["direction"] == "BUY" and c[t] < z["bottom"]:
                    z["active"] = False
                elif z["direction"] == "SELL" and c[t] > z["top"]:
                    z["active"] = False

        # OB invalidation is close below bullish bottom / above bearish top.
        for z in zones:
            if not z["active"] or z["kind"] != "OB":
                continue
            if z["direction"] == "BUY" and c[t] < z["bottom"]:
                z["active"] = False
            elif z["direction"] == "SELL" and c[t] > z["top"]:
                z["active"] = False

        # Pine removes array entries (active or inactive) after 500 bars.
        zones = [z for z in zones if (t - z["origin_index"]) <= settings.max_age_bars]

    return [dict(z) for z in zones if z["active"]]


def confirm_with_latest_zone(
    frame: pd.DataFrame,
    direction: str,
    settings: ZoneSettings = ZoneSettings(),
    lookback: int = 5,
) -> dict[str, Any]:
    """Confirm with the newest active OB/FVG, and only that zone.

    BUY accepts bullish green OB or blue FVG. SELL accepts bearish red OB or
    orange/yellow FVG. At least one of the last `lookback` CLOSED candles must
    overlap the zone by candle range (wick/body); close need not be inside.
    """
    d = validate_ohlcv(frame)
    zones = build_zones(d, settings)
    if not zones:
        return {"confirmed": False, "reason": "NO_ACTIVE_ZONE", "zone": None, "touched": False}
    latest = max(zones, key=lambda z: z["created_order"])
    if latest["direction"] != direction:
        return {"confirmed": False, "reason": "LATEST_ZONE_OPPOSITE", "zone": latest, "touched": False}
    recent_start = max(0, len(d) - lookback, int(latest.get("origin_index", 0)))
    right_index = latest.get("right_index")
    recent_end = len(d) if right_index is None else min(len(d), int(right_index) + 1)
    recent = d.iloc[recent_start:recent_end]
    touched = bool(len(recent) and ((recent.low <= latest["top"]) & (recent.high >= latest["bottom"])).any())
    if not touched:
        return {"confirmed": False, "reason": "NO_TOUCH_LAST_5", "zone": latest, "touched": False}
    return {"confirmed": True, "reason": "CONFIRMED", "zone": latest, "touched": True}


def analyze_market(frame: pd.DataFrame) -> dict[str, Any]:
    """Detect CNV pattern signals; RSI is calculated but NEVER filters signals."""
    d = validate_ohlcv(frame)
    c = cnv(d)
    rsi = rsi_wilder(d.close, 14)
    signals: list[dict[str, Any]] = []
    for i in range(2, len(d)):
        colors = [bool(c.blue.iloc[j]) for j in (i - 2, i - 1, i)]
        if colors == [False, True, True]:
            direction = "BUY"
        elif colors == [True, False, False]:
            direction = "SELL"
        else:
            continue
        stamp = d.index[i]
        signal_rsi = float(rsi.iloc[i]) if pd.notna(rsi.iloc[i]) else None
        signals.append({
            "direction": direction,
            "signal_index": i,
            "timestamp": stamp,
            "timestamp_ms": int(pd.Timestamp(stamp).timestamp() * 1000),
            "signal_time": pd.Timestamp(stamp).strftime("%Y-%m-%d %H:%M UTC"),
            "price": float(d.close.iloc[i]),
            "rsi": signal_rsi,
            "cnv_colors": ["BLUE" if x else "PINK" for x in colors],
        })
    latest_rsi = float(rsi.iloc[-1]) if len(rsi) and pd.notna(rsi.iloc[-1]) else None
    last_two_cnv_colors = [
        "BLUE" if bool(c.blue.iloc[i]) else "PINK"
        for i in range(max(0, len(c) - 2), len(c))
    ]
    return {
        "candles": d,
        "cnv": c,
        "rsi": rsi,
        "signals": signals,
        "latest_rsi": latest_rsi,
        "latest_cnv_color": "BLUE" if len(c) and bool(c.blue.iloc[-1]) else "PINK",
        "last_two_cnv_colors": last_two_cnv_colors,
    }
