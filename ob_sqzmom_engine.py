"""
Core signal engine for:
1) Dynamic Profile Key Levels — AlgoAlpha (Volume Peaks + Ranked Volume Rows)
2) Squeeze Momentum Indicator — LazyBear

Signal rules configured by the user:
BUY:
  - SQZMOM histogram is below zero
  - absolute histogram bar is shorter than the previous bar
  - at least one of the latest three pattern candles is inside an active bullish (green) profile zone

SELL:
  - SQZMOM histogram is above zero
  - absolute histogram bar is shorter than the previous bar
  - at least one of the latest three pattern candles is inside an active bearish (red) profile zone

Signals are evaluated on CLOSED candles only.
This is the calculation module, not yet the 24/7 Binance/Telegram service.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Settings:
    # AlgoAlpha Dynamic Profile defaults used by the supplied Pine source.
    profile_lookback: int = 240
    profile_resolution: int = 64
    profile_refresh_bars: int = 5
    profile_peak_count: int = 2
    profile_peak_detection_pct: float = 7.0
    profile_min_node_volume_pct: float = 1.0
    profile_smoothing_radius: int = 1
    profile_zone_radius: int = 1
    profile_ranked_high_rows: int = 1
    profile_level_expiry_bars: int = 500
    profile_match_tolerance_rows: float = 0.75

    # LazyBear defaults
    bb_length: int = 20
    bb_mult: float = 2.0
    kc_length: int = 20
    kc_mult: float = 1.5
    use_true_range: bool = True

    # TradingView script removes OBs older than 500 bars
    ob_max_age_bars: int = 500


def _validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")

    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    for col in ("open", "high", "low", "close"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out.sort_index()
    return out


def squeeze_momentum(df: pd.DataFrame, s: Settings = Settings()) -> pd.DataFrame:
    """
    Reproduces the supplied LazyBear formula.

    Important: the supplied source uses kc_mult for BB deviation too:
        dev = multKC * stdev(source, length)
    It does NOT use bb_mult in that line. This implementation preserves
    the supplied source rather than silently correcting it.
    """
    d = _validate_ohlcv(df)
    close, high, low = d["close"], d["high"], d["low"]

    basis = close.rolling(s.bb_length).mean()
    # Preserve source-code behavior: multKC, not mult, is used here.
    dev = s.kc_mult * close.rolling(s.bb_length).std(ddof=0)
    upper_bb = basis + dev
    lower_bb = basis - dev

    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    candle_range = true_range if s.use_true_range else (high - low)
    range_ma = candle_range.rolling(s.kc_length).mean()
    ma = close.rolling(s.kc_length).mean()
    upper_kc = ma + range_ma * s.kc_mult
    lower_kc = ma - range_ma * s.kc_mult

    sqz_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)
    sqz_off = (lower_bb < lower_kc) & (upper_bb > upper_kc)
    no_sqz = ~(sqz_on | sqz_off)

    highest_high = high.rolling(s.kc_length).max()
    lowest_low = low.rolling(s.kc_length).min()
    midpoint = (highest_high + lowest_low) / 2.0
    regression_source = close - (midpoint + ma) / 2.0

    # ta.linreg(source, length, 0): fitted value at the newest x in the window.
    x = np.arange(s.kc_length, dtype=float)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()

    def linreg_at_last(values: np.ndarray) -> float:
        if len(values) != s.kc_length or not np.isfinite(values).all():
            return np.nan
        y_mean = values.mean()
        slope = ((x - x_mean) * (values - y_mean)).sum() / denom
        intercept = y_mean - slope * x_mean
        return float(intercept + slope * (s.kc_length - 1))

    val = regression_source.rolling(s.kc_length).apply(linreg_at_last, raw=True)

    out = pd.DataFrame(index=d.index)
    out["sqzmom"] = val
    out["sqz_on"] = sqz_on
    out["sqz_off"] = sqz_off
    out["no_sqz"] = no_sqz
    out["histogram_color"] = np.select(
        [val > 0, val < 0],
        [
            np.where(val > val.shift(1), "lime", "green"),
            np.where(val < val.shift(1), "red", "maroon"),
        ],
        default="zero",
    )
    # User's rule: every shorter bar can qualify, not only the first one.
    out["buy_momentum"] = (val < 0) & (val.abs() < val.shift(1).abs())
    out["sell_momentum"] = (val > 0) & (val.abs() < val.shift(1).abs())
    return out


def dynamic_profile_zones(df: pd.DataFrame, s: Settings = Settings()) -> pd.DataFrame:
    """Calculate AlgoAlpha-style volume-at-price peak and ranked-row zones.

    Uses the supplied Pine source's Fixed 240-bar profile, 64 rows, 5-bar
    refresh, 2 local peaks, 1 highest ranked row, 7% peak neighborhood,
    1% minimum node strength, one-row smoothing and one-row zone radius.
    Volume is distributed uniformly over each chart candle's high-low range,
    and directional volume is estimated from candle direction, matching the
    source's documented chart-candle fallback (not bid/ask order flow).

    The source's persistent level behavior is replayed: matching nodes retain
    their original bounds, old nodes stay active until close-break/expiry, and
    same-kind overlapping nodes are not duplicated. BUY uses green/support
    nodes (price above zone); SELL uses red/resistance nodes (price below zone).
    """
    d = _validate_ohlcv(df)
    if "volume" not in d.columns:
        raise ValueError("Missing volume column required for Dynamic Profile")
    volume = pd.to_numeric(d["volume"], errors="coerce").fillna(0.0).clip(lower=0).to_numpy(float)
    o, h, l, c = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(d)
    bull_inside = np.zeros(n, dtype=bool)
    bear_inside = np.zeros(n, dtype=bool)
    zone_type = np.full(n, "", dtype=object)
    zone_top = np.full(n, np.nan)
    zone_bottom = np.full(n, np.nan)
    zone_count = np.zeros(n, dtype=int)
    matched_candle = np.full(n, -1, dtype=int)
    levels: list[dict] = []
    resolution = s.profile_resolution

    def build_candidates(t: int):
        first_bar = max(0, t - s.profile_lookback + 1)
        rh = float(np.max(h[first_bar:t+1])); rl = float(np.min(l[first_bar:t+1]))
        step = max((rh - rl) / resolution, 1e-12)
        rows = np.zeros(resolution, dtype=float)
        for k in range(first_bar, t + 1):
            vol = volume[k]
            if vol <= 0:
                continue
            lo, hi = float(l[k]), float(h[k])
            first = max(0, min(resolution-1, int(np.floor((lo-rl)/step))))
            last = max(first, min(resolution-1, int(np.floor((hi-rl)/step))))
            if first == last or hi <= lo:
                rows[first] += vol
            else:
                span = hi-lo
                rows[first] += vol * max(0.0, min(1.0, (rl+(first+1)*step-lo)/span))
                rows[last] += vol * max(0.0, min(1.0, (hi-rl-last*step)/span))
                if last > first+1:
                    rows[first+1:last] += vol * step/span
        if not np.any(rows > 0):
            return [], step
        smooth = rows.copy()
        rad = s.profile_smoothing_radius
        if rad:
            for i in range(resolution):
                a, b = max(0, i-rad), min(resolution, i+rad+1)
                smooth[i] = rows[a:b].mean()
        maximum = float(rows.max())
        peak_radius = max(1, int(round(resolution*s.profile_peak_detection_pct/100.0)))
        candidates = []
        used = set()
        for row in np.argsort(-smooth, kind="stable"):
            row = int(row)
            if len([x for x in candidates if x["kind"] == "peak"]) >= s.profile_peak_count:
                break
            if rows[row] <= 0 or rows[row]/maximum*100 < s.profile_min_node_volume_pct:
                continue
            a, b = max(0,row-peak_radius), min(resolution-1,row+peak_radius)
            valid = True; contrast = False
            for j in range(a,b+1):
                if j == row: continue
                if smooth[j] > smooth[row] or (smooth[j] == smooth[row] and j < row):
                    valid=False; break
                contrast = contrast or smooth[j] < smooth[row]
            if valid and contrast and row not in used:
                used.add(row); candidates.append({"row":row,"kind":"peak"})
        # Ranked Volume Rows: source defaults to one highest-volume row.
        ranked_count = 0
        for row in np.argsort(-rows, kind="stable"):
            row = int(row)
            if ranked_count >= s.profile_ranked_high_rows: break
            if rows[row] <= 0 or rows[row]/maximum*100 < s.profile_min_node_volume_pct or row in used:
                continue
            used.add(row); candidates.append({"row":row,"kind":"ranked"}); ranked_count += 1
        out=[]
        for x in candidates:
            row=x["row"]; r=s.profile_zone_radius
            a=max(0,row-r); b=min(resolution-1,row+r)
            out.append({"kind":x["kind"], "price":rl+(row+0.5)*step,
                        "top":rl+(b+1)*step, "bottom":rl+a*step,
                        "born":t, "last_seen":t, "active":True, "side":0})
        return out, step

    for t in range(n):
        if t+1 >= 10 and (t % s.profile_refresh_bars == 0 or t == 9):
            candidates, step = build_candidates(t)
            for cand in candidates:
                tolerance=max(step*s.profile_match_tolerance_rows, 1e-12)
                matches=[z for z in levels if z["active"] and z["kind"]==cand["kind"]
                         and z["last_seen"] != t and abs(z["price"]-cand["price"]) <= tolerance]
                if matches:
                    best=min(matches,key=lambda z:abs(z["price"]-cand["price"]))
                    best["last_seen"]=t
                else:
                    # Avoid overlapping active nodes, as in preventOverlap=true.
                    if not any(z["active"] and cand["top"]>=z["bottom"] and cand["bottom"]<=z["top"] for z in levels):
                        levels.append(cand)
        # Expiry, break handling, and directional side of each retained node.
        for z in levels:
            if not z["active"]: continue
            if t-z["born"] >= s.profile_level_expiry_bars:
                z["active"]=False; continue
            if z["side"] == 1 and c[t] < z["bottom"]:
                z["active"]=False; continue
            if z["side"] == -1 and c[t] > z["top"]:
                z["active"]=False; continue
            if c[t] > z["top"]: z["side"]=1
            elif c[t] < z["bottom"]: z["side"]=-1
        active=[z for z in levels if z["active"]]
        bull=[z for z in active if z["side"]==1]
        bear=[z for z in active if z["side"]==-1]
        bull_inside[t]=any(z["bottom"] <= c[t] <= z["top"] for z in bull)
        bear_inside[t]=any(z["bottom"] <= c[t] <= z["top"] for z in bear)
        zone_count[t]=sum(z["bottom"] <= c[t] <= z["top"] for z in active)
        relevant = bull if bull_inside[t] else bear if bear_inside[t] else []
        if relevant:
            z=min(relevant,key=lambda x:abs(c[t]-x["price"]))
            zone_type[t]=z["kind"]
            zone_top[t]=z["top"]; zone_bottom[t]=z["bottom"]
        # Store most relevant current directional zone metadata even when the
        # latest close is outside; the caller checks the three pattern bars.
        elif active:
            z=min(active,key=lambda x:0 if x["bottom"]<=c[t]<=x["top"] else min(abs(c[t]-x["bottom"]),abs(c[t]-x["top"])))
            zone_type[t]=z["kind"]; zone_top[t]=z["top"]; zone_bottom[t]=z["bottom"]

    return pd.DataFrame({
        "bull_profile_inside": bull_inside,
        "bear_profile_inside": bear_inside,
        "profile_zone_type": zone_type,
        "profile_zone_top": zone_top,
        "profile_zone_bottom": zone_bottom,
        "profile_zone_count": zone_count,
    }, index=d.index)


def evaluate_signals(df: pd.DataFrame, settings: Settings = Settings()) -> pd.DataFrame:
    """Calculate SQZMOM and Dynamic Profile signals on CLOSED candles.

    A SQZMOM contraction on the latest candle is the trigger. At least one
    of the latest three candle closes must have been inside a matching active
    bullish/green or bearish/red profile node zone.
    """
    d = _validate_ohlcv(df)
    sqz = squeeze_momentum(d, settings)
    profile = dynamic_profile_zones(d, settings)
    result = pd.concat([d, sqz, profile], axis=1)
    bull_any = result["bull_profile_inside"].rolling(3, min_periods=1).max().astype(bool)
    bear_any = result["bear_profile_inside"].rolling(3, min_periods=1).max().astype(bool)
    result["profile_pattern_candle"] = -1
    for i in range(len(result)):
        hit = None
        if bool(bull_any.iloc[i]):
            hits=[j for j in range(max(0,i-2),i+1) if bool(result["bull_profile_inside"].iloc[j])]
            hit = hits[-1]
        elif bool(bear_any.iloc[i]):
            hits=[j for j in range(max(0,i-2),i+1) if bool(result["bear_profile_inside"].iloc[j])]
            hit = hits[-1]
        if hit is not None:
            result.iloc[i, result.columns.get_loc("profile_pattern_candle")] = hit-i
            for col in ("profile_zone_type", "profile_zone_top", "profile_zone_bottom"):
                result.iloc[i, result.columns.get_loc(col)] = result.iloc[hit][col]
    result["BUY"] = result["buy_momentum"] & bull_any
    result["SELL"] = result["sell_momentum"] & bear_any
    result["signal"] = np.select([result["BUY"], result["SELL"]], ["BUY", "SELL"], default="")
    return result


def load_ohlcv_csv(path: str) -> pd.DataFrame:
    """
    CSV columns required: timestamp, open, high, low, close.
    Optional volume column is ignored by this calculation engine.
    """
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    return _validate_ohlcv(df)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test the Dynamic Profile + SQZMOM signal engine on a closed-candle OHLCV CSV."
    )
    parser.add_argument("csv", help="CSV file with timestamp,open,high,low,close")
    parser.add_argument("--timeframe", default="30m", choices=["30m"])
    args = parser.parse_args()

    candles = load_ohlcv_csv(args.csv)
    output = evaluate_signals(candles)
    hits = output[output["signal"] != ""][
        ["close", "sqzmom", "bull_profile_inside", "bear_profile_inside", "profile_zone_type", "signal"]
    ]
    print(f"Timeframe: {args.timeframe}")
    print(f"Candles evaluated: {len(output)}")
    print(f"Signals found: {len(hits)}")
    print(hits.tail(100).to_string())
