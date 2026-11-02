"""
Core signal engine for Order Blocks Zones with Signals — Uncle_the_shooter
and Squeeze Momentum Indicator — LazyBear. The original colored OB logic is used.

Signal rules: three CLOSED SQZMOM bars, same side of zero, middle strict
peak/trough; at least one of those three candles must close inside a matching
active colored Order Block. BTC's latest confirmed peak/trough establishes a
persistent directional gate for altcoins. No synthetic Premium/Discount zones,
no multiplier, no amplitude threshold, and no simultaneous opposite signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Settings:
    pivot_length: int = 10
    use_wicks: bool = True
    candle_size_period: int = 20
    ob_size_multiplier: float = 0.1

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


def order_block_zones(df: pd.DataFrame, s: Settings = Settings()) -> pd.DataFrame:
    """
    Replays the supplied Order Blocks zone-creation and invalidation logic
    chronologically. Returns whether the close is inside any active green
    bullish OB or red bearish OB on each candle.

    The FVG drawing logic is intentionally not included because the user's
    signal condition uses the colored Order Block zones, not FVG zones.
    """
    d = _validate_ohlcv(df).reset_index(drop=False)
    index_name = d.columns[0]
    n = len(d)

    o = d["open"].to_numpy(float)
    h = d["high"].to_numpy(float)
    l = d["low"].to_numpy(float)
    c = d["close"].to_numpy(float)
    body = np.abs(c - o)

    # Pine avg_body = sma(abs(close[1] - open[1]), length)
    # At bar t this is the mean of body sizes t-1 ... t-length.
    avg_body = pd.Series(body).shift(1).rolling(s.candle_size_period).mean().to_numpy()

    # Zone record: top, bottom, creation bar, bullish, active, inside.
    zones: list[dict] = []
    bull_inside = np.zeros(n, dtype=bool)
    bear_inside = np.zeros(n, dtype=bool)
    bull_zone_count = np.zeros(n, dtype=int)
    bear_zone_count = np.zeros(n, dtype=int)

    p = s.pivot_length

    for t in range(n):
        center = t - p

        # A pivot is only known after p right-side bars have closed.
        if center >= p and t >= 2 * p:
            left = center - p
            right = center + p + 1
            window_low = l[left:right]
            window_high = h[left:right]
            pivot_low = l[center] == np.min(window_low)
            pivot_high = h[center] == np.max(window_high)

            def green(k: int) -> bool:
                return c[k] > o[k]

            if np.isfinite(avg_body[t]):
                # Pivot-low branch: exact source-candle selection and color rules.
                if pivot_low:
                    pivot_green = green(center)
                    prev_green = green(center - 1)
                    next_green = green(center + 1)
                    source_idx: Optional[int] = None
                    is_bull: Optional[bool] = None

                    if (not pivot_green) and next_green:
                        source_idx, is_bull = center, True
                    elif pivot_green and (not prev_green):
                        source_idx, is_bull = center - 1, True
                    elif pivot_green and prev_green and next_green:
                        source_idx, is_bull = center - 1, False

                    if source_idx is not None:
                        top = h[source_idx] if s.use_wicks else max(o[source_idx], c[source_idx])
                        bot = l[source_idx] if s.use_wicks else min(o[source_idx], c[source_idx])
                        if (top - bot) >= avg_body[t] * s.ob_size_multiplier:
                            zones.append({
                                "top": float(top), "bot": float(bot),
                                "created": t, "bull": bool(is_bull),
                                "active": True, "inside": False,
                            })

                # Pivot-high branch.
                if pivot_high:
                    pivot_green = green(center)
                    prev_green = green(center - 1)
                    next_green = green(center + 1)
                    source_idx = None
                    is_bull = None

                    if pivot_green and (not next_green):
                        source_idx, is_bull = center, False
                    elif (not pivot_green) and prev_green:
                        source_idx, is_bull = center - 1, False
                    elif (not pivot_green) and (not prev_green) and (not next_green):
                        source_idx, is_bull = center - 1, True

                    if source_idx is not None:
                        top = h[source_idx] if s.use_wicks else max(o[source_idx], c[source_idx])
                        bot = l[source_idx] if s.use_wicks else min(o[source_idx], c[source_idx])
                        if (top - bot) >= avg_body[t] * s.ob_size_multiplier:
                            zones.append({
                                "top": float(top), "bot": float(bot),
                                "created": t, "bull": bool(is_bull),
                                "active": True, "inside": False,
                            })

        # Pine removes zones older than 500 bars.
        zones = [z for z in zones if (t - z["created"]) <= s.ob_max_age_bars]

        for z in zones:
            if not z["active"]:
                continue
            # Same invalidation conditions as the supplied Pine source.
            if z["bull"] and c[t] < z["bot"]:
                z["active"] = False
                continue
            if (not z["bull"]) and c[t] > z["top"]:
                z["active"] = False
                continue

            inside = z["bot"] <= c[t] <= z["top"]
            z["inside"] = inside

        active_bull = [z for z in zones if z["active"] and z["bull"]]
        active_bear = [z for z in zones if z["active"] and not z["bull"]]
        bull_inside[t] = any(z["bot"] <= c[t] <= z["top"] for z in active_bull)
        bear_inside[t] = any(z["bot"] <= c[t] <= z["top"] for z in active_bear)
        bull_zone_count[t] = sum(z["bot"] <= c[t] <= z["top"] for z in active_bull)
        bear_zone_count[t] = sum(z["bot"] <= c[t] <= z["top"] for z in active_bear)

    result = pd.DataFrame({
        "bull_ob_inside": bull_inside,
        "bear_ob_inside": bear_inside,
        "bull_ob_count": bull_zone_count,
        "bear_ob_count": bear_zone_count,
    }, index=d[index_name])
    return result


def evaluate_signals(df: pd.DataFrame, settings: Settings = Settings()) -> pd.DataFrame:
    """Evaluate each symbol using three closed SQZMOM bars + true colored OBs.

    The signal candle is the third/last closed candle. At least one of the
    three candles forming the SQZMOM pivot must CLOSE inside a currently active
    matching Order Block. Only the original red/green OB zones are considered;
    no premium/discount/equilibrium substitutes are allowed.
    """
    d = _validate_ohlcv(df)
    sqz = squeeze_momentum(d, settings)
    ob = order_block_zones(d, settings)
    result = pd.concat([d, sqz, ob], axis=1)

    h0 = result["sqzmom"].shift(2)
    h1 = result["sqzmom"].shift(1)
    h2 = result["sqzmom"]
    three_positive = (h0 > 0) & (h1 > 0) & (h2 > 0)
    three_negative = (h0 < 0) & (h1 < 0) & (h2 < 0)

    # Three same-side CLOSED histogram bars; middle is the strict extreme.
    result["sell_peak"] = three_positive & (h1 > h0) & (h1 > h2)
    result["buy_trough"] = three_negative & (h1.abs() > h0.abs()) & (h1.abs() > h2.abs())

    # User's latest refinement: any ONE of the three pivot-forming candles
    # may provide the matching colored OB close confirmation.
    result["sell_ob_confirm"] = result["bear_ob_inside"].rolling(3, min_periods=3).max().fillna(0).astype(bool)
    result["buy_ob_confirm"] = result["bull_ob_inside"].rolling(3, min_periods=3).max().fillna(0).astype(bool)

    result["BUY"] = result["buy_trough"] & result["buy_ob_confirm"]
    result["SELL"] = result["sell_peak"] & result["sell_ob_confirm"]
    # Defensive mutual exclusion. A pathological overlapping-zone condition
    # cannot emit BUY and SELL together.
    conflict = result["BUY"] & result["SELL"]
    result.loc[conflict, ["BUY", "SELL"]] = False
    result["signal"] = np.select([result["BUY"], result["SELL"]], ["BUY", "SELL"], default="")
    result["sqzmom_peak_trough"] = h1
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
        description="Test the OB + SQZMOM signal engine on a closed-candle OHLCV CSV."
    )
    parser.add_argument("csv", help="CSV file with timestamp,open,high,low,close")
    parser.add_argument("--timeframe", default="30m", choices=["30m"])
    args = parser.parse_args()

    candles = load_ohlcv_csv(args.csv)
    output = evaluate_signals(candles)
    hits = output[output["signal"] != ""][
        ["close", "sqzmom", "bull_ob_inside", "bear_ob_inside", "signal"]
    ]
    print(f"Timeframe: {args.timeframe}")
    print(f"Candles evaluated: {len(output)}")
    print(f"Signals found: {len(hits)}")
    print(hits.tail(100).to_string())


def latest_confirmed_sqzmom_pivot(df: pd.DataFrame, settings: Settings = Settings()) -> dict | None:
    """Latest BTC SQZMOM pivot confirmed by exactly three CLOSED bars.

    A positive strict peak activates SELL bias; a negative strict trough
    activates BUY bias. The latest confirmed bias persists until the opposite
    BTC pivot is confirmed. No multiplier or amplitude threshold is used.
    """
    d = _validate_ohlcv(df)
    sqz = squeeze_momentum(d, settings)
    hist_series = sqz["sqzmom"] if isinstance(sqz, pd.DataFrame) else sqz
    hist = hist_series.to_numpy(float)
    latest = None
    for i in range(2, len(hist)):
        a, b, c = map(float, hist[i - 2:i + 1])
        if not np.isfinite([a, b, c]).all():
            continue
        if a > 0 and b > 0 and c > 0 and b > a and b > c:
            latest = {"bias": "SELL", "pivot_value": b, "confirm_index": i}
        elif a < 0 and b < 0 and c < 0 and b < a and b < c:
            latest = {"bias": "BUY", "pivot_value": b, "confirm_index": i}
    if latest is None:
        return None
    latest["confirm_time"] = d.index[latest["confirm_index"]]
    latest["latest_histogram"] = float(hist[-1]) if len(hist) else float("nan")
    return latest
