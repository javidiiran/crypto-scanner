"""SQZMOM + AlgoAlpha Dynamic Profile signal engine for Binance Spot.

Signals are evaluated only on CLOSED 30-minute candles.
BUY: negative SQZMOM histogram contracts on the latest candle and at least
     one of the last five candle ranges touches or approaches an active
     AlgoAlpha-style Trough zone (green/support).
SELL: positive SQZMOM histogram contracts on the latest candle and at least
      one of the last five candle ranges touches or approaches an active
      Peak zone (red/resistance).

Zone distance bands measured from candle range to zone:
  0.00%       overlap/touch (strong)
  >0.00-0.25% near (good)
  >0.25-0.50% near (weak but accepted)
  >0.50%      rejected

The Dynamic Profile calculation is a Python port of the supplied AlgoAlpha
source's fixed-window volume-at-price node logic. It is an approximation of
TradingView's internal bar-volume allocation, not a byte-for-byte Pine runtime.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Settings:
    # AlgoAlpha settings agreed for this scanner
    profile_lookback: int = 240
    profile_resolution: int = 64
    profile_refresh_bars: int = 5
    peak_count: int = 2
    trough_count: int = 1
    peak_detection_pct: float = 7.0
    trough_detection_pct: float = 7.0
    minimum_node_volume_pct: float = 1.0
    smoothing_radius: int = 1
    zone_radius: int = 1
    match_tolerance_rows: float = 0.75
    keep_old_active: bool = True
    stop_on_break: bool = True
    level_expiry_bars: int = 500
    max_levels: int = 150
    # SQZMOM LazyBear defaults
    length: int = 20
    bb_mult: float = 2.0
    kc_mult: float = 1.5
    use_true_range: bool = True
    # User's zone confirmation settings
    pattern_candles: int = 5
    near_good_pct: float = 0.25
    near_max_pct: float = 0.50


def _validate(df: pd.DataFrame) -> pd.DataFrame:
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


def squeeze_momentum(df: pd.DataFrame, s: Settings = Settings()) -> pd.DataFrame:
    d = _validate(df)
    close, high, low = d.close, d.high, d.low
    basis = close.rolling(s.length).mean()
    # Preserve the supplied LazyBear script's kc_mult use in BB deviation.
    dev = s.kc_mult * close.rolling(s.length).std(ddof=0)
    upper_bb, lower_bb = basis + dev, basis - dev
    prev = close.shift(1)
    tr = pd.concat([(high-low), (high-prev).abs(), (low-prev).abs()], axis=1).max(axis=1)
    rng = tr if s.use_true_range else high-low
    range_ma = rng.rolling(s.length).mean()
    ma = close.rolling(s.length).mean()
    upper_kc, lower_kc = ma + range_ma*s.kc_mult, ma-range_ma*s.kc_mult
    highest, lowest = high.rolling(s.length).max(), low.rolling(s.length).min()
    src = close - (((highest+lowest)/2.0 + ma)/2.0)
    x = np.arange(s.length, dtype=float)
    xm = x.mean(); denom = ((x-xm)**2).sum()
    def linreg_last(a: np.ndarray) -> float:
        if len(a) != s.length or not np.isfinite(a).all(): return np.nan
        ym = a.mean(); slope = ((x-xm)*(a-ym)).sum()/denom
        return float(ym-slope*xm+slope*(s.length-1))
    hist = src.rolling(s.length).apply(linreg_last, raw=True)
    out = pd.DataFrame(index=d.index)
    out["sqzmom"] = hist
    out["sqz_on"] = (lower_bb > lower_kc) & (upper_bb < upper_kc)
    out["sqz_off"] = (lower_bb < lower_kc) & (upper_bb > upper_kc)
    out["buy_momentum"] = (hist < 0) & (hist.abs() < hist.shift(1).abs())
    out["sell_momentum"] = (hist > 0) & (hist.abs() < hist.shift(1).abs())
    return out



def money_flow_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Standard MFI; informational grading only, never a signal filter."""
    d = _validate(df)
    typical = (d.high + d.low + d.close) / 3.0
    raw_flow = typical * d.volume
    direction = typical.diff()
    positive = raw_flow.where(direction > 0, 0.0)
    negative = raw_flow.where(direction < 0, 0.0)
    pos_sum = positive.rolling(period, min_periods=period).sum()
    neg_sum = negative.rolling(period, min_periods=period).sum()
    ratio = pos_sum / neg_sum.replace(0.0, np.nan)
    mfi = 100.0 - (100.0 / (1.0 + ratio))
    mfi = mfi.where(neg_sum != 0.0, 100.0)
    mfi = mfi.where(pos_sum != 0.0, 0.0)
    mfi = mfi.where(~((pos_sum == 0.0) & (neg_sum == 0.0)), 50.0)
    return mfi.rename("mfi")

def _allocate_bar(rows: np.ndarray, lo: float, hi: float, vol: float,
                  range_low: float, step: float) -> None:
    n = len(rows)
    first = max(0, min(n-1, int(np.floor((lo-range_low)/step))))
    last = max(first, min(n-1, int(np.floor((hi-range_low)/step))))
    if hi <= lo or first == last:
        rows[first] += vol
        return
    span = hi-lo
    rows[first] += vol * max(0.0, min(1.0, (range_low+(first+1)*step-lo)/span))
    rows[last] += vol * max(0.0, min(1.0, (hi-range_low-last*step)/span))
    if last > first+1:
        rows[first+1:last] += vol*step/span


def _smooth(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0: return values.copy()
    out = np.empty_like(values)
    for i in range(len(values)):
        a, b = max(0, i-radius), min(len(values), i+radius+1)
        out[i] = values[a:b].mean()
    return out


def _is_node(values: np.ndarray, index: int, radius: int, peak: bool) -> bool:
    if not peak and (index == 0 or index == len(values)-1): return False
    value = values[index]; contrast = False
    for j in range(max(0,index-radius), min(len(values)-1,index+radius)+1):
        if j == index: continue
        other = values[j]
        if (other > value if peak else other < value) or (other == value and j < index):
            return False
        contrast = contrast or (other < value if peak else other > value)
    return contrast


def dynamic_profile_zones(df: pd.DataFrame, s: Settings = Settings()) -> list[list[dict]]:
    """Return active Peak (SELL) and Trough (BUY) zones at each candle."""
    d = _validate(df)
    highs, lows = d.high.to_numpy(float), d.low.to_numpy(float)
    closes, vols = d.close.to_numpy(float), d.volume.to_numpy(float)
    n, res = len(d), s.profile_resolution
    levels: list[dict] = []
    zones_by_bar: list[list[dict]] = [[] for _ in range(n)]

    for t in range(n):
        if t >= 9 and (t % s.profile_refresh_bars == 0 or t == 9):
            start = max(0, t-s.profile_lookback+1)
            rh, rl = float(highs[start:t+1].max()), float(lows[start:t+1].min())
            step = max((rh-rl)/res, 1e-12)
            raw = np.zeros(res, dtype=float)
            for k in range(start, t+1):
                if vols[k] > 0:
                    _allocate_bar(raw, lows[k], highs[k], vols[k], rl, step)
            if raw.max() > 0:
                smooth = _smooth(raw, s.smoothing_radius)
                peak_radius = max(1, int(round(res*s.peak_detection_pct/100)))
                trough_radius = max(1, int(round(res*s.trough_detection_pct/100)))
                candidates: list[dict] = []
                used: set[int] = set()
                peak_rows = 0; trough_rows = 0
                # Pine scans smoothed rows descending for peaks and ascending for troughs.
                for row in np.argsort(-smooth, kind="stable"):
                    row = int(row)
                    if peak_rows < s.peak_count and raw[row] > 0 and raw[row]/raw.max()*100 >= s.minimum_node_volume_pct:
                        if row not in used and _is_node(smooth, row, peak_radius, True):
                            candidates.append({"kind":"peak", "row":row}); used.add(row); peak_rows += 1
                for row in np.argsort(smooth, kind="stable"):
                    row = int(row)
                    if trough_rows < s.trough_count and raw[row] > 0 and raw[row]/raw.max()*100 >= s.minimum_node_volume_pct:
                        if row not in used and _is_node(smooth, row, trough_radius, False):
                            candidates.append({"kind":"trough", "row":row}); used.add(row); trough_rows += 1
                for c in candidates:
                    row=c["row"]; a=max(0,row-s.zone_radius); b=min(res-1,row+s.zone_radius)
                    top=rl+(b+1)*step; bottom=rl+a*step; price=rl+(row+0.5)*step
                    tolerance=max(step*s.match_tolerance_rows, 1e-12)
                    matches=[z for z in levels if z["active"] and z["kind"]==c["kind"] and z["last_seen"]!=t and abs(z["price"]-price)<=tolerance]
                    if matches:
                        min(matches,key=lambda z:abs(z["price"]-price))["last_seen"] = t
                    elif not any(z["active"] and top>=z["bottom"] and bottom<=z["top"] for z in levels):
                        side = 1 if closes[t] > top else -1 if closes[t] < bottom else 0
                        levels.append({"kind":c["kind"],"price":price,"top":top,"bottom":bottom,
                                       "born":t,"last_seen":t,"active":True,"side":side})
        # Maintain old active nodes and apply close-break/expiry behavior.
        for z in levels:
            if not z["active"]: continue
            if t-z["born"] >= s.level_expiry_bars:
                z["active"] = False; continue
            if s.stop_on_break:
                if z["side"] == 1 and closes[t] < z["bottom"]:
                    z["active"] = False; continue
                if z["side"] == -1 and closes[t] > z["top"]:
                    z["active"] = False; continue
            if closes[t] > z["top"]: z["side"] = 1
            elif closes[t] < z["bottom"]: z["side"] = -1
        # Only the semantic families requested: Trough=green BUY, Peak=red SELL.
        zones_by_bar[t] = [dict(z) for z in levels if z["active"] and z["kind"] in ("peak","trough")]
        if len(levels) > s.max_levels:
            # Match the source's bounded storage: discard oldest stopped first.
            while len(levels) > s.max_levels:
                idx = next((i for i,z in enumerate(levels) if not z["active"]), 0)
                levels.pop(idx)
    return zones_by_bar


def _gap_pct(candle_low: float, candle_high: float, zone: dict) -> float:
    if candle_high >= zone["bottom"] and candle_low <= zone["top"]:
        return 0.0
    gap = zone["bottom"]-candle_high if candle_high < zone["bottom"] else candle_low-zone["top"]
    reference = max((zone["top"]+zone["bottom"])/2.0, 1e-12)
    return max(0.0, gap/reference*100.0)


def evaluate_signals(df: pd.DataFrame, settings: Settings = Settings()) -> pd.DataFrame:
    d = _validate(df)
    sqz = squeeze_momentum(d, settings)
    zones = dynamic_profile_zones(d, settings)
    out = pd.concat([d, sqz], axis=1)
    # MFI is attached for A-D grading only; it does not affect signal creation.
    out["mfi"] = money_flow_index(d, 14)
    signals=[]; matched_kind=[]; matched_top=[]; matched_bottom=[]; matched_gap=[]; matched_age=[]; strength=[]
    for i in range(len(out)):
        sig = ""
        if bool(out.buy_momentum.iloc[i]): sig = "BUY"
        elif bool(out.sell_momentum.iloc[i]): sig = "SELL"
        best=None
        if sig:
            wanted = "trough" if sig == "BUY" else "peak"
            first=max(0,i-settings.pattern_candles+1)
            # Test each of the last five candle ranges against zones active at that candle.
            for j in range(i, first-1, -1):
                for z in zones[j]:
                    if z["kind"] != wanted: continue
                    gap=_gap_pct(float(d.low.iloc[j]), float(d.high.iloc[j]), z)
                    if gap <= settings.near_max_pct:
                        candidate=(gap, i-j, z)
                        if best is None or candidate[:2] < best[:2]: best=candidate
            if best is None: sig=""
        signals.append(sig)
        if best:
            gap, age, z=best
            matched_kind.append(z["kind"]); matched_top.append(z["top"]); matched_bottom.append(z["bottom"])
            matched_gap.append(round(gap,6)); matched_age.append(age)
            strength.append("TOUCH" if gap == 0 else "GOOD" if gap <= settings.near_good_pct else "WEAK")
        else:
            matched_kind.append(""); matched_top.append(np.nan); matched_bottom.append(np.nan)
            matched_gap.append(np.nan); matched_age.append(-1); strength.append("")
    out["signal"]=signals
    out["profile_zone_type"]=matched_kind
    out["profile_zone_top"]=matched_top
    out["profile_zone_bottom"]=matched_bottom
    out["profile_gap_pct"]=matched_gap
    out["profile_pattern_candle_age"]=matched_age
    out["profile_confirmation"]=strength
    return out


def load_ohlcv_csv(path: str) -> pd.DataFrame:
    d=pd.read_csv(path)
    if "timestamp" in d.columns:
        d["timestamp"]=pd.to_datetime(d.timestamp, utc=True); d=d.set_index("timestamp")
    return _validate(d)

if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser(description="Test Binance Spot SQZMOM + Dynamic Profile engine")
    p.add_argument("csv", help="CSV with timestamp,open,high,low,close,volume")
    args=p.parse_args()
    result=evaluate_signals(load_ohlcv_csv(args.csv))
    hits=result[result.signal!=""][ ["close","sqzmom","signal","profile_zone_type","profile_gap_pct","profile_confirmation"] ]
    print(f"Candles: {len(result)} | Signals: {len(hits)}")
    print(hits.tail(100).to_string())
