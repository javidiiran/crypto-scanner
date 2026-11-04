"""AB engine: Python port of the supplied SQZMOM_LB and OB/FVG Pine scripts.
Only completed candles should be passed to evaluate_symbol().
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from settings import (PIVOT_LENGTH, OB_SIZE_MULTIPLIER, FVG_SIZE_MULTIPLIER,
                      CANDLE_SIZE_PERIOD, FVG_VISIBLE_BARS, MAX_ZONE_AGE_BARS,
                      RECENT_TOUCH_CANDLES)

@dataclass
class Zone:
    top: float
    bottom: float
    bullish: bool
    kind: str
    created_index: int
    active: bool = True
    expires_index: int | None = None


def _sma(a: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(a, dtype="float64").rolling(n, min_periods=n).mean().to_numpy()


def _linreg_endpoint(a: np.ndarray, n: int) -> np.ndarray:
    """Equivalent to Pine ta.linreg(source, length, 0)."""
    out = np.full(len(a), np.nan, dtype=float)
    x = np.arange(n, dtype=float)
    xm = (n - 1) / 2.0
    denom = float(np.sum((x - xm) ** 2))
    for end in range(n - 1, len(a)):
        w = a[end - n + 1:end + 1]
        if not np.isfinite(w).all():
            continue
        ym = float(w.mean())
        slope = float(np.sum((x - xm) * (w - ym)) / denom) if denom else 0.0
        out[end] = ym + slope * ((n - 1) - xm)
    return out


def sqzmom_lb(df: pd.DataFrame) -> pd.DataFrame:
    """LazyBear source defaults; note original source uses multKC for BB dev."""
    h = df.high.to_numpy(float); l = df.low.to_numpy(float); c = df.close.to_numpy(float)
    n = len(df)
    # BB: source explicitly says dev = multKC * stdev (1.5), not mult (2.0).
    basis = _sma(c, 20)
    stdev = pd.Series(c).rolling(20, min_periods=20).std(ddof=0).to_numpy()
    dev = 1.5 * stdev
    upper_bb, lower_bb = basis + dev, basis - dev
    # KC True Range and SMA.
    prev = np.roll(c, 1); prev[0] = np.nan
    tr = np.maximum(h-l, np.maximum(np.abs(h-prev), np.abs(l-prev)))
    tr[0] = h[0]-l[0]
    rangema = _sma(tr, 20)
    upper_kc, lower_kc = basis + rangema*1.5, basis-rangema*1.5
    sqz_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)
    sqz_off = (lower_bb < lower_kc) & (upper_bb > upper_kc)
    no_sqz = ~(sqz_on | sqz_off)
    highest = pd.Series(h).rolling(20, min_periods=20).max().to_numpy()
    lowest = pd.Series(l).rolling(20, min_periods=20).min().to_numpy()
    center = ((highest + lowest) / 2.0 + basis) / 2.0
    val = _linreg_endpoint(c-center, 20)
    prev_val = np.roll(val, 1); prev_val[0] = 0.0
    colors = np.full(n, "", dtype=object)
    for i, v in enumerate(val):
        if not np.isfinite(v): continue
        pv = prev_val[i] if np.isfinite(prev_val[i]) else 0.0
        if v > 0: colors[i] = "lime" if v > pv else "green"
        else: colors[i] = "red" if v < pv else "maroon"
    return pd.DataFrame({"sqzmom":val,"sqzmom_color":colors,"sqz_on":sqz_on,
                         "sqz_off":sqz_off,"no_sqz":no_sqz}, index=df.index)


def _is_green(df: pd.DataFrame, i: int) -> bool:
    return float(df.iloc[i].close) > float(df.iloc[i].open)


def _is_pivot(values: np.ndarray, p: int, left: int, right: int, low_pivot: bool) -> bool:
    a, b = p-left, p+right+1
    if a < 0 or b > len(values): return False
    w = values[a:b]; target = values[p]
    if not np.isfinite(target): return False
    extreme = np.nanmin(w) if low_pivot else np.nanmax(w)
    if target != extreme: return False
    # Pine pivot tie behavior: reject equal extremes to the right.
    return not np.any(w[(p-a+1):] == target)


def _add_zone(zones: list[Zone], z: Zone) -> None:
    same = [i for i, old in enumerate(zones) if old.kind == z.kind]
    if len(same) >= 500:
        zones.pop(same[0])
    zones.append(z)


def build_zones(df: pd.DataFrame) -> list[Zone]:
    """Reproduce OB/FVG creation, break deletion and 500-bar cleanup."""
    n=len(df)
    if n < CANDLE_SIZE_PERIOD + PIVOT_LENGTH*2 + 2: return []
    o=df.open.to_numpy(float); h=df.high.to_numpy(float); l=df.low.to_numpy(float); c=df.close.to_numpy(float)
    body=np.abs(c-o); prior_body=np.roll(body,1); prior_body[0]=np.nan
    avg_body=_sma(prior_body,CANDLE_SIZE_PERIOD)
    zones: list[Zone]=[]
    pl=PIVOT_LENGTH
    for i in range(n):
        avg=avg_body[i]
        # Pivots become known only after `pl` right-side candles have closed.
        p=i-pl
        if p >= pl and np.isfinite(avg):
            # Pivot low OB logic from the Pine source.
            if _is_pivot(l,p,pl,pl,True):
                pg=_is_green(df,p); prevg=_is_green(df,p-1); nextg=_is_green(df,p+1)
                src=None; bull=False
                if not pg and nextg: src=p; bull=True
                elif pg and not prevg: src=p-1; bull=True
                elif pg and prevg and nextg: src=p-1; bull=False
                if src is not None:
                    top=float(h[src]); bot=float(l[src])
                    if top-bot >= avg*OB_SIZE_MULTIPLIER:
                        _add_zone(zones,Zone(top,bot,bull,"OB",src))
            # Pivot high OB logic from the Pine source.
            if _is_pivot(h,p,pl,pl,False):
                pg=_is_green(df,p); prevg=_is_green(df,p-1); nextg=_is_green(df,p+1)
                src=None; bull=False
                if pg and not nextg: src=p; bull=False
                elif not pg and prevg: src=p-1; bull=False
                elif not pg and not prevg and not nextg: src=p-1; bull=True
                if src is not None:
                    top=float(h[src]); bot=float(l[src])
                    if top-bot >= avg*OB_SIZE_MULTIPLIER:
                        _add_zone(zones,Zone(top,bot,bull,"OB",src))
        # FVG: visible for configured bars; invalidated earlier if broken.
        if np.isfinite(avg) and i >= 2:
            if l[i] > h[i-2]:
                top=float(l[i]); bot=float(h[i-2])
                if top-bot >= avg*FVG_SIZE_MULTIPLIER:
                    _add_zone(zones,Zone(top,bot,True,"FVG",i-2,True,(i-2)+FVG_VISIBLE_BARS))
            if h[i] < l[i-2]:
                top=float(l[i-2]); bot=float(h[i])
                if top-bot >= avg*FVG_SIZE_MULTIPLIER:
                    _add_zone(zones,Zone(top,bot,False,"FVG",i-2,True,(i-2)+FVG_VISIBLE_BARS))
        # Pine removes broken FVGs and OBs using candle close.
        for z in zones:
            if not z.active: continue
            if z.kind == "FVG":
                if (z.bullish and c[i] < z.bottom) or ((not z.bullish) and c[i] > z.top): z.active=False
            elif z.kind == "OB":
                if (z.bullish and c[i] < z.bottom) or ((not z.bullish) and c[i] > z.top): z.active=False
            if z.active and i-z.created_index > MAX_ZONE_AGE_BARS: z.active=False
    return zones


def candle_touches_zone(candle: pd.Series, zone: Zone) -> bool:
    return float(candle.high) >= zone.bottom and float(candle.low) <= zone.top


def evaluate_symbol(df: pd.DataFrame, as_of_time_ms: int | None = None) -> dict[str, Any] | None:
    """Signal on latest closed candle (or the candle at as_of_time_ms); confirm in last four."""
    if as_of_time_ms is not None:
        df=df[df.time_ms <= int(as_of_time_ms)].reset_index(drop=True)
    if len(df) < 60: return None
    mom=sqzmom_lb(df); last=len(df)-1
    color=str(mom.iloc[last].sqzmom_color)
    if color == "maroon": side, bullish = "BUY", True
    elif color == "green": side, bullish = "SELL", False
    else: return None
    zones=build_zones(df)
    recent=df.iloc[max(0,len(df)-RECENT_TOUCH_CANDLES):]
    matching=[z for z in zones if z.active and z.bullish == bullish]
    touched=[]
    for _, candle in recent.iterrows():
        for z in matching:
            if candle_touches_zone(candle,z): touched.append(z)
    if not touched: return None
    unique={(z.kind,z.created_index,z.top,z.bottom):z for z in touched}
    return {"side":side,"sqzmom_color":color,"sqzmom_value":float(mom.iloc[last].sqzmom),
            "candle_time":int(df.iloc[last].time_ms),"close":float(df.iloc[last].close),
            "zones_confirmed":sorted({z.kind for z in unique.values()}),"zone_count":len(unique)}
