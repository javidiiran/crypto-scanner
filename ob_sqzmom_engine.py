"""SQZMOM [LazyBear] + MFI confirmation engine; closed candles only."""
from __future__ import annotations
import numpy as np
import pandas as pd

BB_LENGTH=20
KC_LENGTH=20
KC_MULT=1.5
MFI_LENGTH=14
MFI_MIDLINE=50.0

def _validate(df):
    d=df.copy()
    d.columns=[str(c).lower() for c in d.columns]
    required={"open","high","low","close","volume"}
    missing=required-set(d.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
    for c in required: d[c]=pd.to_numeric(d[c],errors="coerce")
    return d.dropna(subset=list(required)).sort_index()

def squeeze_momentum(d):
    close, high, low=d.close, d.high, d.low
    basis=close.rolling(BB_LENGTH).mean()
    # Preserve the supplied LazyBear source's KC multiplier in BB deviation.
    dev=KC_MULT*close.rolling(BB_LENGTH).std(ddof=0)
    upper_bb,lower_bb=basis+dev,basis-dev
    prev=close.shift(1)
    tr=pd.concat([(high-low),(high-prev).abs(),(low-prev).abs()],axis=1).max(axis=1)
    range_ma=tr.rolling(KC_LENGTH).mean()
    ma=close.rolling(KC_LENGTH).mean()
    upper_kc,lower_kc=ma+range_ma*KC_MULT,ma-range_ma*KC_MULT
    hh, ll=high.rolling(KC_LENGTH).max(),low.rolling(KC_LENGTH).min()
    src=close-(((hh+ll)/2.0+ma)/2.0)
    x=np.arange(KC_LENGTH,dtype=float); xm=x.mean(); denom=((x-xm)**2).sum()
    def linreg(v):
        if len(v)!=KC_LENGTH or not np.isfinite(v).all(): return np.nan
        ym=v.mean(); slope=((x-xm)*(v-ym)).sum()/denom
        return float(ym-slope*xm+slope*(KC_LENGTH-1))
    return src.rolling(KC_LENGTH).apply(linreg,raw=True)

def money_flow_index(d):
    typical=(d.high+d.low+d.close)/3.0
    raw=typical*d.volume
    direction=typical.diff()
    positive=raw.where(direction>0,0.0).rolling(MFI_LENGTH).sum()
    negative=raw.where(direction<0,0.0).rolling(MFI_LENGTH).sum()
    ratio=positive/negative.replace(0,np.nan)
    mfi=100-(100/(1+ratio))
    mfi=mfi.mask((negative==0)&(positive>0),100.0)
    mfi=mfi.mask((positive==0)&(negative>0),0.0)
    mfi=mfi.mask((positive==0)&(negative==0),50.0)
    return mfi

def evaluate_signals(df):
    d=_validate(df)
    out=d.copy()
    out["sqzmom"]=squeeze_momentum(d)
    out["mfi"]=money_flow_index(d)
    h0,h1,h2=out.sqzmom.shift(2),out.sqzmom.shift(1),out.sqzmom
    # Three closed bars: middle bar forms the local peak/trough.
    out["sqz_buy"]= (h0<0)&(h1<0)&(h2<0)&(h1.abs()>h0.abs())&(h1.abs()>h2.abs())
    out["sqz_sell"]= (h0>0)&(h1>0)&(h2>0)&(h1.abs()>h0.abs())&(h1.abs()>h2.abs())
    out["BUY"]=out.sqz_buy & (out.mfi>MFI_MIDLINE)
    out["SELL"]=out.sqz_sell & (out.mfi<MFI_MIDLINE)
    out["signal"]=np.select([out.BUY,out.SELL],["BUY","SELL"],default="")
    return out
