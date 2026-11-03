"""CNV + RSI engine for Kraken Futures Scanner 2. Only closed candles are used."""
from __future__ import annotations
import numpy as np
import pandas as pd


def validate_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    d=frame.copy(); d.columns=[str(c).lower() for c in d.columns]
    required={'open','high','low','close','volume'}
    missing=required-set(d.columns)
    if missing: raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    for c in required: d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.dropna(subset=list(required)); d=d[~d.index.duplicated(keep='last')].sort_index()
    return d


def rsi_wilder(close: pd.Series, length: int=14) -> pd.Series:
    delta=close.astype(float).diff(); gain=delta.clip(lower=0); loss=-delta.clip(upper=0)
    def rma(x):
        a=x.to_numpy(float); out=np.full(len(a),np.nan)
        if len(a)<=length: return pd.Series(out,index=x.index)
        seed=a[1:length+1]
        if not np.isfinite(seed).all(): return pd.Series(out,index=x.index)
        out[length]=seed.mean()
        for i in range(length+1,len(a)):
            if np.isfinite(a[i]): out[i]=(out[i-1]*(length-1)+a[i])/length
        return pd.Series(out,index=x.index)
    ag=rma(gain); al=rma(loss)
    rs=ag/al.replace(0,np.nan); value=100-100/(1+rs)
    value=value.mask((al==0)&(ag>0),100).mask((ag==0)&(al>0),0).mask((ag==0)&(al==0),50)
    return value


def cnv(frame: pd.DataFrame) -> pd.DataFrame:
    d=validate_ohlcv(frame)
    change=d.close.diff()
    signed_volume=pd.Series(np.where(change>0,d.volume,np.where(change<0,-d.volume,0.0)),index=d.index)
    cumulative=signed_volume.cumsum()
    cnv_tb=cumulative-cumulative.rolling(20,min_periods=20).mean()
    # Pine rule: blue only when CNV trend is strictly above zero; otherwise pink.
    blue=cnv_tb>0
    return pd.DataFrame({'cnv':cumulative,'cnv_sma20':cumulative.rolling(20,min_periods=20).mean(),
                         'cnv_tb':cnv_tb,'blue':blue.fillna(False).astype(bool),
                         'pink':(~blue).astype(bool)},index=d.index)


def analyze_market(frame: pd.DataFrame) -> dict:
    d=validate_ohlcv(frame); c=cnv(d); rsi=rsi_wilder(d.close,14)
    signals=[]
    # A signal is assigned to the third candle of the exact three-candle sequence.
    for i in range(2,len(d)):
        colors=[bool(c.blue.iloc[j]) for j in (i-2,i-1,i)]
        vals=rsi.iloc[i-2:i+1]
        if colors==[False,True,True] and bool((vals<50).any()): direction='BUY'
        elif colors==[True,False,False] and bool((vals>50).any()): direction='SELL'
        else: continue
        stamp=d.index[i]
        signals.append({'direction':direction,'signal_index':i,'timestamp':stamp,
          'timestamp_ms':int(pd.Timestamp(stamp).timestamp()*1000),
          'signal_time':pd.Timestamp(stamp).strftime('%Y-%m-%d %H:%M UTC'),
          'price':float(d.close.iloc[i]),'rsi':float(vals.iloc[-1]),
          'rsi_values':[None if pd.isna(x) else float(x) for x in vals],
          'rsi_condition_value':float(vals[vals<50].iloc[0]) if direction=='BUY' else float(vals[vals>50].iloc[0]),
          'cnv_colors':['BLUE' if x else 'PINK' for x in colors]})
    return {'candles':d,'cnv':c,'rsi':rsi,'signals':signals,
            'latest_rsi':float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else float('nan'),
            'latest_cnv_color':'BLUE' if bool(c.blue.iloc[-1]) else 'PINK'}
