from __future__ import annotations
import time
import requests
import pandas as pd
from settings import CANDLE_MS, CANDLES_TO_FETCH, REQUEST_TIMEOUT_SECONDS

BASE_CHARTS = "https://futures.kraken.com/api/charts/v1"
BASE_REST = "https://futures.kraken.com/derivatives/api/v3"
SESSION = requests.Session()
SESSION.headers.update({"Accept":"application/json", "User-Agent":"Scanner3-Kraken-Futures/1.0"})


def get_perpetual_symbols() -> list[str]:
    """List tradable perpetual futures only (PF_ linear and PI_ inverse)."""
    r=SESSION.get(f"{BASE_REST}/instruments",timeout=REQUEST_TIMEOUT_SECONDS); r.raise_for_status()
    payload=r.json()
    if payload.get("result") == "error": raise RuntimeError(f"Kraken instruments error: {payload}")
    out=[]
    for item in payload.get("instruments",[]):
        symbol=str(item.get("symbol", ""))
        if not symbol.startswith(("PF_","PI_")): continue
        if item.get("tradeable") is False: continue
        status=str(item.get("status", "")).lower()
        if status and status not in {"tradeable","open","online"}: continue
        out.append(symbol)
    return sorted(set(out))


def fetch_ohlc(symbol: str, limit: int=CANDLES_TO_FETCH) -> pd.DataFrame:
    now=int(time.time()); start=now-limit*900
    url=f"{BASE_CHARTS}/trade/{symbol}/15m"
    r=SESSION.get(url,params={"from":start,"to":now,"count":limit},timeout=REQUEST_TIMEOUT_SECONDS)
    r.raise_for_status(); payload=r.json()
    rows=payload.get("candles",[])
    if not rows: return pd.DataFrame()
    df=pd.DataFrame(rows)
    if "time" not in df: raise ValueError(f"Kraken response has no candle time for {symbol}")
    # Kraken chart candle `time` is explicitly epoch milliseconds.
    df["time_ms"]=pd.to_numeric(df["time"],errors="coerce")
    for col in ("open","high","low","close","volume"):
        if col in df: df[col]=pd.to_numeric(df[col],errors="coerce")
    df=df.dropna(subset=["time_ms","open","high","low","close"])
    df["time_ms"]=df["time_ms"].astype("int64")
    df=df.sort_values("time_ms").drop_duplicates("time_ms").reset_index(drop=True)
    # Exclude current forming candle and any malformed future candle.
    now_ms=int(time.time()*1000)
    df=df[(df.time_ms + CANDLE_MS <= now_ms) & (df.time_ms <= now_ms)].reset_index(drop=True)
    return df
