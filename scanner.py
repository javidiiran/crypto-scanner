"""KCEX Spot 30m scanner. SQZMOM issues; MFI confirms; BTC is annotation only.

IMPORTANT: The public KCEX REST documentation currently located online is for
api.kkcex.com (legacy domain). Set KCEX_API_BASE only to a host verified for
this KCEX account/site. This script does not place orders.
"""
from __future__ import annotations
import json, os, time
from pathlib import Path
from datetime import datetime, timezone
import requests
import pandas as pd
from ob_sqzmom_engine import evaluate_signals

BASE=(os.getenv("KCEX_API_BASE", "").strip() or "https://api.kkcex.com").rstrip("/")
TF="30min"
CANDLE_SECONDS=1800
STATE=Path("state.json")
S=requests.Session(); S.headers.update({"User-Agent":"Mozilla/5.0 KCEX-Spot-Signal-Scanner/1.0"})

def get_json(path,params=None):
    r=S.get(BASE+path,params=params,timeout=25); r.raise_for_status(); data=r.json()
    if isinstance(data,dict) and data.get("code") not in (None,200,"200"):
        raise RuntimeError(f"KCEX API error: {data}")
    return data

def fetch_markets():
    data=get_json("/data/api/v1/getMarketConfig")
    if not isinstance(data,dict) or not data: raise RuntimeError("KCEX market config response is empty/unexpected")
    return sorted(k.lower() for k in data if k.lower().endswith("_usdt"))

def fetch_candles(market):
    payload=get_json("/data/api/v1/getKLine",{"market":market,"type":TF,"since":0})
    rows=payload.get("datas",[]) if isinstance(payload,dict) else []
    if not isinstance(rows,list) or len(rows)<80: return pd.DataFrame()
    now=int(time.time())
    parsed=[]
    for row in rows:
        if not isinstance(row,(list,tuple)) or len(row)<6: continue
        ts=int(row[0]);
        # Legacy API timestamps are seconds; discard the currently forming candle.
        if ts>10**12: ts//=1000
        if ts+CANDLE_SECONDS>now: continue
        parsed.append([pd.to_datetime(ts,unit="s",utc=True),float(row[1]),float(row[2]),float(row[3]),float(row[4]),float(row[5])])
    if len(parsed)<80: return pd.DataFrame()
    d=pd.DataFrame(parsed,columns=["timestamp","open","high","low","close","volume"]).drop_duplicates("timestamp").set_index("timestamp").sort_index()
    return d.tail(300)

def evaluate(market):
    candles=fetch_candles(market)
    if candles.empty: return None
    row=evaluate_signals(candles).iloc[-1]
    direction=str(row.signal)
    return (direction,row) if direction in ("BUY","SELL") else None

def load_state():
    try: return set(json.loads(STATE.read_text()).get("sent",[]))
    except Exception: return set()
def save_state(sent): STATE.write_text(json.dumps({"sent":sorted(sent)[-10000:]},indent=2))

def notify(market,direction,row,btc_direction):
    token=os.environ["TELEGRAM_BOT_TOKEN"]; chat=os.environ["TELEGRAM_CHAT_ID"]
    opposition=(market!="btc_usdt" and btc_direction in ("BUY","SELL") and direction!=btc_direction)
    label="⚠️ خلاف جهت سیگنال BTC" if opposition else ("هم‌جهت با BTC" if market!="btc_usdt" and btc_direction in ("BUY","SELL") else "وضعیت BTC خنثی/بدون سیگنال")
    lines=["🟢 BUY" if direction=="BUY" else "🔴 SELL",f"Market: {market.upper()}","Exchange: KCEX Spot",f"Timeframe: 30m",f"Close: {float(row.close):.10g}",f"SQZMOM: {float(row.sqzmom):.10g}",f"MFI(14): {float(row.mfi):.2f}"]
    if market!="btc_usdt": lines.extend([f"BTC direction: {btc_direction or 'NEUTRAL'}",label])
    lines.append("Closed candle: "+row.name.strftime("%Y-%m-%d %H:%M UTC"))
    resp=S.post(f"https://api.telegram.org/bot{token}/sendMessage",data={"chat_id":chat,"text":"\n".join(lines)},timeout=25); resp.raise_for_status()

def main():
    sent=load_state()
    markets=fetch_markets()
    if "btc_usdt" not in markets: raise RuntimeError("BTC_USDT not found in KCEX Spot market config")
    btc=evaluate("btc_usdt"); btc_direction=btc[0] if btc else "NEUTRAL"
    print(f"KCEX Spot markets: {len(markets)}; BTC reference: {btc_direction}")
    checked=alerts=0
    for market in markets:
        try:
            found=btc if market=="btc_usdt" else evaluate(market)
            checked+=1
            if not found: continue
            direction,row=found
            key=f"{market}|30m|{int(row.name.timestamp())}|{direction}"
            if key in sent: continue
            notify(market,direction,row,btc_direction); sent.add(key); save_state(sent); alerts+=1
            print("ALERT",key)
        except Exception as e: print(f"SKIP {market}: {type(e).__name__}: {e}")
    print(f"Finished. checked={checked}; alerts={alerts}; BTC={btc_direction}")
if __name__=="__main__": main()
