"""AB — BTC-led Scanner 3 entrypoint for Kraken Futures 30m."""
from __future__ import annotations
import json, os, time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from ob_sqzmom_engine import evaluate_symbol, sqzmom_lb, build_zones, candle_touches_zone
from kraken_client import get_perpetual_symbols, fetch_ohlc
from notifier import send_telegram, format_alert, format_btc_alert, format_stop_alert
from settings import REQUEST_PAUSE_SECONDS, STATE_RETENTION_DAYS, MIN_CANDLES

ROOT=Path(__file__).resolve().parent
STATE_FILE=ROOT/"state.json"
DARK_BUY="maroon"
DARK_SELL="green"
STOP_BUY="red"       # BUY license stops on one closed bright-red SQZMOM bar
STOP_SELL="lime"     # SELL license stops on one closed bright-green SQZMOM bar
LICENSE_BARS=4


def _load_local_env() -> None:
    env_file=ROOT/".env"
    if not env_file.exists(): return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line=line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        key,value=line.split("=",1)
        os.environ.setdefault(key.strip(),value.strip().strip('"').strip("'"))


def load_state() -> dict:
    if not STATE_FILE.exists(): return {"sent":{},"btc_control":{}}
    try:
        state=json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if not isinstance(state,dict): state={}
        state.setdefault("sent",{}); state.setdefault("btc_control",{})
        return state
    except (OSError,ValueError): return {"sent":{},"btc_control":{}}


def save_state(state: dict) -> None:
    tmp=STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8")
    tmp.replace(STATE_FILE)


def _btc_symbol(symbols: list[str]) -> str:
    # Kraken Futures uses XBT as the BTC ticker. Prefer the liquid linear perpetual.
    for candidate in ("PF_XBTUSD","PI_XBTUSD","PF_BTCUSD","PI_BTCUSD"):
        if candidate in symbols: return candidate
    matches=[s for s in symbols if ("XBT" in s or "BTC" in s) and s.startswith(("PF_","PI_"))]
    if matches: return sorted(matches,key=lambda s:(not s.startswith("PF_"),s))[0]
    raise RuntimeError("قرارداد دائمی بیت‌کوین (XBT/BTC) در فهرست Kraken Futures پیدا نشد.")


def _rsi(close, period: int=14) -> float | None:
    """TradingView-style RSI using Wilder RMA seeded by the first simple average."""
    import numpy as np
    values=np.asarray(close,dtype=float)
    if len(values)<period+1: return None
    delta=np.diff(values)
    gains=np.maximum(delta,0.0); losses=np.maximum(-delta,0.0)
    avg_gain=float(np.mean(gains[:period])); avg_loss=float(np.mean(losses[:period]))
    for i in range(period,len(gains)):
        avg_gain=(avg_gain*(period-1)+float(gains[i]))/period
        avg_loss=(avg_loss*(period-1)+float(losses[i]))/period
    if avg_loss==0: return 100.0 if avg_gain>0 else 50.0
    rs=avg_gain/avg_loss
    return 100.0-(100.0/(1.0+rs))


def _btc_signal(df):
    """Two consecutive dark SQZMOM bars plus same-direction OB/FVG touch in last 4 bars."""
    if len(df)<MIN_CANDLES: return None
    mom=sqzmom_lb(df); last=len(df)-1
    c1=str(mom.iloc[last].sqzmom_color); c0=str(mom.iloc[last-1].sqzmom_color)
    if c0==c1==DARK_SELL: side="SELL"; bullish=False
    elif c0==c1==DARK_BUY: side="BUY"; bullish=True
    else: return None
    zones=build_zones(df)
    recent=df.iloc[max(0,len(df)-4):]
    matched=[z for z in zones if z.active and z.bullish==bullish]
    touched=[]
    for _,candle in recent.iterrows():
        for z in matched:
            if candle_touches_zone(candle,z): touched.append(z)
    if not touched: return None
    unique={(z.kind,z.created_index,z.top,z.bottom):z for z in touched}
    return {"side":side,"sqzmom_color":c1,"previous_sqzmom_color":c0,
            "sqzmom_value":float(mom.iloc[last].sqzmom),"candle_time":int(df.iloc[last].time_ms),
            "close":float(df.iloc[last].close),"rsi":_rsi(df.close),
            "zones_confirmed":sorted({z.kind for z in unique.values()}),"zone_count":len(unique)}


def _btc_color(df) -> tuple[str,int,float|None]:
    mom=sqzmom_lb(df); last=len(df)-1
    return str(mom.iloc[last].sqzmom_color),int(df.iloc[last].time_ms),_rsi(df.close)


def _send_once(sent: dict, key: str, message: str) -> bool:
    if key in sent: return False
    send_telegram(message)
    sent[key]=datetime.now(timezone.utc).isoformat()
    return True


def main() -> None:
    _load_local_env()
    state=load_state(); sent=state.setdefault("sent",{}); control=state.setdefault("btc_control",{})
    symbols=get_perpetual_symbols(); btc=_btc_symbol(symbols)
    print(f"Scanner 3 | Kraken Futures | 30m | BTC leader={btc} | contracts={len(symbols)}")
    btc_df=fetch_ohlc(btc)
    if btc_df.empty or len(btc_df)<MIN_CANDLES: raise RuntimeError("دادهٔ کافی برای بیت‌کوین دریافت نشد.")
    btc_color,btc_time,btc_rsi=_btc_color(btc_df)
    last_processed=int(control.get("last_btc_candle_time",0) or 0)
    active_side=control.get("active_side")
    remaining=int(control.get("remaining_bars",0) or 0)
    new_btc_candle=btc_time>last_processed
    stop_color=STOP_SELL if active_side=="SELL" else STOP_BUY if active_side=="BUY" else None
    stopped=False

    # First: update BTC command state only once per newly closed 30m candle.
    should_scan=False
    started_this_run=False
    if new_btc_candle:
        if active_side and btc_color==stop_color:
            msg=format_stop_alert(btc,active_side,btc_color,btc_time,btc_rsi)
            _send_once(sent,f"BTC_STOP:{active_side}:{btc_time}",msg)
            print(msg)
            active_side=None; remaining=0; stopped=True
        elif active_side:
            # Scan this newly closed bar, then count it toward the four-bar license.
            should_scan=True
        if not active_side and not stopped:
            candidate=_btc_signal(btc_df)
            if candidate and candidate["candle_time"]==btc_time:
                active_side=candidate["side"]; remaining=LICENSE_BARS
                started_this_run=True
                message=format_btc_alert(btc,candidate)
                _send_once(sent,f"BTC_SIGNAL:{active_side}:{btc_time}",message)
                print(message)
        control["last_btc_candle_time"]=btc_time

    # The four-bar license begins with the first completed BTC candle AFTER the signal candle.
    if should_scan and active_side:
        print(f"BTC license active: {active_side} | bars available including this close: {remaining}")
        checked=alerts=errors=0
        for symbol in symbols:
            if symbol==btc: continue
            try:
                df=fetch_ohlc(symbol)
                if df.empty or len(df)<MIN_CANDLES: continue
                checked+=1
                signal=evaluate_symbol(df)
                if signal is None or signal["side"]!=active_side: continue
                signal["rsi"]=_rsi(df.close)
                signal["btc_rsi"]=btc_rsi
                key=f"{symbol}:{signal['side']}:{signal['candle_time']}"
                if _send_once(sent,key,format_alert(symbol,signal)):
                    alerts+=1; print(format_alert(symbol,signal))
            except Exception as exc:
                errors+=1; print(f"ERROR {symbol}: {type(exc).__name__}: {exc}")
            time.sleep(float(os.getenv("REQUEST_PAUSE_SECONDS",REQUEST_PAUSE_SECONDS)))
        remaining=max(0,remaining-1)
        if remaining==0:
            msg=format_stop_alert(btc,active_side,"WINDOW_EXPIRED",btc_time,btc_rsi)
            _send_once(sent,f"BTC_WINDOW_END:{active_side}:{btc_time}",msg)
            print(msg)
            active_side=None
        print(f"Alt scan finished | checked={checked} alerts={alerts} errors={errors}")
    elif active_side:
        print(f"BTC license waiting for next closed 30m candle | {active_side} | remaining={remaining}")
    else:
        print(f"BTC has no active command | latest SQZMOM={btc_color} | RSI={btc_rsi if btc_rsi is not None else 'N/A'}")

    control["active_side"]=active_side
    control["remaining_bars"]=remaining if active_side else 0
    control["last_btc_color"]=btc_color
    control["last_btc_rsi"]=btc_rsi
    cutoff=int(time.time()*1000)-STATE_RETENTION_DAYS*24*60*60*1000
    state["sent"]={k:v for k,v in sent.items() if k.startswith(("BTC_SIGNAL:","BTC_STOP:","BTC_WINDOW_END:")) or (k.rsplit(":",1)[-1].isdigit() and int(k.rsplit(":",1)[-1])>=cutoff)}
    state["last_run_utc"]=datetime.now(timezone.utc).isoformat()
    save_state(state)

if __name__=="__main__": main()
