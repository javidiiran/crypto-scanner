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


def _btc_signal(df, with_diagnostics: bool=False):
    """Two consecutive dark SQZMOM bars plus same-direction OB/FVG touch in last 4 bars."""
    import pandas as pd
    diag={"reason":"unknown","bars":[],"active_bearish":0,"active_bullish":0,
          "bearish_touches":[],"bullish_touches":[],"bearish_zones":[],"bullish_zones":[]}
    if len(df)<MIN_CANDLES:
        diag["reason"]=f"insufficient_candles:{len(df)}<{MIN_CANDLES}"
        return (None,diag) if with_diagnostics else None

    mom=sqzmom_lb(df); last=len(df)-1
    for i in range(max(0,last-5),last+1):
        t=pd.to_datetime(int(df.iloc[i].time_ms),unit="ms",utc=True).isoformat()
        diag["bars"].append({"time":t,"color":str(mom.iloc[i].sqzmom_color),
                             "value":None if pd.isna(mom.iloc[i].sqzmom) else float(mom.iloc[i].sqzmom),
                             "open":float(df.iloc[i].open),"high":float(df.iloc[i].high),
                             "low":float(df.iloc[i].low),"close":float(df.iloc[i].close)})

    # Build and report zones independently of the SQZMOM color test, so a rejected
    # color pair still exposes whether Kraken's OB/FVG port detected the chart touch.
    zones=build_zones(df)
    active_bearish=[z for z in zones if z.active and not z.bullish]
    active_bullish=[z for z in zones if z.active and z.bullish]
    diag["active_bearish"]=len(active_bearish); diag["active_bullish"]=len(active_bullish)
    def zone_info(z):
        return {"kind":z.kind,"created_index":z.created_index,"top":z.top,"bottom":z.bottom}
    diag["bearish_zones"]=[zone_info(z) for z in active_bearish[-12:]]
    diag["bullish_zones"]=[zone_info(z) for z in active_bullish[-12:]]
    touched_by_side={False:[],True:[]}
    recent_start=max(0,len(df)-4)
    for i in range(recent_start,len(df)):
        candle=df.iloc[i]
        for direction in (False,True):
            hits=[z for z in zones if z.active and z.bullish==direction and candle_touches_zone(candle,z)]
            if hits:
                key="bullish_touches" if direction else "bearish_touches"
                diag[key].append({"time":pd.to_datetime(int(candle.time_ms),unit="ms",utc=True).isoformat(),
                                  "zones":[zone_info(z) for z in hits]})
                touched_by_side[direction].extend(hits)

    c1=str(mom.iloc[last].sqzmom_color); c0=str(mom.iloc[last-1].sqzmom_color)
    if c0==c1==DARK_SELL:
        side="SELL"; bullish=False
    elif c0==c1==DARK_BUY:
        side="BUY"; bullish=True
    else:
        diag["reason"]=f"color_pair_mismatch:previous={c0},latest={c1},required=green/green_or_maroon/maroon"
        return (None,diag) if with_diagnostics else None

    touched=touched_by_side[bullish]
    if not touched:
        diag["reason"]=f"no_active_{'bullish' if bullish else 'bearish'}_OB_FVG_touch_in_last_4_closed_candles"
        return (None,diag) if with_diagnostics else None
    unique={(z.kind,z.created_index,z.top,z.bottom):z for z in touched}
    diag["reason"]="signal_confirmed"
    result={"side":side,"sqzmom_color":c1,"previous_sqzmom_color":c0,
            "btc_last_two_colors":[c0,c1],
            "sqzmom_value":float(mom.iloc[last].sqzmom),"candle_time":int(df.iloc[last].time_ms),
            "close":float(df.iloc[last].close),"rsi":_rsi(df.close),
            "zones_confirmed":sorted({z.kind for z in unique.values()}),"zone_count":len(unique)}
    return (result,diag) if with_diagnostics else result



def _btc_color(df) -> tuple[str,int,float|None]:
    mom=sqzmom_lb(df); last=len(df)-1
    return str(mom.iloc[last].sqzmom_color),int(df.iloc[last].time_ms),_rsi(df.close)


def _last_two_btc_colors(df) -> list[str]:
    mom=sqzmom_lb(df)
    if len(mom)<2: return ["N/A","N/A"]
    return [str(mom.iloc[-2].sqzmom_color),str(mom.iloc[-1].sqzmom_color)]


def _change_24h_pct(df) -> float | None:
    """24h percentage change from the close 48 completed 30m candles ago."""
    if len(df)<49: return None
    old=float(df.iloc[-49].close)
    latest=float(df.iloc[-1].close)
    if old<=0: return None
    return (latest/old-1.0)*100.0


def _send_once(sent: dict, key: str, message: str) -> bool:
    if key in sent: return False
    send_telegram(message)
    sent[key]=datetime.now(timezone.utc).isoformat()
    return True


def _scan_alts(symbols: list[str], btc: str, side: str, btc_time: int,
               btc_rsi: float | None, btc_colors: list[str], sent: dict) -> tuple[int,int,int]:
    checked=alerts=errors=0
    pending_signals=[]
    for symbol in symbols:
        if symbol==btc: continue
        try:
            df=fetch_ohlc(symbol)
            if df.empty or len(df)<MIN_CANDLES: continue
            checked+=1
            # Align every altcoin check to the exact BTC 30m close being processed.
            signal=evaluate_symbol(df,as_of_time_ms=btc_time)
            if signal is None or signal["side"]!=side or signal["candle_time"]!=btc_time: continue
            signal["rsi"]=_rsi(df[df.time_ms<=btc_time].close)
            signal["btc_rsi"]=btc_rsi
            signal["btc_last_two_colors"]=btc_colors
            signal["change_24h_pct"]=_change_24h_pct(df[df.time_ms<=btc_time].reset_index(drop=True))
            signal["symbol"]=symbol
            pending_signals.append(signal)
        except Exception as exc:
            errors+=1
            print(f"ERROR {symbol}: {type(exc).__name__}: {exc}")
        time.sleep(float(os.getenv("REQUEST_PAUSE_SECONDS",REQUEST_PAUSE_SECONDS)))

    # Direction-specific ranking: BUY = strongest positive 24h change first;
    # SELL = greatest negative 24h change first. Missing changes go last.
    if side=="BUY":
        pending_signals.sort(key=lambda x:(x.get("change_24h_pct") is None,
                                            -(x.get("change_24h_pct") if x.get("change_24h_pct") is not None else 0.0)))
    else:
        pending_signals.sort(key=lambda x:(x.get("change_24h_pct") is None,
                                            x.get("change_24h_pct") if x.get("change_24h_pct") is not None else float("inf")))
    for signal in pending_signals:
        symbol=signal["symbol"]
        key=f"{symbol}:{signal['side']}:{signal['candle_time']}"
        message=format_alert(symbol,signal)
        if _send_once(sent,key,message):
            alerts+=1
            print(message)
    print(f"Signal ranking: {side} | {'highest 24h gain first' if side=='BUY' else 'largest 24h decline first'}")
    return checked,alerts,errors


def main() -> None:
    _load_local_env()
    state=load_state(); sent=state.setdefault("sent",{}); control=state.setdefault("btc_control",{})
    symbols=get_perpetual_symbols(); btc=_btc_symbol(symbols)
    print(f"Scanner 3 | Kraken Futures | 30m | BTC leader={btc} | contracts={len(symbols)}")
    btc_df=fetch_ohlc(btc)
    if btc_df.empty or len(btc_df)<MIN_CANDLES: raise RuntimeError("دادهٔ کافی برای بیت‌کوین دریافت نشد.")

    last_processed=int(control.get("last_btc_candle_time",0) or 0)
    active_side=control.get("active_side")
    remaining=int(control.get("remaining_bars",0) or 0)
    if last_processed:
        new_indices=[i for i,t in enumerate(btc_df.time_ms) if int(t)>last_processed]
    else:
        # On first installation, start from the latest completed candle only.
        new_indices=[len(btc_df)-1]

    if not new_indices:
        latest=btc_df
        colors=_last_two_btc_colors(latest)
        rsi=_rsi(latest.close)
        if active_side:
            print(f"BTC license waiting for next closed 30m candle | {active_side} | remaining={remaining}")
        else:
            _,diag=_btc_signal(latest,with_diagnostics=True)
            print(f"BTC has no active command | last two SQZMOM={colors[0]},{colors[1]} | RSI={rsi if rsi is not None else 'N/A'}")
            print(f"BTC diagnostic reason: {diag['reason']}")
            print(f"BTC active zones: bearish={diag['active_bearish']} bullish={diag['active_bullish']}")
            print(f"BTC bearish touches in last 4: {len(diag['bearish_touches'])} | bullish touches in last 4: {len(diag['bullish_touches'])}")
            for bar in diag['bars'][-6:]:
                print(f"BTC SQZMOM bar | {bar['time']} | color={bar['color']} | value={bar['value']} | O={bar['open']} H={bar['high']} L={bar['low']} C={bar['close']}")
            for touch in diag['bearish_touches']:
                print(f"BTC bearish zone touch | {touch['time']} | {touch['zones']}")
            for touch in diag['bullish_touches']:
                print(f"BTC bullish zone touch | {touch['time']} | {touch['zones']}")

    total_checked=total_alerts=total_errors=0
    for idx in new_indices:
        btc_slice=btc_df.iloc[:idx+1].reset_index(drop=True)
        btc_color,btc_time,btc_rsi=_btc_color(btc_slice)
        btc_colors=_last_two_btc_colors(btc_slice)
        stop_color=STOP_SELL if active_side=="SELL" else STOP_BUY if active_side=="BUY" else None
        print(f"BTC closed candle | time={btc_time} | colors={btc_colors[0]},{btc_colors[1]} | RSI={btc_rsi if btc_rsi is not None else 'N/A'}")

        if active_side:
            # Stop on the very first closed bright-color bar against the active command.
            if btc_color==stop_color:
                msg=format_stop_alert(btc,active_side,btc_color,btc_time,btc_rsi)
                _send_once(sent,f"BTC_STOP:{active_side}:{btc_time}",msg)
                print(msg)
                active_side=None; remaining=0
            else:
                checked,alerts,errors=_scan_alts(symbols,btc,active_side,btc_time,btc_rsi,btc_colors,sent)
                total_checked+=checked; total_alerts+=alerts; total_errors+=errors
                remaining=max(0,remaining-1)
                if remaining==0:
                    msg=format_stop_alert(btc,active_side,"WINDOW_EXPIRED",btc_time,btc_rsi)
                    _send_once(sent,f"BTC_WINDOW_END:{active_side}:{btc_time}",msg)
                    print(msg)
                    active_side=None
        else:
            candidate,diag=_btc_signal(btc_slice,with_diagnostics=True)
            if candidate and candidate["candle_time"]==btc_time:
                active_side=candidate["side"]
                remaining=LICENSE_BARS
                message=format_btc_alert(btc,candidate)
                _send_once(sent,f"BTC_SIGNAL:{active_side}:{btc_time}",message)
                print(message)
            else:
                print(f"BTC has no active command | reason={diag['reason']} | active bearish={diag['active_bearish']} bullish={diag['active_bullish']} | bearish touches={len(diag['bearish_touches'])} bullish touches={len(diag['bullish_touches'])}")

        last_processed=btc_time
        control["last_btc_candle_time"]=last_processed
        control["active_side"]=active_side
        control["remaining_bars"]=remaining if active_side else 0
        control["last_btc_color"]=btc_color
        control["last_btc_rsi"]=btc_rsi
        # Persist after each processed BTC close so an interruption cannot replay a bar.
        state["last_run_utc"]=datetime.now(timezone.utc).isoformat()
        save_state(state)

    cutoff=int(time.time()*1000)-STATE_RETENTION_DAYS*24*60*60*1000
    state["sent"]={k:v for k,v in sent.items() if k.startswith(("BTC_SIGNAL:","BTC_STOP:","BTC_WINDOW_END:")) or (k.rsplit(":",1)[-1].isdigit() and int(k.rsplit(":",1)[-1])>=cutoff)}
    state["last_run_utc"]=datetime.now(timezone.utc).isoformat()
    save_state(state)
    print(f"Alt scan totals | checked={total_checked} alerts={total_alerts} errors={total_errors}")

if __name__=="__main__": main()
