"""24/7 Binance Spot USDT scanner. Signals only on closed 30m candles."""
from __future__ import annotations
import json, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import pandas as pd
import requests
from ob_sqzmom_engine import evaluate_signals

TIMEFRAME="30m"
INTERVAL_MS=30*60*1000
CANDLE_LIMIT=300
STATE_FILE=Path("state.json")
BINANCE_BASE="https://data-api.binance.vision"
MAX_WORKERS=8
SESSION=requests.Session()
SESSION.headers.update({"User-Agent":"Mozilla/5.0 (compatible; BinanceSpotSQZScanner/2.0)"})


def get_json(url: str, params: dict[str,Any] | None=None) -> Any:
    error=None
    for attempt in range(3):
        try:
            r=SESSION.get(url,params=params,timeout=25); r.raise_for_status(); return r.json()
        except Exception as exc:
            error=exc
            if attempt<2: time.sleep(1.0+attempt*1.5)
    raise RuntimeError(f"Request failed: {url}: {error}")


def telegram_send(message: str) -> None:
    token=os.environ.get("TELEGRAM_BOT_TOKEN","").strip()
    chat_id=os.environ.get("TELEGRAM_CHAT_ID","").strip()
    if not token or not chat_id:
        raise RuntimeError("GitHub Secrets TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required")
    r=SESSION.post(f"https://api.telegram.org/bot{token}/sendMessage",
                   data={"chat_id":chat_id,"text":message,"disable_web_page_preview":True},timeout=25)
    r.raise_for_status()


def load_state() -> set[str]:
    if not STATE_FILE.exists(): return set()
    try: return set(json.loads(STATE_FILE.read_text(encoding="utf-8")).get("sent",[]))
    except Exception as exc:
        print(f"WARNING: state.json unreadable; starting clean: {exc}"); return set()


def save_state(sent: set[str]) -> None:
    STATE_FILE.write_text(json.dumps({"sent":sorted(sent)[-15000:]},ensure_ascii=False,indent=2),encoding="utf-8")


def fetch_symbols() -> list[str]:
    payload=get_json(f"{BINANCE_BASE}/api/v3/exchangeInfo")
    excluded={"USDT","USDC","BUSD","TUSD","FDUSD","USDP","DAI","EUR","TRY","BRL","GBP","AUD","BIDR","RUB","UAH","NGN","ZAR","PLN","RON","ARS","JPY"}
    symbols=[]
    for x in payload.get("symbols",[]):
        if x.get("status")=="TRADING" and x.get("quoteAsset")=="USDT" and x.get("isSpotTradingAllowed",True):
            base=str(x.get("baseAsset","")).upper()
            if base and base not in excluded and x.get("symbol"): symbols.append(x["symbol"])
    return sorted(set(symbols))


def fetch_closed_ohlcv(symbol: str) -> pd.DataFrame:
    rows=get_json(f"{BINANCE_BASE}/api/v3/klines",{"symbol":symbol,"interval":TIMEFRAME,"limit":CANDLE_LIMIT})
    now=int(time.time()*1000)
    closed=[x for x in rows if int(x[0])+INTERVAL_MS<=now]
    if len(closed)<100: return pd.DataFrame()
    d=pd.DataFrame([[int(x[0]),float(x[1]),float(x[2]),float(x[3]),float(x[4]),float(x[5])] for x in closed],
                    columns=["timestamp","open","high","low","close","volume"])
    d["timestamp"]=pd.to_datetime(d.timestamp,unit="ms",utc=True)
    return d.set_index("timestamp")


def candle_key(symbol: str, candle_time: pd.Timestamp, signal: str) -> str:
    return f"{symbol}|{TIMEFRAME}|{int(candle_time.timestamp()*1000)}|{signal}"


def mfi_grade(mfi_value: float, signal: str) -> str:
    """Grade an already-valid signal by directional MFI alignment only.

    BUY: lower MFI is stronger (A<=20, B<=35, C<=50, otherwise D).
    SELL: higher MFI is stronger (A>=80, B>=65, C>=50, otherwise D).
    This grade never suppresses a signal.
    """
    if pd.isna(mfi_value):
        return "D"
    value=float(mfi_value)
    if signal == "BUY":
        return "A" if value <= 20 else "B" if value <= 35 else "C" if value <= 50 else "D"
    return "A" if value >= 80 else "B" if value >= 65 else "C" if value >= 50 else "D"


def grade_order(grade: str) -> int:
    return {"A": 4, "B": 3, "C": 2, "D": 1}.get(grade, 0)


def evaluate_one(symbol: str):
    candles=fetch_closed_ohlcv(symbol)
    if candles.empty: return None
    result=evaluate_signals(candles)
    row=result.iloc[-1].copy()
    signal=str(row.get("signal",""))
    if signal not in ("BUY","SELL"): return None
    row["sqzmom_prev"]=float(result.sqzmom.iloc[-2])
    avg=float(candles.volume.rolling(20).mean().iloc[-1])
    row["volume_ratio"]=float(candles.volume.iloc[-1])/max(avg,1e-12)
    return symbol,signal,row


def format_price(value: float) -> str:
    return f"{value:.10g}"


def send_signal(symbol: str, signal: str, row: pd.Series, grade: str) -> None:
    emoji="🟢 BUY" if signal=="BUY" else "🔴 SELL"
    t=row.name.strftime("%Y-%m-%d %H:%M UTC")
    zone="Trough / green support" if signal=="BUY" else "Peak / red resistance"
    lines=[emoji,f"Pair: {symbol}",f"Market: Binance Spot",f"Timeframe: {TIMEFRAME}",
           f"MFI grade: {grade} (all valid signals are sent)",
           f"MFI(14): {float(row.mfi):.2f}",
           f"Close: {format_price(float(row.close))}",f"SQZMOM: {float(row.sqzmom):.8g}",
           f"Dynamic Profile: {zone}",
           f"Zone: {format_price(float(row.profile_zone_bottom))} - {format_price(float(row.profile_zone_top))}",
           f"Candle-to-zone distance: {float(row.profile_gap_pct):.3f}% ({row.profile_confirmation})",
           f"Matching candle: {int(row.profile_pattern_candle_age)} candle(s) before signal",f"Closed candle: {t}",
           "Spot alert only — no automatic order and no short selling."]
    telegram_send("\n".join(lines))


def main() -> None:
    sent=load_state()
    symbols=fetch_symbols()
    print(f"Binance Spot USDT pairs loaded: {len(symbols)}; timeframe={TIMEFRAME}")
    found=[]; checked=0; errors=0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures={pool.submit(evaluate_one,s):s for s in symbols}
        for future in as_completed(futures):
            symbol=futures[future]
            try:
                result=future.result(); checked+=1
                if result:
                    pair,signal,row=result
                    key=candle_key(pair,row.name,signal)
                    if key not in sent:
                        found.append((pair,signal,row,key,mfi_grade(float(row.mfi),signal)))
            except Exception as exc:
                errors+=1; print(f"Skipped {symbol}: {type(exc).__name__}: {exc}")
    found.sort(key=lambda x:grade_order(x[4]),reverse=True)
    sent_now=0
    for pair,signal,row,key,grade in found:
        try:
            send_signal(pair,signal,row,grade)
            sent.add(key); sent_now+=1; save_state(sent)
            print(f"ALERT {key} MFI_grade={grade} MFI={float(row.mfi):.2f}")
        except Exception as exc:
            print(f"Telegram send failed for {pair}: {exc}")
    print(f"Done: checked={checked}/{len(symbols)} signals_sent={sent_now} errors={errors}")

if __name__=="__main__": main()
