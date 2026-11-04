"""AB — settings and main execution entrypoint for Scanner 3."""
from __future__ import annotations
import json, os, time
from pathlib import Path
from datetime import datetime, timezone
from ob_sqzmom_engine import evaluate_symbol
from kraken_client import get_perpetual_symbols, fetch_ohlc
from notifier import send_telegram, format_alert
from settings import REQUEST_PAUSE_SECONDS, STATE_RETENTION_DAYS, MIN_CANDLES

ROOT=Path(__file__).resolve().parent
STATE_FILE=ROOT/"state.json"


def _load_local_env() -> None:
    """Load simple KEY=VALUE entries from .env without extra dependencies."""
    env_file=ROOT/".env"
    if not env_file.exists(): return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line=line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        key,value=line.split("=",1)
        os.environ.setdefault(key.strip(),value.strip().strip('"').strip("'"))


def load_state() -> dict:
    if not STATE_FILE.exists(): return {"sent":{}}
    try:
        state=json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if not isinstance(state,dict): return {"sent":{}}
        state.setdefault("sent",{})
        return state
    except (OSError,ValueError): return {"sent":{}}


def save_state(state: dict) -> None:
    tmp=STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8")
    tmp.replace(STATE_FILE)


def main() -> None:
    _load_local_env()
    state=load_state(); sent=state.setdefault("sent",{})
    symbols=get_perpetual_symbols()
    print(f"Scanner 3 | Kraken Futures | 15m | tradable perpetual contracts: {len(symbols)}")
    alerts=0; errors=0; checked=0
    for index,symbol in enumerate(symbols,1):
        try:
            df=fetch_ohlc(symbol)
            if df.empty or len(df)<MIN_CANDLES: continue
            checked += 1
            signal=evaluate_symbol(df)
            if signal is None: continue
            key=f"{symbol}:{signal['side']}:{signal['candle_time']}"
            if key in sent: continue
            message=format_alert(symbol,signal)
            # Mark sent only after Telegram succeeds; local no-token mode still records once.
            send_telegram(message)
            sent[key]=datetime.now(timezone.utc).isoformat()
            alerts += 1
            print(message)
        except Exception as exc:
            errors += 1; print(f"ERROR {symbol}: {type(exc).__name__}: {exc}")
        time.sleep(float(os.getenv("REQUEST_PAUSE_SECONDS",REQUEST_PAUSE_SECONDS)))
    cutoff=int(time.time()*1000)-STATE_RETENTION_DAYS*24*60*60*1000
    state["sent"]={k:v for k,v in sent.items() if k.rsplit(":",1)[-1].isdigit() and int(k.rsplit(":",1)[-1])>=cutoff}
    state["last_run_utc"]=datetime.now(timezone.utc).isoformat()
    state["last_symbols"]=len(symbols); state["last_checked"]=checked
    state["last_alerts"]=alerts; state["last_errors"]=errors
    save_state(state)
    print(f"Finished | checked={checked} alerts={alerts} errors={errors}")

if __name__ == "__main__": main()
