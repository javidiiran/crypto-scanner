"""Kraken Futures Scanner 2 — CNV + RSI, 1h closed candles."""
from __future__ import annotations
import json, math, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from ob_sqzmom_engine import analyze_market

REST_BASE='https://futures.kraken.com/derivatives/api/v3'; CHART_BASE='https://futures.kraken.com/api/charts/v1'
TIMEFRAME='1h'; INTERVAL_MS=60*60*1000; CANDLE_LIMIT=1000; MIN_CANDLES=40
MAX_CANDLE_STALENESS_MS=2*INTERVAL_MS; MAX_WORKERS=5; STATE_FILE=Path('state.json')
PERPETUAL_PREFIXES=('PF_','PI_'); BTC_SYMBOLS=('PF_XBTUSD','PI_XBTUSD')
SESSION=requests.Session(); SESSION.headers.update({'User-Agent':'Kraken-Futures-Scanner-2-CNV-RSI/2.0.0','Accept':'application/json','Cache-Control':'no-cache','Pragma':'no-cache'})
retry=Retry(total=4,connect=4,read=3,status=3,backoff_factor=.7,status_forcelist=(408,429,500,502,503,504),allowed_methods=frozenset(['GET']),respect_retry_after_header=True)
SESSION.mount('https://',HTTPAdapter(max_retries=retry,pool_connections=MAX_WORKERS,pool_maxsize=MAX_WORKERS))

def get_json(url,params=None):
 r=SESSION.get(url,params=params,timeout=(10,30)); r.raise_for_status(); p=r.json()
 if isinstance(p,dict) and (p.get('result')=='error' or p.get('error')): raise RuntimeError(f'Kraken API error: {p}')
 return p

def telegram_send(message):
 token=os.getenv('TELEGRAM_BOT_TOKEN','').strip(); chat=os.getenv('TELEGRAM_CHAT_ID','').strip()
 if not token or not chat: raise RuntimeError('Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in GitHub Secrets')
 r=SESSION.post(f'https://api.telegram.org/bot{token}/sendMessage',data={'chat_id':chat,'text':message,'disable_web_page_preview':True},timeout=(10,25)); r.raise_for_status()
 if not r.json().get('ok'): raise RuntimeError(f'Telegram rejected message: {r.text}')

def load_state():
 default={'sent':[],'btc_direction':None,'btc_signal_timestamp_ms':0,'btc_signal_time':'—','btc_signal_rsi':None}
 try:
  if STATE_FILE.exists(): default.update(json.loads(STATE_FILE.read_text(encoding='utf-8')))
 except Exception as e: print(f'WARNING state reset: {e}')
 default['sent']=set(default.get('sent',[])); return default

def save_state(s):
 p=dict(s); p['sent']=sorted(s['sent'])[-20000:]; p['scanner']='Kraken Futures Scanner 2 CNV RSI 2.0.0'
 tmp=STATE_FILE.with_suffix('.tmp'); tmp.write_text(json.dumps(p,ensure_ascii=False,indent=2),encoding='utf-8'); tmp.replace(STATE_FILE)

def fetch_symbols():
 payload=get_json(f'{REST_BASE}/instruments'); instruments=payload.get('instruments',[]) if isinstance(payload,dict) else []
 symbols=set()
 for x in instruments:
  sym=str(x.get('symbol') or x.get('tradeable') or '').upper()
  if not sym.startswith(PERPETUAL_PREFIXES) or x.get('tradeable') is False or x.get('isExpired') is True: continue
  status=str(x.get('status','')).lower()
  if status and status not in {'online','trading','active'}: continue
  symbols.add(sym)
 if not symbols: raise RuntimeError('No tradable Kraken perpetual contracts found')
 return sorted(symbols)

def fetch_closed_candles(symbol,now_ms):
 url=f'{CHART_BASE}/trade/{quote(symbol,safe="_")}/{TIMEFRAME}'; end=now_ms//1000; start=end-21*86400
 modes=[('range',{'from':start,'to':end,'count':CANDLE_LIMIT}),('latest',{'count':CANDLE_LIMIT}),('default_latest',None)]
 last=None
 for attempt in range(1,4):
  for mode,params in modes:
   payload=get_json(url,params); rows=payload.get('candles',[]) if isinstance(payload,dict) else []
   normalized=[]
   for row in rows if isinstance(rows,list) else []:
    try:
     stamp=int(row['time'] if isinstance(row,dict) else row[0]); stamp=stamp*1000 if stamp<10**12 else stamp
     if stamp+INTERVAL_MS>now_ms: continue
     if isinstance(row,dict): op,hi,lo,cl,vol=row['open'],row['high'],row['low'],row['close'],row.get('volume',0)
     else: op,hi,lo,cl,vol=row[1:6]
     normalized.append((stamp,float(op),float(hi),float(lo),float(cl),float(vol)))
    except (KeyError,TypeError,ValueError,IndexError): continue
   if not normalized: continue
   d=pd.DataFrame(normalized,columns=['timestamp','open','high','low','close','volume']).drop_duplicates('timestamp').sort_values('timestamp')
   last=int(d.timestamp.iloc[-1]); stale=now_ms-(last+INTERVAL_MS)
   if stale<=MAX_CANDLE_STALENESS_MS:
    d.timestamp=pd.to_datetime(d.timestamp,unit='ms',utc=True); d=d.set_index('timestamp')
    if len(d)<MIN_CANDLES: return pd.DataFrame()
    print(f'DATA_OK {symbol}: {mode}, candles={len(d)}, last_closed={d.index[-1]}')
    return d
   print(f'DATA_STALE {symbol}: {mode}, stale_min={stale/60000:.1f}')
  if attempt<3: time.sleep(attempt)
 raise RuntimeError(f'No fresh closed Kraken {TIMEFRAME} candles for {symbol}; last={last}')

def evaluate_symbol(symbol,now):
 d=fetch_closed_candles(symbol,now)
 if d.empty: return {'symbol':symbol,'status':'INSUFFICIENT_DATA'}
 return {'symbol':symbol,'status':'OK','analysis':analyze_market(d)}

def current_signals(symbol,analysis):
 last=len(analysis['candles'])-1; out=[]
 # Sequence is confirmed on the third candle; signal remains actionable only at that closed candle.
 for s in analysis['signals']:
  if last-int(s['signal_index'])==0:
   item=dict(s); item['symbol']=symbol; item['current_rsi']=analysis['latest_rsi']; item['cnv_current_color']=analysis['latest_cnv_color']; out.append(item)
 return out

def fmt(x): return f'{x:.1f}' if x is not None and math.isfinite(float(x)) else 'N/A'

def alert(item,bctx):
 buy=item['direction']=='BUY'; title='🟢 BUY / LONG' if buy else '🔴 SELL / SHORT'
 return '\n'.join([title,f"قرارداد: {item['symbol']} — Kraken Futures Perpetual",f'تایم‌فریم: {TIMEFRAME}',
  f"قیمت کندل سوم الگو: {item['price']:.10g}",f"الگوی CNV: {'صورتی ← آبی ← آبی' if buy else 'آبی ← صورتی ← صورتی'}",
  f"RSI(14) سه کندل: {' / '.join(fmt(x) for x in item['rsi_values'])}",
  f"RSI شرط سیگنال: {fmt(item['rsi_condition_value'])} {'(< 50)' if buy else '(> 50)' }",
  f"RSI فعلی ارز: {fmt(item['current_rsi'])}",f"جهت آخرین سیگنال BTC: {bctx['direction']}",
  f"زمان آخرین سیگنال BTC: {bctx['time']}",f"RSI بیت‌کوین هنگام سیگنال: {fmt(bctx['rsi'])}",
  f"دو رنگ آخر CNV بیت‌کوین (جدیدترین / قبلی): {bctx['cnv_colors']}",f"RSI فعلی بیت‌کوین: {fmt(bctx['current_rsi'])}"])

def main():
 now=int(time.time()*1000); print(f'Start Kraken Futures Scanner 2 | CNV + RSI(14) | {TIMEFRAME}')
 symbols=fetch_symbols(); btc_symbol=next((x for x in BTC_SYMBOLS if x in symbols),None)
 if not btc_symbol: raise RuntimeError('BTC perpetual symbol missing from Kraken instruments')
 # BTC is always fetched and evaluated FIRST, before any other futures contract.
 btc_data=fetch_closed_candles(btc_symbol,now)
 if btc_data.empty: raise RuntimeError('BTC closed candles unavailable; altcoin scan aborted')
 btc_analysis=analyze_market(btc_data); state=load_state(); btc_signals=current_signals(btc_symbol,btc_analysis)
 btc_colors=btc_analysis['cnv']['blue'].iloc[-2:].tolist()
 btc_color_text=' / '.join('آبی' if x else 'صورتی' for x in reversed(btc_colors))
 if btc_signals:
  latest=btc_signals[-1]; state.update({'btc_direction':latest['direction'],'btc_signal_timestamp_ms':latest['timestamp_ms'],'btc_signal_time':latest['signal_time'],'btc_signal_rsi':latest['rsi']})
  btc_item=dict(latest,symbol=btc_symbol,current_rsi=btc_analysis['latest_rsi'])
  key=f"{btc_symbol}|{latest['timestamp_ms']}|{latest['direction']}"
  if key not in state['sent']:
   telegram_send(alert(btc_item,{'direction':latest['direction'],'time':latest['signal_time'],'rsi':latest['rsi'],'cnv_colors':btc_color_text,'current_rsi':btc_analysis['latest_rsi']})); state['sent'].add(key)
   print(f"BTC_ALERT_SENT {latest['direction']} {latest['signal_time']}")
 else: print('BTC_NO_SIGNAL: no matching 3-candle CNV+RSI pattern on latest closed candle')
 save_state(state)
 # BTC is checked first, but absence of a fresh BTC signal does NOT stop altcoin scanning.
 # The latest confirmed BTC signal is context only; it is not a directional filter.
 bctx={'direction':state.get('btc_direction') or 'بدون سیگنال ثبت‌شده',
       'time':state.get('btc_signal_time','—'),'rsi':state.get('btc_signal_rsi'),
       'cnv_colors':btc_color_text,'current_rsi':btc_analysis['latest_rsi']}
 alts=[s for s in symbols if s not in BTC_SYMBOLS]; records=[]; errors=0
 with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
  futures={pool.submit(evaluate_symbol,s,now):s for s in alts}
  for f in as_completed(futures):
   sym=futures[f]
   try: records.append(f.result())
   except Exception as e: errors+=1; print(f'SKIP {sym}: {type(e).__name__}: {e}')
 alerts=[]
 for rec in records:
  if rec.get('status')!='OK': continue
  for item in current_signals(rec['symbol'],rec['analysis']):
   # Altcoin signals are reported independently; BTC direction is informational only.
   alerts.append(item)
 alerts.sort(key=lambda x:(x['timestamp_ms'],x['symbol']))
 sent=0
 for item in alerts:
  key=f"{item['symbol']}|{item['timestamp_ms']}|{item['direction']}"
  if key in state['sent']: continue
  try:
   telegram_send(alert(item,bctx)); state['sent'].add(key); save_state(state); sent+=1
   print(f"ALERT {item['symbol']} {item['direction']} RSI={fmt(item['rsi'])} {item['signal_time']}")
  except Exception as e: print(f"TELEGRAM_ERROR {item['symbol']}: {e}")
 save_state(state); print(f'Finished: perpetuals={len(symbols)} altcoins={len(alts)} signals_sent={sent} errors={errors}')

if __name__=='__main__':
 try: main()
 except Exception as exc: print(f'FATAL: {type(exc).__name__}: {exc}',file=sys.stderr); raise
