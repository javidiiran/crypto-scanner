import pandas as pd
from AB import _change_24h_pct
from notifier import format_alert, format_btc_alert

# 48 completed 30m intervals = 24 hours.
df=pd.DataFrame({'close':[100.0]*49})
df.loc[48,'close']=112.0
assert abs(_change_24h_pct(df)-12.0)<1e-9

buy={'side':'BUY','zones_confirmed':['FVG'],'rsi':55.5,'btc_rsi':61.2,
     'btc_last_two_colors':['green','green'],'change_24h_pct':12.0,
     'sqzmom_color':'maroon','sqzmom_value':-1.25,'candle_time':1790000000000,
     'close':1.23,'zone_count':1}
msg=format_alert('PF_TESTUSD',buy)
for part in ('green','61.20','55.50','+12.00%'):
    assert part in msg, (part,msg)

btc={'side':'SELL','zones_confirmed':['OB'],'rsi':62.1,'sqzmom_color':'green',
     'previous_sqzmom_color':'green','candle_time':1790000000000,
     'close':65000,'zone_count':1}
assert 'green ، green' in format_btc_alert('PF_XBTUSD',btc)
print('Scanner 3 rule-format tests passed')
