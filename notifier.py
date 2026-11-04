from __future__ import annotations
import os, requests
from datetime import datetime, timezone


def send_telegram(message: str) -> bool:
    token=os.getenv("TELEGRAM_BOT_TOKEN","").strip(); chat=os.getenv("TELEGRAM_CHAT_ID","").strip()
    if not token or not chat:
        print("Telegram not configured; alert printed locally:\n"+message)
        return False
    url=f"https://api.telegram.org/bot{token}/sendMessage"
    r=requests.post(url,json={"chat_id":chat,"text":message,"disable_web_page_preview":True},timeout=20)
    r.raise_for_status(); data=r.json()
    if not data.get("ok"): raise RuntimeError(f"Telegram API error: {data}")
    return True


def format_alert(symbol: str, signal: dict) -> str:
    stamp=datetime.fromtimestamp(signal["candle_time"]/1000,tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    side_fa="خرید" if signal["side"]=="BUY" else "فروش"
    zones=" + ".join(signal["zones_confirmed"])
    return (f"🔔 اسکنر سه | Kraken Futures\n{symbol} | {signal['side']} ({side_fa})\n"
            f"تایم‌فریم: 15 دقیقه | کندل بسته‌شده: {stamp}\n"
            f"رنگ SQZMOM: {signal['sqzmom_color']} | مقدار: {signal['sqzmom_value']:.8g}\n"
            f"قیمت بسته‌شدن: {signal['close']:.10g}\n"
            f"تأیید: {zones} | تعداد محدوده‌های لمس‌شده: {signal['zone_count']}")
