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


def _stamp(ms: int) -> str:
    return datetime.fromtimestamp(ms/1000,tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def format_btc_alert(symbol: str, signal: dict) -> str:
    side_fa="خرید" if signal["side"]=="BUY" else "فروش"
    zones=" + ".join(signal["zones_confirmed"])
    rsi="N/A" if signal.get("rsi") is None else f"{signal['rsi']:.2f}"
    return (f"🟠 فرمان بیت‌کوین | اسکنر سه\n{symbol} | {signal['side']} ({side_fa})\n"
            f"تایم‌فریم: ۳۰ دقیقه | کندل بسته‌شده: {_stamp(signal['candle_time'])}\n"
            f"رنگ دو کندل آخر SQZMOM: {signal.get('previous_sqzmom_color','N/A')} ، {signal.get('sqzmom_color','N/A')}\n"
            f"RSI بیت‌کوین (14): {rsi}\nقیمت بسته‌شدن: {signal['close']:.10g}\n"
            f"تأیید محدوده: {zones} | تعداد محدوده‌های لمس‌شده: {signal['zone_count']}\n"
            f"مجوز اسکن هم‌جهت‌ها: چهار کندل بسته‌شدهٔ بعدی")


def format_stop_alert(symbol: str, side: str, color: str, candle_time: int, rsi: float|None) -> str:
    rsi_text="N/A" if rsi is None else f"{rsi:.2f}"
    if color=="WINDOW_EXPIRED": reason="پایان پنجرهٔ چهارکندلی"
    else: reason=f"بسته‌شدن کندل SQZMOM با رنگ {color}"
    return (f"⛔ توقف اسکنر سه با فرمان بیت‌کوین\n{symbol} | جهت قبلی: {side}\n"
            f"علت: {reason}\nکندل بسته‌شده: {_stamp(candle_time)}\nRSI بیت‌کوین (14): {rsi_text}\n"
            "اسکن آلت‌کوین‌ها متوقف شد؛ منتظر سیگنال معتبر جدید بیت‌کوین.")


def format_alert(symbol: str, signal: dict) -> str:
    side_fa="خرید" if signal["side"]=="BUY" else "فروش"
    zones=" + ".join(signal["zones_confirmed"])
    rsi="N/A" if signal.get("rsi") is None else f"{signal['rsi']:.2f}"
    btc_rsi="N/A" if signal.get("btc_rsi") is None else f"{signal['btc_rsi']:.2f}"
    btc_colors=signal.get("btc_last_two_colors",["N/A","N/A"])
    change=signal.get("change_24h_pct")
    change_text="N/A" if change is None else f"{change:+.2f}%"
    return (f"🔔 اسکنر سه | Kraken Futures\n{symbol} | {signal['side']} ({side_fa})\n"
            f"تایم‌فریم: ۳۰ دقیقه | کندل بسته‌شده: {_stamp(signal['candle_time'])}\n"
            f"رنگ دو کندل آخر BTC (SQZMOM): {btc_colors[0]} ، {btc_colors[1]}\n"
            f"RSI ارز (14): {rsi} | RSI بیت‌کوین (14): {btc_rsi}\n"
            f"تغییر ۲۴ ساعته: {change_text}\n"
            f"رنگ SQZMOM ارز: {signal['sqzmom_color']} | مقدار: {signal['sqzmom_value']:.8g}\n"
            f"قیمت بسته‌شدن: {signal['close']:.10g}\n"
            f"تأیید: {zones} | تعداد محدوده‌های لمس‌شده: {signal['zone_count']}")
