# اسکنر فیوچرز بایننس — SQZMOM + SMC (Reversal اطلاعات جانبی)

## تنظیمات ثابت
- Binance USDⓈ-M USDT-M Perpetual Futures
- تایم‌فریم ۳۰ دقیقه؛ تمام قراردادهای فعال USDT
- فقط کندل‌های بسته‌شده
- اجرای GitHub Actions هر ۵ دقیقه
- فقط هشدار تلگرام؛ بدون معامله خودکار

## منطق نهایی سیگنال
1. **SQZMOM LazyBear تریگر اصلی است.** سه میله بسته‌شده آخر باید همگی بالای صفر با قله در میله وسط (SELL)، یا همگی زیر صفر با دره در میله وسط (BUY) باشند.
2. **SMC تنها تأییدکننده است.** کافی است محدوده High/Low یکی از پنج کندل بسته‌شده آخر با یک ناحیه هم‌جهت SMC هم‌پوشانی داشته باشد. Close لازم نیست داخل باکس باشد.
3. **Reversal Probability هیچ نقشی در قبول یا رد سیگنال ندارد.** رنگ باکس و محدوده آن برای خود نماد فقط به‌صورت اطلاعات جانبی در تلگرام گزارش می‌شود: سبز، قرمز یا بدون باکس/خنثی.
4. رنگ آخرین باکس Reversal در BTCUSDT نیز در تمام هشدارها گزارش می‌شود و صرفاً اطلاعات زمینه‌ای است؛ جهت سیگنال آلت‌کوین را تغییر نمی‌دهد.

## تنظیمات اندیکاتورها
- SQZMOM: BB Length=20، BB Mult=2، KC Length=20، KC Mult=1.5، True Range روشن؛ رفتار محاسباتی سورس LazyBear ارسالی حفظ شده است.
- SMC: Internal pivot=5، Swing pivot=50، ATR filter، حداکثر ۵ OB فعال برای هر سمت/نوع؛ نواحی Internal/Swing OB و Premium/Discount بررسی می‌شوند. Equilibrium خنثی است و تأیید جهت‌دار نمی‌دهد.
- SMC تماس سایه (wick touch) را در پنج کندل بسته‌شده آخر می‌سنجد.
- Reversal: Swing Length=20، Maximum Reversals=1000، Normalize=false، Percentile=90؛ فقط برای گزارش رنگ/محدوده محاسبه می‌شود.

## نصب و اجرا
1. همه فایل‌های بسته را در ریشه مخزن GitHub جایگزین کنید؛ پوشه `.github/workflows/scan.yml` باید حفظ شود.
2. در Settings → Secrets and variables → Actions دو Secret بسازید: `TELEGRAM_BOT_TOKEN` و `TELEGRAM_CHAT_ID`.
3. در Settings → Actions → General، دسترسی Workflow به Read and write فعال باشد تا `state.json` ثبت شود.
4. از Actions، workflow را دستی اجرا کنید.

## محدودیت مهم
اگر GitHub Actions برای `https://fapi.binance.com` خطای HTTP 451 بدهد، اجرای اسکنر به‌دلیل محدودیت دسترسی Futures API از محیط Runner متوقف می‌شود. این نسخه به Spot یا بازار دیگری تغییر مسیر نمی‌دهد و محدودیت منطقه‌ای را دور نمی‌زند. ساخت و تست آفلاین فایل‌ها به معنی تأیید اتصال زنده Binance یا ارسال واقعی تلگرام نیست.
