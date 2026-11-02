# اسکنر فیوچرز بایننس — SQZMOM + Reversal Probability + SMC

## تنظیمات
- صرافی: Binance
- بازار: **USDⓈ-M USDT-M Perpetual Futures**
- تایم‌فریم: ۳۰ دقیقه
- نمادها: تمام قراردادهای دائمی فعال با Quote Asset برابر USDT (بدون فیلتر رتبه یا حذف نمادهای فعال)
- داده‌ها: فقط کندل‌های کاملاً بسته‌شده
- هشدار: Telegram
- اجرا: GitHub Actions هر ۵ دقیقه
- معامله خودکار: ندارد؛ فقط پیام هشدار می‌فرستد.

## منطق سیگنال

### ۱) SQZMOM [LazyBear] — تریگر اصلی
از سه کندل بسته‌شده آخر استفاده می‌شود:
- **BUY:** هر سه میله زیر صفر باشند و میله وسط از نظر قدر مطلق، از میله اول و سوم عمیق‌تر باشد (دره).
- **SELL:** هر سه میله بالای صفر باشند و میله وسط از هر دو میله کناری بلندتر باشد (قله).
- سیگنال پس از بسته‌شدن کندل سوم بررسی می‌شود؛ کندل باز وارد محاسبه نمی‌شود.

### ۲) LuxAlgo Reversal Probability — تأیید اول
- BUY فقط وقتی پذیرفته می‌شود که Close آخرین کندل داخل آخرین باکس Forecast سبز باشد.
- SELL فقط وقتی پذیرفته می‌شود که Close آخرین کندل داخل آخرین باکس Forecast قرمز باشد.
- Swing Length=20، Maximum Reversals=1000، Normalize Data=false و بیشترین صدک فعال=90.
- منطق Pivot، تفکیک نمونه‌های حرکت صعودی/نزولی و محاسبه nearest-rank percentile از سورس Pine ارسالی بازسازی شده است.

### ۳) LuxAlgo Smart Money Concepts — تأیید دوم
پنج کندل بسته‌شده آخر بررسی می‌شوند. **وجود حتی یک کندل** با Close داخل ناحیه هم‌جهت کافی است؛ لازم نیست کندل آخر داخل ناحیه باشد.

تمام خانواده‌های ناحیه بررسی می‌شوند:
- Internal Bullish / Bearish Order Blocks
- Swing Bullish / Bearish Order Blocks
- Premium (نزولی/قرمز)
- Discount (صعودی/سبز)
- Equilibrium (خنثی؛ می‌تواند BUY یا SELL را تأیید کند)

تنظیمات ساختار: Internal pivot=5، Swing pivot=50، حداکثر ۵ OB فعال برای هر سمت و هر نوع ساختار، فیلتر ATR و mitigation بر اساس High/Low. Swing OB و Premium/Discount نیز برای پوشش همه ناحیه‌های درخواستی فعال‌اند.

## شرط نهایی
سیگنال فقط وقتی به تلگرام می‌رود که هر سه بخش هم‌زمان برقرار باشند:
1. الگوی قله/دره سه‌میله‌ای SQZMOM؛
2. قرار گرفتن Close آخرین کندل در باکس هم‌رنگ Reversal Probability؛
3. تأیید SMC در حداقل یکی از پنج کندل بسته‌شده آخر.

## فایل‌ها
- `scanner.py`: دریافت قراردادهای Futures، دریافت کندل‌ها، اسکن نمادها، ارسال تلگرام و جلوگیری از هشدار تکراری.
- `indicator_engine.py`: محاسبات SQZMOM، Reversal Probability و SMC.
- `requirements.txt`: وابستگی‌ها.
- `.github/workflows/scan.yml`: اجرای زمان‌بندی‌شده.
- `state.json`: حافظه جلوگیری از هشدار تکراری.

## نصب در GitHub
۱. فایل‌های این بسته را در ریشه مخزن خود بارگذاری و فایل‌های هم‌نام را جایگزین کنید.
۲. مسیر `.github/workflows/scan.yml` را دقیقاً حفظ کنید.
۳. در Settings → Secrets and variables → Actions دو Repository Secret بسازید:
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
۴. در Settings → Actions → General → Workflow permissions، گزینه Read and write permissions را فعال کنید.
۵. از تب Actions، workflow را با Run workflow یک‌بار دستی اجرا کنید.

## نکات فنی مهم
- کلید API بایننس لازم نیست؛ فقط endpointهای عمومی Futures استفاده می‌شوند.
- اجرای زمان‌بندی‌شده ممکن است به‌دلیل صف GitHub Actions چند دقیقه تأخیر داشته باشد.
- اجرای موفق روی GitHub و دریافت پیام تلگرام تا زمان اجرای workflow در مخزن شما قابل تأیید نیست.
- محاسبه SMC در Python بازسازی محاسباتی ناحیه‌ها و ساختارهای اصلی است، نه موتور رسم TradingView؛ بنابراین مرز/زمان بعضی OBها ممکن است در موارد مرزی با نسخه بصری Pine دقیقاً یکسان نباشد. منطق اصلی Reversal Probability نیز از روی Pine ارسالی پیاده‌سازی شده، اما خروجی زنده باید با TradingView تطبیق داده شود.
