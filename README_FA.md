# KCEX Spot Crypto Scanner (30m)

## Final signal rules
- SQZMOM (LazyBear calculation preserved) issues the three-closed-candle peak/trough setup.
- BUY is confirmed only when MFI(14) > 50; SELL only when MFI(14) < 50.
- No Order Book, Order Blocks, Dynamic Profile, EMA 180, scoring, or additional indicator.
- BTC is evaluated only as a reference. It never blocks an altcoin signal; Telegram labels an opposite-direction altcoin signal.
- Closed 30-minute candles only. Telegram alerts; GitHub Actions every five minutes. No orders are placed.

## Critical KCEX API note
The public REST documentation found for `/data/api/v1/getMarketConfig` and `/data/api/v1/getKLine` documents `https://api.kkcex.com`, a legacy domain. It does **not** establish that these endpoints work on the current `kcex.com` service. The workflow therefore takes `KCEX_API_BASE` from a GitHub Actions repository variable. Set it to a base URL only after confirming it is the current KCEX Spot API host. Default in `scanner.py` is the documented legacy host `https://api.kkcex.com` and may fail if retired.

## GitHub setup
1. Replace repository `scanner.py` and `ob_sqzmom_engine.py` with these files.
2. Replace `.github/workflows/scan.yml` with the supplied workflow.
3. Keep `requirements.txt`.
4. In Settings → Secrets and variables → Actions, add secrets `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.
5. Add variable `KCEX_API_BASE` only when you have the correct current KCEX Spot API base URL.
6. Run Actions → KCEX Spot Scanner → Run workflow. Check logs before relying on alerts.

The scanner is read-only. It requires no KCEX API key and does not send orders.
