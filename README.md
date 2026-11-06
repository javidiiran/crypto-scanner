# Scanner 4 — Kraken Futures

- Kraken Futures markets; 15-minute candles; closed candles only.
- Each tradable contract is evaluated independently. BTC does not gate altcoin signals.
- Primary signal uses only the Pine `B-Xtrender Color` line (`maShortTermXtrender`): green→red = SELL, red→green = BUY. The two transitions covered are the second and third candles in the latest three closed candles.
- Confirmation uses active green/red Delta Reaction Zones, confirmed 12-bar pivots, six latest closed candles for touch, 8-zone cap, and 20-tick same-color merging.
- Telegram secrets: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.
- Run: `pip install -r requirements.txt && python scanner.py`.

The report's signal rows contain only a direction circle, contract symbol, 24-hour percentage change, and 1-hour percentage change. BUYs are sorted by strongest 24h gain; SELLs by largest 24h decline.
