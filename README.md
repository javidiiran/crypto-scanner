# Scanner 4 — Kraken Futures

Python scanner for all tradable Kraken Futures contracts using 15-minute closed candles.

## Locked signal rules

- Each contract, including BTC, is evaluated independently. BTC never filters, authorizes, blocks, or delays another market.
- B-Xtrender Color uses the short line (T3(5)); rising is GREEN and falling is RED.
- Examine the last three closed indicator-color observations. A RED→GREEN transition is BUY; GREEN→RED is SELL. A transition on either of the two steps in that three-candle window is accepted. If both steps changed, the most recent transition determines direction.
- Delta Reaction Zones confirmation: green support confirms BUY; red resistance confirms SELL. At least one of the last six closed candles must overlap/touch an active matching zone; the close need not be inside. A breached zone is invalid. Active-zone collision/overlap vetoes the symbol.
- A closed candle older than 15 minutes beyond its close time is rejected.
- Output is BUY first (highest 24h gain first), then SELL (largest 24h decline first). Each signal shows only symbol, 24h change and 1h change, with a green BUY dot or red SELL dot.

## Delta-zone calculation parameters

- Smoothed signed volume: EMA(3), cumulative delta source
- Pivot length: 12
- Zone half-width: ATR(14) × 0.35
- Maximum active zones: 8
- Same-side zone merge gap: 20 ticks
- Confirmation window: 6 closed candles

## Install in GitHub

1. Extract this ZIP into the root of the repository, preserving `.github/workflows/scan.yml`.
2. In repository **Settings → Secrets and variables → Actions**, add:
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
3. Commit and push. Run **Actions → Scanner 4 - Kraken Futures → Run workflow** once to test.
4. The workflow also runs every five minutes and commits `state.json` to prevent repeated alerts for the same symbol/candle.

Telegram secrets are optional for local testing; without them, the scanner writes results to the GitHub Actions log.

## Local run

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scanner.py
```

The scanner uses the public Kraken Futures REST endpoints. Network availability, Kraken's live response schema, Telegram delivery and a real GitHub Actions run must be validated in the user's own environment; they are not represented as live-tested by the local unit tests.
