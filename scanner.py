import json
import os
import time
from pathlib import Path
from datetime import datetime, timezone

import ccxt
import pandas as pd

from ob_sqzmom_engine import evaluate_signals

TIMEFRAME = "30m"
CANDLE_LIMIT = 300
STATE_FILE = Path("state.json")


def telegram_send(text: str) -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    import requests
    response = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data={"chat_id": chat_id, "text": text},
        timeout=20,
    )
    response.raise_for_status()


def load_state() -> set[str]:
    if not STATE_FILE.exists():
        return set()
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return set(data.get("sent", []))
    except Exception:
        return set()


def save_state(sent: set[str]) -> None:
    # Keep only recent entries so the state file remains small.
    STATE_FILE.write_text(
        json.dumps({"sent": sorted(sent)[-5000:]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })

    exchange.urls["api"]["public"] = "https://data-api.binance.vision/api/v3"

    markets = exchange.load_markets()
    symbols = sorted(
        symbol for symbol, market in markets.items()
        if market.get("spot")
        and market.get("active", True)
        and market.get("quote") == "USDT"
        and market.get("base") not in {"USDT", "USDC", "BUSD", "TUSD", "FDUSD"}
    )

    sent = load_state()
    new_keys = set(sent)
    checked = 0
    found = 0
    now_ms = exchange.milliseconds()
    timeframe_ms = exchange.parse_timeframe(TIMEFRAME) * 1000

    print(f"Scanning {len(symbols)} Binance Spot USDT pairs on {TIMEFRAME}")

    for symbol in symbols:
        try:
            candles = exchange.fetch_ohlcv(
                symbol, timeframe=TIMEFRAME, limit=CANDLE_LIMIT
            )
            if not candles:
                continue

            # Keep only fully closed candles; never evaluate the live candle.
            closed = [row for row in candles if row[0] + timeframe_ms <= now_ms]
            if len(closed) < 80:
                continue

            df = pd.DataFrame(
                closed, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp")
            result = evaluate_signals(df)
            last = result.iloc[-1]
            checked += 1

            signal = last["signal"]
            if signal not in ("BUY", "SELL"):
                continue

            candle_ms = int(result.index[-1].timestamp() * 1000)
            key = f"{symbol}|{TIMEFRAME}|{candle_ms}|{signal}"
            if key in sent:
                continue

            arrow = "🟢 BUY" if signal == "BUY" else "🔴 SELL"
            candle_time = result.index[-1].strftime("%Y-%m-%d %H:%M UTC")
            message = (
                f"{arrow}\n"
                f"Pair: {symbol}\n"
                f"Timeframe: {TIMEFRAME}\n"
                f"Close: {float(last['close']):.10g}\n"
                f"SQZMOM: {float(last['sqzmom']):.10g}\n"
                f"Closed candle: {candle_time}\n"
                f"Rule: shorter histogram bar + price inside matching Order Block"
            )
            telegram_send(message)
            new_keys.add(key)
            found += 1
            print(f"ALERT {key}")

        except ccxt.BaseError as exc:
            print(f"Exchange error for {symbol}: {exc}")
        except Exception as exc:
            print(f"Skipped {symbol}: {type(exc).__name__}: {exc}")

    save_state(new_keys)
    print(f"Done. Checked {checked} pairs; new alerts sent: {found}")


if __name__ == "__main__":
    main()
