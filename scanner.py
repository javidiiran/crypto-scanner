"""BTC-gated Binance Spot scanner.

Execution order:
1. Evaluate BTCUSDT only, using CLOSED 30-minute candles.
2. If BTC has no new three-bar peak/trough + matching Order Block signal,
   stop immediately: do not request CoinMarketCap and do not scan altcoins.
3. If BTC signals, retrieve CoinMarketCap's top 500 by market cap, intersect
   them with Binance Spot USDT markets, and scan only signals in BTC's direction.

Both indicator calculations live in ob_sqzmom_engine.py and are preserved.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from ob_sqzmom_engine import evaluate_signals

TIMEFRAME = "30m"
INTERVAL_MS = 30 * 60 * 1000
CANDLE_LIMIT = 300
CMC_LIMIT = 500
STATE_FILE = Path("state.json")
BINANCE_BASE = "https://data-api.binance.vision"
CMC_LISTING_URL = "https://api.coinmarketcap.com/data-api/v3/cryptocurrency/listing"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (compatible; CryptoSignalScanner/1.0)"})


def telegram_send(message: str) -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    response = SESSION.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data={"chat_id": chat_id, "text": message},
        timeout=25,
    )
    response.raise_for_status()


def load_state() -> dict[str, set[str]]:
    if not STATE_FILE.exists():
        return {"sent": set(), "processed_btc": set()}
    try:
        raw = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        # Backward compatibility with the previous {"sent": [...]} state file.
        return {
            "sent": set(raw.get("sent", [])),
            "processed_btc": set(raw.get("processed_btc", [])),
        }
    except Exception:
        return {"sent": set(), "processed_btc": set()}


def save_state(state: dict[str, set[str]]) -> None:
    STATE_FILE.write_text(
        json.dumps(
            {
                "sent": sorted(state["sent"])[-10000:],
                "processed_btc": sorted(state["processed_btc"])[-1000:],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = SESSION.get(url, params=params, timeout=25)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Request failed for {url}: {last_error}")


def fetch_binance_spot_symbols() -> dict[str, str]:
    """Return {base_asset: symbol} for active Binance Spot/USDT pairs."""
    payload = get_json(f"{BINANCE_BASE}/api/v3/exchangeInfo")
    mapping: dict[str, str] = {}
    for market in payload.get("symbols", []):
        if (
            market.get("status") == "TRADING"
            and market.get("quoteAsset") == "USDT"
            and market.get("isSpotTradingAllowed", True)
        ):
            base = market.get("baseAsset")
            symbol = market.get("symbol")
            if base and symbol and base not in {"USDT", "USDC", "BUSD", "TUSD", "FDUSD"}:
                mapping.setdefault(base.upper(), symbol)
    return mapping


def fetch_top500_cmc_symbols() -> list[dict[str, Any]]:
    """Fetch CoinMarketCap's first 500 assets ordered by market cap."""
    params = {
        "start": 1,
        "limit": CMC_LIMIT,
        "sortBy": "market_cap",
        "sortType": "desc",
        "convert": "USD",
        "cryptoType": "all",
        "tagType": "all",
        "audited": "false",
    }
    payload = get_json(CMC_LISTING_URL, params=params)
    data = payload.get("data", {}) if isinstance(payload, dict) else {}
    assets = data.get("cryptoCurrencyList", [])
    if not isinstance(assets, list) or not assets:
        raise RuntimeError("CoinMarketCap did not return its top-500 listing data")

    # Keep the highest-ranked occurrence if CMC has duplicate ticker symbols.
    ranked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in assets[:CMC_LIMIT]:
        symbol = str(item.get("symbol", "")).upper().strip()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        ranked.append({
            "symbol": symbol,
            "name": str(item.get("name", symbol)),
            "rank": item.get("cmcRank", len(ranked) + 1),
        })
    return ranked


def fetch_closed_ohlcv(symbol: str) -> pd.DataFrame:
    rows = get_json(
        f"{BINANCE_BASE}/api/v3/klines",
        params={"symbol": symbol, "interval": TIMEFRAME, "limit": CANDLE_LIMIT},
    )
    if not rows:
        return pd.DataFrame()

    now_ms = int(time.time() * 1000)
    closed_rows = [row for row in rows if int(row[0]) + INTERVAL_MS <= now_ms]
    if len(closed_rows) < 80:
        return pd.DataFrame()

    df = pd.DataFrame(
        [
            [int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])]
            for row in closed_rows
        ],
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp")


def evaluate_symbol(symbol: str) -> tuple[str, pd.Series] | None:
    candles = fetch_closed_ohlcv(symbol)
    if candles.empty:
        return None
    result = evaluate_signals(candles)
    last = result.iloc[-1]
    signal = str(last["signal"])
    if signal not in ("BUY", "SELL"):
        return None
    return signal, last


def candle_key(symbol: str, candle_time: pd.Timestamp, signal: str) -> str:
    candle_ms = int(candle_time.timestamp() * 1000)
    return f"{symbol}|{TIMEFRAME}|{candle_ms}|{signal}"


def send_signal(symbol: str, signal: str, row: pd.Series, rank: str | int | None = None) -> None:
    emoji = "🟢 BUY" if signal == "BUY" else "🔴 SELL"
    candle_time = row.name.strftime("%Y-%m-%d %H:%M UTC")
    lines = [emoji, f"Pair: {symbol}", f"Timeframe: {TIMEFRAME}"]
    if rank is not None:
        lines.append(f"CoinMarketCap rank: {rank}")
    lines.extend([
        f"Close: {float(row['close']):.10g}",
        f"SQZMOM (latest closed bar): {float(row['sqzmom']):.10g}",
        f"Three-bar pattern: {'peak' if signal == 'SELL' else 'trough'}",
        f"Order Block: {'red/bearish' if signal == 'SELL' else 'green/bullish'} zone",
        f"Confirmation candle: {candle_time}",
    ])
    telegram_send("\n".join(lines))


def main() -> None:
    state = load_state()
    print(f"Stage 1: evaluating BTCUSDT only on closed {TIMEFRAME} candles")

    # The BTC check is always first. No CMC request or altcoin scan occurs
    # unless this exact BTC 30m candle has a valid three-bar signal.
    btc = evaluate_symbol("BTCUSDT")
    if btc is None:
        print("BTCUSDT has no valid signal. Altcoin scan skipped.")
        return

    btc_signal, btc_row = btc
    btc_key = candle_key("BTCUSDT", btc_row.name, btc_signal)
    if btc_key in state["processed_btc"]:
        print(f"BTC signal already processed: {btc_key}. No repeat scan.")
        return

    if btc_key not in state["sent"]:
        send_signal("BTCUSDT", btc_signal, btc_row)
        state["sent"].add(btc_key)
        save_state(state)
        print(f"BTC ALERT: {btc_key}")

    # Only now retrieve the top-500 list and Binance Spot market universe.
    print(f"Stage 2: BTC {btc_signal}; loading CoinMarketCap top {CMC_LIMIT}")
    top_assets = fetch_top500_cmc_symbols()
    spot_symbols = fetch_binance_spot_symbols()

    candidates = [
        (asset, spot_symbols[asset["symbol"]])
        for asset in top_assets
        if asset["symbol"] in spot_symbols and asset["symbol"] != "BTC"
    ]
    print(f"Scanning {len(candidates)} Binance Spot USDT pairs from CMC top {CMC_LIMIT}; direction={btc_signal}")

    checked = 0
    alerts = 0
    for asset, pair in candidates:
        try:
            evaluated = evaluate_symbol(pair)
            checked += 1
            if evaluated is None:
                continue
            signal, row = evaluated
            if signal != btc_signal:
                continue

            key = candle_key(pair, row.name, signal)
            if key in state["sent"]:
                continue

            send_signal(pair, signal, row, asset.get("rank"))
            state["sent"].add(key)
            save_state(state)
            alerts += 1
            print(f"ALERT {key} (CMC rank {asset.get('rank')})")
        except Exception as exc:
            print(f"Skipped {pair}: {type(exc).__name__}: {exc}")

    state["processed_btc"].add(btc_key)
    save_state(state)
    print(f"Done. BTC={btc_signal}; checked={checked}; same-direction alerts={alerts}")


if __name__ == "__main__":
    main()
