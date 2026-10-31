"""BTC-gated Binance Spot scanner with a four-closed-candle permission window.

Rules:
- BTCUSDT is evaluated first on CLOSED 30-minute candles.
- A valid BTC signal opens/resets a permission window for the NEXT four
  closed 30-minute candles.
- During that window, the CMC top-500 Binance Spot/USDT universe is scanned.
- Altcoin signals must independently satisfy the unchanged OB + SQZMOM rules
  and match the current BTC permission direction.
- A new BTC signal in either direction immediately resets the window.
- Window and deduplication state are persisted in state.json.
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
WINDOW_CANDLES = 4
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


def _empty_state() -> dict[str, Any]:
    return {
        "sent": set(),
        "processed_btc": set(),
        "window_direction": None,
        "window_signal_candle_ms": None,
        "window_remaining": 0,
        "last_processed_btc_candle_ms": None,
        "last_btc_candle_allowed": False,
        "last_btc_candle_direction": None,
    }


def load_state() -> dict[str, Any]:
    state = _empty_state()
    if not STATE_FILE.exists():
        return state
    try:
        raw = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        state["sent"] = set(raw.get("sent", []))
        state["processed_btc"] = set(raw.get("processed_btc", []))
        state["window_direction"] = raw.get("window_direction")
        state["window_signal_candle_ms"] = raw.get("window_signal_candle_ms")
        state["window_remaining"] = max(0, min(
            WINDOW_CANDLES, int(raw.get("window_remaining", 0) or 0)
        ))
        state["last_processed_btc_candle_ms"] = raw.get("last_processed_btc_candle_ms")
        state["last_btc_candle_allowed"] = bool(raw.get("last_btc_candle_allowed", False))
        state["last_btc_candle_direction"] = raw.get("last_btc_candle_direction")
        # Backward compatibility: old state files may only contain sent/processed_btc.
        if state["window_direction"] not in ("BUY", "SELL"):
            state["window_direction"] = None
            state["window_remaining"] = 0
            state["last_btc_candle_allowed"] = False
        return state
    except Exception as exc:
        print(f"WARNING: state.json could not be read ({exc}); starting with empty state.")
        return state


def save_state(state: dict[str, Any]) -> None:
    payload = {
        "sent": sorted(state["sent"])[-10000:],
        "processed_btc": sorted(state["processed_btc"])[-2000:],
        "window_direction": state["window_direction"],
        "window_signal_candle_ms": state["window_signal_candle_ms"],
        "window_remaining": int(state["window_remaining"]),
        "last_processed_btc_candle_ms": state["last_processed_btc_candle_ms"],
        "last_btc_candle_allowed": bool(state["last_btc_candle_allowed"]),
        "last_btc_candle_direction": state["last_btc_candle_direction"],
    }
    STATE_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
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


def update_btc_window(state: dict[str, Any], btc_result: pd.DataFrame) -> None:
    """Process every newly closed BTC candle once and update the 4-bar window."""
    if btc_result.empty:
        raise RuntimeError("BTCUSDT returned no closed candles")

    latest_ms = int(btc_result.index[-1].timestamp() * 1000)
    previous_ms = state["last_processed_btc_candle_ms"]

    if previous_ms is None:
        # On first install, do not replay historical BTC signals. Only inspect
        # the latest closed candle; a signal there opens the next four candles.
        new_rows = btc_result.iloc[-1:]
    else:
        previous_ms = int(previous_ms)
        new_rows = btc_result[
            (btc_result.index.astype("int64") // 1_000_000) > previous_ms
        ]

    for candle_time, row in new_rows.iterrows():
        candle_ms = int(candle_time.timestamp() * 1000)
        signal = str(row["signal"])

        state["last_btc_candle_allowed"] = False
        state["last_btc_candle_direction"] = None

        if signal in ("BUY", "SELL"):
            key = candle_key("BTCUSDT", candle_time, signal)

            # Any new BTC signal, same or opposite direction, resets the
            # permission window. The signal candle itself is not one of the
            # four following candles.
            state["window_direction"] = signal
            state["window_signal_candle_ms"] = candle_ms
            state["window_remaining"] = WINDOW_CANDLES
            state["processed_btc"].add(key)

            if key not in state["sent"]:
                send_signal("BTCUSDT", signal, row)
                state["sent"].add(key)
                print(f"BTC ALERT: {key}")
            else:
                print(f"BTC signal already alerted; permission window reset: {key}")

        elif state["window_direction"] in ("BUY", "SELL") and state["window_remaining"] > 0:
            # This is one of the four closed candles AFTER the BTC signal.
            state["last_btc_candle_allowed"] = True
            state["last_btc_candle_direction"] = state["window_direction"]
            state["window_remaining"] -= 1

            if state["window_remaining"] == 0:
                print("Four-candle permission window: final permitted candle reached.")

        state["last_processed_btc_candle_ms"] = candle_ms

    # If the current latest candle was already processed on a previous
    # five-minute GitHub Actions run, retain its permission for retry/dedup.
    if not len(new_rows) and previous_ms is not None and latest_ms == int(previous_ms):
        pass

    save_state(state)


def scan_altcoins(direction: str, state: dict[str, Any]) -> tuple[int, int]:
    print(f"BTC permission active: {direction}; loading CoinMarketCap top {CMC_LIMIT}")
    top_assets = fetch_top500_cmc_symbols()
    spot_symbols = fetch_binance_spot_symbols()

    candidates = [
        (asset, spot_symbols[asset["symbol"]])
        for asset in top_assets
        if asset["symbol"] in spot_symbols and asset["symbol"] != "BTC"
    ]
    print(
        f"Scanning {len(candidates)} Binance Spot USDT pairs from CMC top "
        f"{CMC_LIMIT}; permitted direction={direction}"
    )

    checked = 0
    alerts = 0
    for asset, pair in candidates:
        try:
            evaluated = evaluate_symbol(pair)
            checked += 1
            if evaluated is None:
                continue

            signal, row = evaluated
            if signal != direction:
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

    return checked, alerts


def main() -> None:
    state = load_state()

    print(f"Stage 1: evaluating BTCUSDT on CLOSED {TIMEFRAME} candles")
    btc_candles = fetch_closed_ohlcv("BTCUSDT")
    if btc_candles.empty:
        print("BTCUSDT has insufficient closed-candle data. Scan stopped.")
        return

    btc_result = evaluate_signals(btc_candles)
    update_btc_window(state, btc_result)

    latest_btc_time = btc_result.index[-1].strftime("%Y-%m-%d %H:%M UTC")
    if not state["last_btc_candle_allowed"]:
        if state["window_remaining"] > 0:
            print(
                f"No altcoin scan on BTC candle {latest_btc_time}: "
                "waiting for the next closed 30m candle after the BTC signal."
            )
        else:
            print("BTC permission window is inactive/expired. Altcoin scan skipped.")
        save_state(state)
        return

    direction = state["last_btc_candle_direction"]
    if direction not in ("BUY", "SELL"):
        print("No valid BTC permission direction. Altcoin scan skipped.")
        return

    checked, alerts = scan_altcoins(direction, state)
    print(
        f"Done. BTC permission={direction}; checked={checked}; "
        f"same-direction alerts={alerts}; remaining future candles={state['window_remaining']}"
    )


if __name__ == "__main__":
    main()
