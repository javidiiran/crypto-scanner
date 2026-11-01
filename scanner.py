"""BTC-gated Binance Spot scanner with a four-closed-candle permission window.

Rules:
- BTCUSDT is evaluated first on CLOSED 30-minute candles.
- A valid BTC signal opens/resets a permission window for the NEXT four
  closed 30-minute candles.
- During that window, the CMC top-500 Binance Spot/USDT universe is scanned.
- Altcoin signals must independently satisfy the Dynamic Profile + SQZMOM rules
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


def _mfi_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    raw_flow = typical * df["volume"].clip(lower=0)
    direction = typical.diff()
    positive = raw_flow.where(direction > 0, 0.0).rolling(period).sum()
    negative = raw_flow.where(direction < 0, 0.0).rolling(period).sum()
    ratio = positive / negative.replace(0, float("nan"))
    mfi = 100.0 - (100.0 / (1.0 + ratio))
    mfi = mfi.mask((negative == 0) & (positive > 0), 100.0)
    mfi = mfi.mask((positive == 0) & (negative > 0), 0.0)
    return mfi.fillna(50.0)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def score_signal(candles: pd.DataFrame, result: pd.DataFrame, signal: str,
                 hourly: pd.DataFrame | None = None) -> dict[str, Any]:
    """Experimental 0-100 quality score; never changes the underlying signal."""
    row = result.iloc[-1]
    prev = result.iloc[-2]
    close = max(float(row["close"]), 1e-12)

    # 1) SQZMOM strength relative to ATR, comparable across different coins.
    tr = pd.concat([
        candles["high"] - candles["low"],
        (candles["high"] - candles["close"].shift()).abs(),
        (candles["low"] - candles["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    if not pd.notna(atr) or atr <= 0:
        atr = close * 0.01
    hist = float(row["sqzmom"])
    prev_hist = float(prev["sqzmom"])
    strength_ratio = abs(hist) / max(atr, close * 1e-8)
    strength_score = 25.0 * _clamp(strength_ratio / 1.25)

    # 2) Contraction/turn quality: stronger contraction after a larger prior bar.
    prior_abs = abs(prev_hist)
    contraction = (prior_abs - abs(hist)) / max(prior_abs, atr * 1e-6)
    reversal_score = 20.0 * _clamp(contraction / 0.45)

    # 3) MFI direction confirmation, with a penalty for extreme late entries.
    mfi = float(_mfi_series(candles).iloc[-1])
    directional_mfi = mfi if signal == "BUY" else 100.0 - mfi
    mfi_score = 20.0 * _clamp((directional_mfi - 45.0) / 30.0)
    if (signal == "BUY" and mfi >= 85) or (signal == "SELL" and mfi <= 15):
        mfi_score *= 0.65

    # 4) Closed-candle volume relative to its 20-bar average.
    vol_avg = float(candles["volume"].rolling(20).mean().iloc[-1])
    vol_ratio = float(row["volume"]) / max(vol_avg, 1e-12)
    volume_score = 15.0 * _clamp((vol_ratio - 0.65) / 1.35)
    if vol_ratio > 4.0:
        volume_score *= 0.75  # avoid over-rewarding isolated volume spikes

    # 5) One-hour EMA trend confirmation (fetch only for actual signal candidates).
    trend_score = 5.0
    if hourly is not None and len(hourly) >= 55:
        ema20 = hourly["close"].ewm(span=20, adjust=False).mean().iloc[-1]
        ema50 = hourly["close"].ewm(span=50, adjust=False).mean().iloc[-1]
        aligned = (float(ema20) > float(ema50)) if signal == "BUY" else (float(ema20) < float(ema50))
        trend_score = 10.0 if aligned else 0.0

    # 6) Entry quality: penalize chasing price far from the 30m EMA20.
    ema20_30 = float(candles["close"].ewm(span=20, adjust=False).mean().iloc[-1])
    distance_atr = abs(close - ema20_30) / max(atr, 1e-12)
    entry_score = 10.0 * (1.0 - _clamp((distance_atr - 0.25) / 2.25))

    components = {
        "SQZMOM strength": round(strength_score, 2),
        "Histogram turn": round(reversal_score, 2),
        "MFI(14)": round(mfi_score, 2),
        "Volume": round(volume_score, 2),
        "1H trend": round(trend_score, 2),
        "Entry quality": round(entry_score, 2),
    }
    total = int(round(sum(components.values())))
    grade = "A" if total >= 85 else "B" if total >= 70 else "C" if total >= 55 else "D"
    return {"score": total, "grade": grade, "mfi": round(mfi, 1),
            "volume_ratio": round(vol_ratio, 2), "components": components}


def fetch_closed_hourly(symbol: str) -> pd.DataFrame:
    rows = get_json(
        f"{BINANCE_BASE}/api/v3/klines",
        params={"symbol": symbol, "interval": "1h", "limit": 100},
    )
    now_ms = int(time.time() * 1000)
    closed_rows = [row for row in rows if int(row[0]) + 60 * 60 * 1000 <= now_ms]
    if len(closed_rows) < 55:
        return pd.DataFrame()
    df = pd.DataFrame(
        [[int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])] for x in closed_rows],
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


def send_signal(symbol: str, signal: str, row: pd.Series, rank: str | int | None = None,
                quality: dict[str, Any] | None = None) -> None:
    emoji = "🟢 BUY" if signal == "BUY" else "🔴 SELL"
    candle_time = row.name.strftime("%Y-%m-%d %H:%M UTC")
    lines = [emoji, f"Pair: {symbol}", f"Timeframe: {TIMEFRAME}"]
    if rank is not None:
        lines.append(f"CoinMarketCap rank: {rank}")
    if quality is not None:
        lines.append(f"Quality: {quality['score']}/100 | Grade: {quality['grade']}")
        lines.append(f"MFI(14): {quality['mfi']} | Volume/20 avg: {quality['volume_ratio']}x")
    lines.extend([
        f"Close: {float(row['close']):.10g}",
        f"SQZMOM (latest closed bar): {float(row['sqzmom']):.10g}",
        f"Three-bar pattern: {'peak' if signal == 'SELL' else 'trough'}",
        f"Profile zone: {str(row.get('profile_zone_type', 'node'))} / {'red-bearish' if signal == 'SELL' else 'green-bullish'}",
        f"Zone bounds: {float(row['profile_zone_bottom']):.10g} - {float(row['profile_zone_top']):.10g}" if pd.notna(row.get('profile_zone_bottom')) and pd.notna(row.get('profile_zone_top')) else "Zone bounds: unavailable",
        f"Pattern candle in zone: {int(row.get('profile_pattern_candle', -1))} bar(s) from confirmation" if int(row.get('profile_pattern_candle', -1)) >= -2 else "Pattern candle in zone: unavailable",
        f"Confirmation candle: {candle_time}",
    ])
    telegram_send("\n".join(lines))


def update_btc_window(state: dict[str, Any], btc_result: pd.DataFrame, btc_candles: pd.DataFrame) -> None:
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
                quality = score_signal(btc_candles.loc[:candle_time], btc_result.loc[:candle_time], signal)
                send_signal("BTCUSDT", signal, row, quality=quality)
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
    print(f"Scanning {len(candidates)} Binance Spot USDT pairs; direction={direction}")

    checked = 0
    pending: list[dict[str, Any]] = []
    for asset, pair in candidates:
        try:
            candles = fetch_closed_ohlcv(pair)
            if candles.empty:
                continue
            result = evaluate_signals(candles)
            row = result.iloc[-1]
            checked += 1
            signal = str(row["signal"])
            if signal != direction:
                continue
            key = candle_key(pair, row.name, signal)
            if key in state["sent"]:
                continue
            hourly = fetch_closed_hourly(pair)
            quality = score_signal(candles, result, signal, hourly if not hourly.empty else None)
            pending.append({"pair": pair, "asset": asset, "signal": signal,
                            "row": row, "quality": quality, "key": key})
        except Exception as exc:
            print(f"Skipped {pair}: {type(exc).__name__}: {exc}")

    # Sort the complete batch before sending; preserve the existing dedup key.
    pending.sort(key=lambda item: (item["quality"]["score"], -int(item["asset"].get("rank") or 999999)), reverse=True)
    for item in pending:
        send_signal(item["pair"], item["signal"], item["row"],
                    item["asset"].get("rank"), item["quality"])
        state["sent"].add(item["key"])
        save_state(state)
        print(f"ALERT {item['key']} | score={item['quality']['score']} grade={item['quality']['grade']}")
    return checked, len(pending)


def main() -> None:
    state = load_state()

    print(f"Stage 1: evaluating BTCUSDT on CLOSED {TIMEFRAME} candles")
    btc_candles = fetch_closed_ohlcv("BTCUSDT")
    if btc_candles.empty:
        print("BTCUSDT has insufficient closed-candle data. Scan stopped.")
        return

    btc_result = evaluate_signals(btc_candles)
    update_btc_window(state, btc_result, btc_candles)

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
