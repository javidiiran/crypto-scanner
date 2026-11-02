"""Top-300-by-market-cap Kraken scanner, 30m SQZMOM + SMC, BTC phase gate."""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from indicator_engine import evaluate_signals, latest_confirmed_btc_pivot, squeeze_momentum

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
KRAKEN_SPOT_BASE = "https://api.kraken.com/0/public"
KRAKEN_FUTURES_REST = "https://futures.kraken.com/derivatives/api/v3"
KRAKEN_FUTURES_CHARTS = "https://futures.kraken.com/api/charts/v1"
TIMEFRAME = "30m"
INTERVAL_MS = 30 * 60 * 1000
TOP_MARKET_CAP = 300
MIN_CANDLES = 250
MAX_WORKERS = 6
STATE_FILE = Path("state.json")
QUOTE_PRIORITY = {"USD": 0, "USDT": 1, "USDC": 2, "EUR": 3}
SYMBOL_ALIASES = {
    "BTC": "XBT", "WBTC": "XBT", "ETH": "ETH", "WETH": "ETH",
    "DOGE": "XDG", "XDG": "XDG", "BCH": "BCH", "LUNA": "LUNA",
}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Kraken-Top300-SQZMOM-SMC-Scanner/2.0", "Accept": "application/json"})
retry = Retry(total=5, connect=5, read=4, status=4, backoff_factor=1.0,
              status_forcelist=(408, 429, 500, 502, 503, 504),
              allowed_methods=frozenset(["GET"]), respect_retry_after_header=True)
adapter = HTTPAdapter(max_retries=retry, pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS)
SESSION.mount("https://", adapter)


def get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    response = SESSION.get(url, params=params, timeout=(12, 35))
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict):
        if payload.get("error"):
            raise RuntimeError(f"API error: {payload['error']}")
        if payload.get("result") == "error":
            raise RuntimeError(f"API error: {payload.get('errors', payload)}")
    return payload


def telegram_send(message: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        raise RuntimeError("GitHub Secrets TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required")
    response = SESSION.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data={"chat_id": chat_id, "text": message, "disable_web_page_preview": True},
        timeout=(10, 25),
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("ok"):
        raise RuntimeError(f"Telegram API rejected the message: {payload}")


def load_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return {"sent": [], "btc_last_event": ""}
    try:
        raw = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        # Backward-compatible with the previous state.json format.
        if isinstance(raw, list):
            return {"sent": raw, "btc_last_event": ""}
        return {"sent": list(raw.get("sent", [])), "btc_last_event": str(raw.get("btc_last_event", ""))}
    except (OSError, ValueError, TypeError) as exc:
        print(f"WARNING: state.json unreadable; starting fresh: {exc}")
        return {"sent": [], "btc_last_event": ""}


def save_state(state: dict[str, Any]) -> None:
    state["sent"] = sorted(set(state.get("sent", [])))[-20000:]
    temp = STATE_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(STATE_FILE)


def normalize_symbol(value: str) -> str:
    symbol = str(value or "").upper().strip()
    for prefix in ("XX", "X", "Z"):
        if symbol.startswith(prefix) and len(symbol) > 3:
            candidate = symbol[len(prefix):]
            if candidate in {"XBT", "ETH", "XDG", "USD", "EUR", "USDT", "USDC"}:
                symbol = candidate
                break
    return SYMBOL_ALIASES.get(symbol, symbol)


def fetch_top_market_cap() -> list[dict[str, Any]]:
    """CoinGecko top 300, sorted by market cap; two pages (250 + 50)."""
    coins: list[dict[str, Any]] = []
    for page in (1, 2):
        per_page = 250
        payload = get_json(
            f"{COINGECKO_BASE}/coins/markets",
            {"vs_currency": "usd", "order": "market_cap_desc", "per_page": per_page,
             "page": page, "sparkline": "false"},
        )
        if not isinstance(payload, list):
            raise RuntimeError("CoinGecko returned an unexpected market-cap response")
        coins.extend(payload)
        time.sleep(1.2)
    # Preserve the ranking order and only keep real ranked assets.
    result = []
    # CoinGecko pages are each 250 rows; use only the first 50 rows of page 2.
    for ordinal, item in enumerate(coins[:TOP_MARKET_CAP], start=1):
        if not item.get("symbol") or item.get("market_cap") is None:
            continue
        rank = item.get("market_cap_rank") or ordinal
        result.append({
            "id": str(item.get("id", "")), "name": str(item.get("name", "")),
            "symbol": str(item["symbol"]).upper(), "rank": int(rank),
            "market_cap": item.get("market_cap"),
        })
    result.sort(key=lambda x: x["rank"])
    return result[:TOP_MARKET_CAP]


def _market_choice(current: dict[str, Any] | None, candidate: dict[str, Any]) -> dict[str, Any]:
    if current is None:
        return candidate
    return candidate if candidate["quote_priority"] < current["quote_priority"] else current


def fetch_kraken_market_maps() -> dict[str, dict[str, Any]]:
    """Build symbol -> Kraken market map; perpetual first, then spot fallback."""
    by_base: dict[str, dict[str, Any]] = {}

    futures = get_json(f"{KRAKEN_FUTURES_REST}/instruments")
    instruments = futures.get("instruments", []) if isinstance(futures, dict) else []
    for item in instruments:
        symbol = str(item.get("symbol") or "").upper()
        if not symbol.startswith(("PF_", "PI_")):
            continue
        if item.get("tradeable") is False or item.get("isExpired") is True:
            continue
        status = str(item.get("status", "")).lower()
        if status and status not in {"online", "trading", "active"}:
            continue
        contract = symbol.split("_", 1)[1]
        quote_ccy = next((q for q in ("USDT", "USDC", "USD", "EUR") if contract.endswith(q)), None)
        if not quote_ccy:
            continue
        base = normalize_symbol(contract[:-len(quote_ccy)])
        if not base:
            continue
        candidate = {
            "symbol": symbol, "market_type": "Futures Perpetual", "pair": symbol,
            "quote": quote_ccy, "quote_priority": QUOTE_PRIORITY.get(quote_ccy, 99),
        }
        by_base[base] = _market_choice(by_base.get(base), candidate)

    spot = get_json(f"{KRAKEN_SPOT_BASE}/AssetPairs")
    pairs = spot.get("result", {}) if isinstance(spot, dict) else {}
    for key, item in pairs.items():
        wsname = str(item.get("wsname") or "")
        altname = str(item.get("altname") or key)
        if "/" in wsname:
            base_raw, quote_raw = wsname.split("/", 1)
        else:
            base_raw = str(item.get("base") or "")
            quote_raw = str(item.get("quote") or "")
        base = normalize_symbol(base_raw)
        quote_ccy = normalize_symbol(quote_raw)
        if not base or quote_ccy not in QUOTE_PRIORITY:
            continue
        candidate = {
            "symbol": altname, "market_type": "Spot", "pair": altname,
            "quote": quote_ccy, "quote_priority": QUOTE_PRIORITY[quote_ccy],
        }
        # A perpetual contract is preferred. Otherwise choose best spot quote.
        by_base[base] = _market_choice(by_base.get(base), candidate)
    return by_base


def map_top300_to_kraken(coins: list[dict[str, Any]], markets: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    mapped: list[dict[str, Any]] = []
    used_bases: set[str] = set()
    for coin in coins:
        base = normalize_symbol(coin["symbol"])
        market = markets.get(base)
        if not market:
            continue
        # If CoinGecko has duplicate tickers, use the highest-ranked match only.
        if base in used_bases:
            continue
        used_bases.add(base)
        mapped.append({**coin, **market, "base": base})
    return mapped


def _normalize_candle_frame(rows: list[Any], now_ms: int, source: str) -> pd.DataFrame:
    normalized = []
    for row in rows:
        try:
            if isinstance(row, dict):
                stamp = int(row["time"])
                opn, high, low, close = row["open"], row["high"], row["low"], row["close"]
                volume = row.get("volume", 0)
            else:
                stamp = int(row[0])
                if source == "spot":
                    opn, high, low, close, volume = row[1], row[2], row[3], row[4], row[6]
                else:
                    opn, high, low, close, volume = row[1:6]
            if stamp < 10**12:
                stamp *= 1000
            if stamp + INTERVAL_MS > now_ms:
                continue
            normalized.append((stamp, float(opn), float(high), float(low), float(close), float(volume)))
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    if not normalized:
        return pd.DataFrame()
    frame = pd.DataFrame(normalized, columns=["timestamp", "open", "high", "low", "close", "volume"])
    frame = frame.drop_duplicates("timestamp").sort_values("timestamp")
    frame["timestamp"] = pd.to_datetime(frame.timestamp, unit="ms", utc=True)
    frame = frame.set_index("timestamp")
    return frame if len(frame) >= MIN_CANDLES else pd.DataFrame()


def fetch_futures_candles(symbol: str, now_ms: int) -> pd.DataFrame:
    end_s = now_ms // 1000
    start_s = end_s - 1005 * (INTERVAL_MS // 1000)
    url = f"{KRAKEN_FUTURES_CHARTS}/trade/{quote(symbol, safe='_')}/{TIMEFRAME}"
    payload = get_json(url, {"from": start_s, "to": end_s, "count": 1000})
    rows = payload.get("candles", []) if isinstance(payload, dict) else []
    return _normalize_candle_frame(rows, now_ms, "futures")


def fetch_spot_candles(pair: str, now_ms: int) -> pd.DataFrame:
    payload = get_json(f"{KRAKEN_SPOT_BASE}/OHLC", {"pair": pair, "interval": 30})
    if not isinstance(payload, dict):
        return pd.DataFrame()
    result = payload.get("result", {})
    rows = next((value for key, value in result.items() if key != "last" and isinstance(value, list)), [])
    return _normalize_candle_frame(rows, now_ms, "spot")


def fetch_market_candles(market: dict[str, Any], now_ms: int) -> pd.DataFrame:
    if market["market_type"] == "Futures Perpetual":
        try:
            frame = fetch_futures_candles(market["pair"], now_ms)
            if not frame.empty:
                return frame
        except requests.RequestException:
            # Spot fallback is attempted only if a spot pair is also available
            # in the selected market descriptor; current selection is one market.
            raise
        return pd.DataFrame()
    return fetch_spot_candles(market["pair"], now_ms)


def btc_phase_description(signal: str) -> str:
    return "نزولی؛ از قله مثبت به سمت دره منفی" if signal == "SELL" else "صعودی؛ از دره منفی به سمت قله مثبت"


def evaluate_asset(coin: dict[str, Any], now_ms: int, btc_phase: str, phase_start_ms: int) -> dict[str, Any] | None:
    candles = fetch_market_candles(coin, now_ms)
    if candles.empty:
        return None
    result = evaluate_signals(candles)
    if result.empty:
        return None
    last = result.iloc[-1]
    signal = str(last.signal)
    if signal not in ("BUY", "SELL") or signal != btc_phase:
        return None
    candle_time = result.index[-1]
    # Do not emit an old altcoin pivot from before the current BTC phase began.
    if int(candle_time.timestamp() * 1000) < phase_start_ms:
        return None
    return {
        "symbol": coin["symbol"], "name": coin["name"], "rank": coin["rank"],
        "market_type": coin["market_type"], "pair": coin["pair"],
        "signal": signal, "timestamp_ms": int(candle_time.timestamp() * 1000),
        "candle_time": candle_time.strftime("%Y-%m-%d %H:%M UTC"),
        "close": float(last.close), "sqzmom": float(last.sqzmom),
        "peak_trough": float(last.sqzmom_peak_trough),
        "smc_matches": str(last.smc_matches), "smc_match_age": int(last.smc_match_age),
        "btc_phase": btc_phase,
    }


def btc_alert(btc: dict[str, Any], event: dict[str, Any]) -> str:
    signal = event["signal"]
    title = "🟢 BTC BUY — دره معتبر SQZMOM" if signal == "BUY" else "🔴 BTC SELL — قله معتبر SQZMOM"
    return "\n".join([
        title,
        f"نماد مرجع: BTC / {btc['pair']} ({btc['market_type']})",
        f"تایم‌فریم: {TIMEFRAME}",
        f"موقعیت SQZMOM: {'دره زیر صفر' if signal == 'BUY' else 'قله بالای صفر'}",
        f"مقدار میله قله/دره: {event['pivot_value']:.10g}",
        f"جهت فعال اسکن: {btc_phase_description(signal)}",
        f"زمان قله/دره: {event['pivot_time'].strftime('%Y-%m-%d %H:%M UTC')}",
        f"زمان تأیید (دو کندل بسته‌شده بعد): {event['confirmation_time'].strftime('%Y-%m-%d %H:%M UTC')}",
        "BTC بدون نیاز به تأیید SMC است.",
    ])


def asset_alert(item: dict[str, Any]) -> str:
    title = "🟢 BUY / LONG" if item["signal"] == "BUY" else "🔴 SELL / SHORT"
    return "\n".join([
        title,
        f"ارز: {item['name']} ({item['symbol']}) — رتبه مارکت‌کپ: #{item['rank']}",
        f"بازار Kraken: {item['market_type']} — {item['pair']}",
        f"تایم‌فریم: {TIMEFRAME}",
        f"قیمت بسته‌شدن: {item['close']:.10g}",
        f"SQZMOM میله قله/دره: {item['peak_trough']:.10g}",
        f"SQZMOM آخرین میله: {item['sqzmom']:.10g}",
        f"تأیید SMC در ۵ کندل بسته‌شده آخر: {item['smc_matches']}",
        f"روند فعلی بیت‌کوین: {btc_phase_description(item['btc_phase'])}",
        f"زمان کندل بسته‌شده: {item['candle_time']}",
        "هشدار تحلیلی است؛ هیچ سفارش یا معامله‌ای ثبت نمی‌شود.",
    ])


def main() -> None:
    now_ms = int(time.time() * 1000)
    print(f"Starting top-{TOP_MARKET_CAP} market-cap Kraken scanner | {TIMEFRAME} | BTC SQZMOM phase gate")
    coins = fetch_top_market_cap()
    markets = fetch_kraken_market_maps()
    universe = map_top300_to_kraken(coins, markets)
    if not universe:
        raise RuntimeError("No top-300 market-cap assets could be matched to Kraken Spot/Futures markets")
    btc = next((x for x in universe if x["symbol"] == "BTC" or x["base"] == "XBT"), None)
    if btc is None:
        raise RuntimeError("BTC is not present in the Kraken-matched top-300 universe")

    btc_candles = fetch_market_candles(btc, now_ms)
    if btc_candles.empty:
        raise RuntimeError("Not enough closed 30m BTC candles from Kraken to determine market phase")
    btc_hist = squeeze_momentum(btc_candles)
    btc_event = latest_confirmed_btc_pivot(btc_hist)
    if btc_event is None:
        raise RuntimeError("BTC SQZMOM has no confirmed positive peak or negative trough yet; altcoin scan paused")
    btc_phase = str(btc_event["signal"])
    btc_event_key = f"BTC|{TIMEFRAME}|{int(btc_event['pivot_time'].timestamp() * 1000)}|{btc_phase}"

    state = load_state()
    sent = set(state.get("sent", []))
    print(f"BTC phase: {btc_phase_description(btc_phase)} | pivot={btc_event['pivot_value']:.8g}")
    print(f"Top-300 assets: {len(coins)} | Kraken matched: {len(universe)} | BTC market: {btc['market_type']} {btc['pair']}")

    # The BTC pivot alert is always delivered before any altcoin alerts.
    if state.get("btc_last_event", "") != btc_event_key and btc_event_key not in sent:
        telegram_send(btc_alert(btc, btc_event))
        sent.add(btc_event_key)
        state["btc_last_event"] = btc_event_key
        state["sent"] = list(sent)
        save_state(state)
        print(f"BTC FIRST ALERT: {btc_phase} {btc_event['pivot_time']}")
    else:
        state["btc_last_event"] = btc_event_key

    phase_start_ms = int(btc_event["confirmation_time"].timestamp() * 1000)
    scan_universe = [x for x in universe if x is not btc and x["base"] != "XBT"]
    found: list[dict[str, Any]] = []
    checked = errors = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(evaluate_asset, coin, now_ms, btc_phase, phase_start_ms): coin for coin in scan_universe}
        for future in as_completed(futures):
            coin = futures[future]
            try:
                item = future.result()
                checked += 1
                if item:
                    key = f"{item['pair']}|{TIMEFRAME}|{item['timestamp_ms']}|{item['signal']}"
                    if key not in sent:
                        item["dedupe_key"] = key
                        found.append(item)
            except requests.HTTPError as exc:
                errors += 1
                status = exc.response.status_code if exc.response is not None else "unknown"
                print(f"HTTP error for {coin['symbol']} rank#{coin['rank']} (status {status}): {exc}")
                if status == 429:
                    raise
            except Exception as exc:
                errors += 1
                print(f"Skipped {coin['symbol']} rank#{coin['rank']}: {type(exc).__name__}: {exc}")

    # Preserve market-cap order; do not sort by signal or completion time.
    found.sort(key=lambda x: x["rank"])
    alerts = 0
    for item in found:
        telegram_send(asset_alert(item))
        sent.add(item["dedupe_key"])
        state["sent"] = list(sent)
        state["btc_last_event"] = btc_event_key
        save_state(state)
        alerts += 1
        print(f"ALERT {item['symbol']} {item['signal']} rank#{item['rank']} {item['candle_time']}")

    state["sent"] = list(sent)
    state["btc_last_event"] = btc_event_key
    save_state(state)
    print(f"Finished: checked={checked}/{len(scan_universe)} new_asset_alerts={alerts} errors={errors}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
