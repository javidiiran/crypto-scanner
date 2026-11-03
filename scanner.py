"""Kraken Futures Wave Trend + Order Blocks/FVG scanner (30-minute, closed bars)."""
from __future__ import annotations

import json
import math
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

from indicator_engine import analyze_market

REST_BASE = "https://futures.kraken.com/derivatives/api/v3"
CHART_BASE = "https://futures.kraken.com/api/charts/v1"
TIMEFRAME = "30m"
INTERVAL_MS = 30 * 60 * 1000
CANDLE_LIMIT = 1000
MIN_CANDLES = 60
# Reject chart-cache responses whose newest CLOSED candle is older than this.
MAX_CANDLE_STALENESS_MS = 2 * INTERVAL_MS
CHART_FETCH_ATTEMPTS = 3
MAX_WORKERS = 5
STATE_FILE = Path("state.json")
PERPETUAL_PREFIXES = ("PF_", "PI_")
BTC_SYMBOLS = ("PF_XBTUSD", "PI_XBTUSD")

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Kraken-Futures-WaveTrend-OB-FVG-Scanner/1.2.0", "Accept": "application/json", "Cache-Control": "no-cache", "Pragma": "no-cache"})
retry = Retry(total=4, connect=4, read=3, status=3, backoff_factor=0.7,
              status_forcelist=(408, 429, 500, 502, 503, 504),
              allowed_methods=frozenset(["GET"]), respect_retry_after_header=True)
adapter = HTTPAdapter(max_retries=retry, pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS)
SESSION.mount("https://", adapter)


def get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    response = SESSION.get(url, params=params, timeout=(10, 30))
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"Kraken returned non-JSON data: {url}") from exc
    if isinstance(payload, dict) and payload.get("result") == "error":
        raise RuntimeError(f"Kraken API error: {payload.get('errors', payload)}")
    if isinstance(payload, dict) and payload.get("error"):
        raise RuntimeError(f"Kraken API error: {payload['error']}")
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
    defaults: dict[str, Any] = {
        "sent": set(), "cancelled": set(), "btc_direction": None,
        "btc_signal_timestamp_ms": 0, "btc_signal_time": "—",
        "btc_signal_rsi": float("nan"), "btc_source": None,
    }
    if not STATE_FILE.exists():
        return defaults
    try:
        raw = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            defaults.update({
                "sent": set(raw.get("sent", [])),
                "cancelled": set(raw.get("cancelled", [])),
                "btc_direction": raw.get("btc_direction"),
                "btc_signal_timestamp_ms": int(raw.get("btc_signal_timestamp_ms") or 0),
                "btc_signal_time": raw.get("btc_signal_time", "—"),
                "btc_signal_rsi": float(raw["btc_signal_rsi"]) if raw.get("btc_signal_rsi") is not None else float("nan"),
                "btc_source": raw.get("btc_source"),
            })
            return defaults
    except (OSError, ValueError, TypeError) as exc:
        print(f"WARNING: state.json could not be read; starting empty: {exc}")
    return defaults


def save_state(state: dict[str, Any]) -> None:
    payload = {
        "sent": sorted(state["sent"])[-20000:],
        "cancelled": sorted(state["cancelled"])[-20000:],
        "btc_direction": state.get("btc_direction"),
        "btc_signal_timestamp_ms": int(state.get("btc_signal_timestamp_ms") or 0),
        "btc_signal_time": state.get("btc_signal_time", "—"),
        "btc_signal_rsi": state.get("btc_signal_rsi") if math.isfinite(float(state.get("btc_signal_rsi", float("nan")))) else None,
        "btc_source": state.get("btc_source"),
        "scanner": "Kraken Futures WaveTrend OB-FVG RSI 1.2.0",
    }
    temp = STATE_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(STATE_FILE)


def fetch_symbols() -> list[str]:
    """Only currently tradable Kraken perpetual contracts; no spot/dated contracts."""
    payload = get_json(f"{REST_BASE}/instruments")
    instruments = payload.get("instruments", []) if isinstance(payload, dict) else []
    symbols: set[str] = set()
    for item in instruments:
        symbol = str(item.get("symbol") or item.get("tradeable") or "").upper()
        if not symbol.startswith(PERPETUAL_PREFIXES):
            continue
        if item.get("tradeable") is False or item.get("isExpired") is True:
            continue
        status = str(item.get("status", "")).lower()
        if status and status not in {"online", "trading", "active"}:
            continue
        symbols.add(symbol)
    if not symbols:
        raise RuntimeError("No Kraken perpetual instruments found; refusing to scan another exchange")
    return sorted(symbols)


def fetch_closed_candles(symbol: str, now_ms: int | None = None) -> pd.DataFrame:
    """Fetch current 30m trade candles, discard the forming bar, reject stale data.

    Kraken's documented charts endpoint accepts epoch-second `from`/`to` values,
    while candle `time` is epoch milliseconds. If a ranged response is stale,
    retry using the endpoint's latest-candles response (without range filters)
    before declaring the feed stale. Stale data is never scanned.
    """
    now_ms = int(now_ms if now_ms is not None else time.time() * 1000)
    url = f"{CHART_BASE}/trade/{quote(symbol, safe='_')}/{TIMEFRAME}"
    end_s = now_ms // 1000
    # 14 days supplies enough history for RSI/OB calculations while staying
    # inside the endpoint's practical candle-window limits.
    start_s = end_s - 14 * 24 * 60 * 60
    request_modes = [
        ("range", {"from": start_s, "to": end_s, "count": CANDLE_LIMIT}),
        ("latest", {"count": CANDLE_LIMIT}),
        ("default_latest", None),
    ]
    last_seen_ms: int | None = None
    last_staleness_ms: int | None = None

    for attempt in range(1, CHART_FETCH_ATTEMPTS + 1):
        for mode, params in request_modes:
            payload = get_json(url, params)
            rows = payload.get("candles", []) if isinstance(payload, dict) else []
            if not isinstance(rows, list) or not rows:
                print(f"DATA_EMPTY {symbol}: attempt={attempt}/{CHART_FETCH_ATTEMPTS} mode={mode}")
                continue

            normalized: list[tuple[int, float, float, float, float, float]] = []
            for row in rows:
                try:
                    stamp = int(row["time"] if isinstance(row, dict) else row[0])
                    if stamp < 10**12:
                        stamp *= 1000
                    # Kraken's candle time is the candle OPEN time. Keep only
                    # bars whose full 30-minute interval has elapsed.
                    if stamp + INTERVAL_MS > now_ms:
                        continue
                    if isinstance(row, dict):
                        opn, high, low, close = row["open"], row["high"], row["low"], row["close"]
                        volume = row.get("volume", 0)
                    else:
                        opn, high, low, close, volume = row[1:6]
                    normalized.append((stamp, float(opn), float(high), float(low), float(close), float(volume)))
                except (KeyError, TypeError, ValueError, IndexError):
                    continue

            if not normalized:
                continue
            frame = pd.DataFrame(normalized, columns=["timestamp", "open", "high", "low", "close", "volume"])
            frame = frame.drop_duplicates("timestamp").sort_values("timestamp")
            last_seen_ms = int(frame["timestamp"].iloc[-1])
            staleness_ms = now_ms - (last_seen_ms + INTERVAL_MS)
            last_staleness_ms = staleness_ms
            if staleness_ms <= MAX_CANDLE_STALENESS_MS:
                frame["timestamp"] = pd.to_datetime(frame.timestamp, unit="ms", utc=True)
                frame = frame.set_index("timestamp")
                if len(frame) < MIN_CANDLES:
                    print(f"DATA_INSUFFICIENT {symbol}: candles={len(frame)} minimum={MIN_CANDLES}")
                    return pd.DataFrame()
                print(
                    f"DATA_OK {symbol}: mode={mode} candles={len(frame)} last_closed="
                    f"{frame.index[-1].strftime('%Y-%m-%d %H:%M UTC')} "
                    f"staleness_min={max(0, staleness_ms) / 60000:.1f}"
                )
                return frame
            print(
                f"DATA_STALE {symbol}: attempt={attempt}/{CHART_FETCH_ATTEMPTS} mode={mode} "
                f"last_candle_ms={last_seen_ms} staleness_min={staleness_ms / 60000:.1f}"
            )
        if attempt < CHART_FETCH_ATTEMPTS:
            time.sleep(min(2 ** (attempt - 1), 4))

    if last_seen_ms is None:
        raise RuntimeError(f"NO_KRAKEN_CHART_DATA: {symbol} returned no usable closed candles")
    raise RuntimeError(
        f"STALE_KRAKEN_CHART_DATA: {symbol} latest closed candle is "
        f"{last_staleness_ms / 60000:.1f} minutes old after {CHART_FETCH_ATTEMPTS} attempts "
        f"(range + latest-candles fallback); refusing to scan stale candles"
    )


def _fmt_rsi(value: float) -> str:
    return f"{value:.1f}" if math.isfinite(value) else "N/A"


def _signal_key(symbol: str, candidate: dict[str, Any]) -> str:
    return f"{symbol}|{TIMEFRAME}|{candidate['timestamp_ms']}|{candidate['direction']}"


def _btc_context(btc_symbol: str, btc_analysis: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Initialize BTC bias from latest WT alone; later flips require OB/FVG confirmation."""
    candidates = sorted(btc_analysis["candidates"], key=lambda c: c["timestamp_ms"])
    direction = state.get("btc_direction")
    accepted_ts = int(state.get("btc_signal_timestamp_ms") or 0)
    source = state.get("btc_source")

    if direction not in {"BUY", "SELL"}:
        # First-ever direction is intentionally exempt from OB/FVG confirmation.
        if candidates:
            latest = candidates[-1]
            direction = latest["direction"]
            accepted_ts = int(latest["timestamp_ms"])
            source = "INITIAL_WAVETREND"
            state.update({
                "btc_direction": direction,
                "btc_signal_timestamp_ms": accepted_ts,
                "btc_signal_time": latest["timestamp"].strftime("%Y-%m-%d %H:%M UTC"),
                "btc_signal_rsi": latest["rsi"],
                "btc_source": source,
            })
            print(f"BTC_INITIAL_BIAS: {direction} from latest WaveTrend signal; OB/FVG confirmation bypassed")
    else:
        # Only newer WaveTrend events with matching OB/FVG confirmation may
        # advance the persisted BTC direction. Pending/rejected events do not.
        for candidate in candidates:
            ts = int(candidate["timestamp_ms"])
            if not candidate.get("recent_actionable"):
                continue
            if ts <= accepted_ts or candidate["status"] != "CONFIRMED":
                continue
            direction = candidate["direction"]
            accepted_ts = ts
            source = "OB_FVG_CONFIRMED"
            state.update({
                "btc_direction": direction,
                "btc_signal_timestamp_ms": accepted_ts,
                "btc_signal_time": candidate["timestamp"].strftime("%Y-%m-%d %H:%M UTC"),
                "btc_signal_rsi": candidate["rsi"],
                "btc_source": source,
            })
            print(f"BTC_BIAS_UPDATE: {direction} confirmed by OB/FVG; signal={state['btc_signal_time']}")

    signal_time = state.get("btc_signal_time", "—") if direction in {"BUY", "SELL"} else "—"
    signal_rsi = float(state.get("btc_signal_rsi", float("nan"))) if direction in {"BUY", "SELL"} else float("nan")
    return {
        "symbol": btc_symbol,
        "direction": direction if direction in {"BUY", "SELL"} else "NO_WAVETREND_SIGNAL",
        "signal_time": signal_time,
        "signal_rsi": signal_rsi,
        "current_rsi": btc_analysis["latest_rsi"],
        "source": source or "UNKNOWN",
    }


def evaluate_symbol(symbol: str, now_ms: int) -> dict[str, Any]:
    candles = fetch_closed_candles(symbol, now_ms)
    if candles.empty:
        return {"symbol": symbol, "status": "NO_DATA_OR_LT_60_CANDLES"}
    analysis = analyze_market(candles)
    return {"symbol": symbol, "status": "OK", "analysis": analysis}


def make_alert(item: dict[str, Any], btc: dict[str, Any]) -> str:
    buy = item["direction"] == "BUY"
    title = "🟢 BUY / LONG" if buy else "🔴 SELL / SHORT"
    priority = "اولویت اول — RSI هم‌جهت (خرید > ۵۰ / فروش < ۵۰)" if item["priority"] == 1 else "اولویت دوم — RSI خلاف جهت (خرید ≤ ۵۰ / فروش ≥ ۵۰)"
    btc_alignment = item.get("btc_alignment", "UNKNOWN")
    btc_alignment_fa = {"ALIGNED": "هم‌جهت با بیت‌کوین", "OPPOSITE": "خلاف جهت بیت‌کوین", "UNKNOWN": "جهت BTC نامشخص"}.get(btc_alignment, btc_alignment)
    match_kind = item.get("match_kind", "OB/FVG")
    return "\n".join([
        title,
        f"قرارداد: {item['symbol']} — Kraken Futures Perpetual",
        f"تایم‌فریم: {TIMEFRAME}",
        f"قیمت کندل سیگنال: {item['price']:.10g}",
        f"RSI(14) ارز هنگام تأیید: {_fmt_rsi(item['rsi'])}",
        f"RSI(14) فعلی ارز: {_fmt_rsi(item.get('current_rsi', float('nan')))}",
        f"جهت نسبت به BTC: {btc_alignment_fa}",
        f"اولویت RSI: {priority}",
        f"تأیید: برخورد به باکس هم‌رنگ {match_kind} در هفت کندل بسته اخیر؛ بدون برخورد باکس مخالف",
        f"زمان سیگنال Wave Trend: {item['signal_time']}",
        "",
        f"آخرین جهت BTC ({btc['symbol']}): {btc['direction']} — {'جهت اولیه WaveTrend بدون نیاز به OB/FVG' if btc.get('source') == 'INITIAL_WAVETREND' else 'تأییدشده با OB/FVG' if btc.get('source') == 'OB_FVG_CONFIRMED' else 'نامشخص'}",
        f"زمان سیگنال BTC: {btc['signal_time']}",
        f"RSI(14) بیت‌کوین هنگام آخرین سیگنال: {_fmt_rsi(btc['signal_rsi'])}",
        f"RSI(14) فعلی بیت‌کوین در آخرین کندل بسته: {_fmt_rsi(btc['current_rsi'])}",
    ])


def make_cancellation(item: dict[str, Any]) -> str:
    return "\n".join([
        "⚠️ لغو سیگنال قبلی",
        f"قرارداد: {item['symbol']} — Kraken Futures",
        f"سیگنال قبلی: {item['direction']} | {item['signal_time']}",
        "علت: در پنجره تأیید، قیمت با باکس مخالف برخورد کرد.",
        "طبق قانون اسکنر، وجود حتی یک برخورد مخالف باعث رد کامل سیگنال است.",
    ])


def _candidate_for_current_window(symbol: str, analysis: dict[str, Any], last_index: int) -> list[dict[str, Any]]:
    """Only newly relevant candidates (signal candle age 0..2) are actionable."""
    out = []
    candles = analysis["candles"]
    touches = analysis["touches"]
    for candidate in analysis["candidates"]:
        age = last_index - int(candidate["signal_index"])
        if age < 0 or age > 2:
            continue
        item = dict(candidate)
        item["symbol"] = symbol
        item["current_rsi"] = analysis["latest_rsi"]
        item["signal_time"] = candidate["timestamp"].strftime("%Y-%m-%d %H:%M UTC")
        # First matching zone class at the first same-direction touch.
        match_idx = candidate.get("first_match_index")
        if match_idx is not None:
            if item["direction"] == "BUY":
                ob = bool(touches["bull_ob_touch"].iloc[match_idx])
                fvg = bool(touches["bull_fvg_touch"].iloc[match_idx])
            else:
                ob = bool(touches["bear_ob_touch"].iloc[match_idx])
                fvg = bool(touches["bear_fvg_touch"].iloc[match_idx])
            item["match_kind"] = "Order Block و FVG" if ob and fvg else "Order Block" if ob else "FVG"
        else:
            item["match_kind"] = "OB/FVG"
        out.append(item)
    return out


def filter_btc_aligned_alerts(items: list[dict[str, Any]], btc_direction: str) -> tuple[list[dict[str, Any]], int]:
    """Keep only alerts aligned with persisted BTC bias (initial WT or later OB/FVG-confirmed)."""
    if btc_direction not in {"BUY", "SELL"}:
        return [], len(items)
    aligned = [item for item in items if item.get("direction") == btc_direction]
    for item in aligned:
        item["btc_alignment"] = "ALIGNED"
    return aligned, len(items) - len(aligned)


def main() -> None:
    now_ms = int(time.time() * 1000)
    print(f"Starting Kraken Futures scanner 1.2.0 | {TIMEFRAME} | Wave Trend + Order Blocks/FVG + RSI")
    symbols = fetch_symbols()
    btc_symbol = next((s for s in BTC_SYMBOLS if s in symbols), None)
    if btc_symbol is None:
        raise RuntimeError("No BTC perpetual contract found in Kraken instrument catalogue")

    btc_candles = fetch_closed_candles(btc_symbol, now_ms)
    if btc_candles.empty:
        raise RuntimeError("BTC 30m closed candles unavailable; cannot build BTC context")
    btc_analysis = analyze_market(btc_candles)
    state = load_state()
    btc = _btc_context(btc_symbol, btc_analysis, state)
    save_state(state)
    btc_diag = btc_analysis.get("diagnostics", {})
    btc_recent_candidates = [c for c in btc_analysis["candidates"] if c.get("recent_actionable")]
    print(
        f"DIAGNOSTIC BTC {btc_symbol}: WT_buy_total/recent3={btc_diag.get('wt_buy_total', 0)}/{btc_diag.get('wt_buy_recent_3', 0)} "
        f"WT_sell_total/recent3={btc_diag.get('wt_sell_total', 0)}/{btc_diag.get('wt_sell_recent_3', 0)} "
        f"pivot_low/high={btc_diag.get('pivot_low_confirmed_total', 0)}/{btc_diag.get('pivot_high_confirmed_total', 0)} "
        f"OB_created/deleted={btc_diag.get('ob_created_total', 0)}/{btc_diag.get('ob_deleted_total', 0)} "
        f"FVG_created/deleted={btc_diag.get('fvg_created_total', 0)}/{btc_diag.get('fvg_deleted_total', 0)} "
        f"active_OB(B/S)={btc_diag.get('active_bull_ob_latest', 0)}/{btc_diag.get('active_bear_ob_latest', 0)} "
        f"active_FVG(B/S)={btc_diag.get('active_bull_fvg_latest', 0)}/{btc_diag.get('active_bear_fvg_latest', 0)} "
        f"recent_WT_candidates={len(btc_recent_candidates)}"
    )
    for candidate in btc_recent_candidates:
        print(f"DIAGNOSTIC BTC {candidate['direction']} {candidate['timestamp']}: {candidate['status']} — {candidate['diagnostic_reason']}; same_touches={candidate['same_touch_count']}; opposite_touches={candidate['opposite_touch_count']}")
    print(f"BTC context: direction={btc['direction']} source={btc.get('source')} | signal_time={btc['signal_time']} | signal_RSI={_fmt_rsi(btc['signal_rsi'])} | current_RSI={_fmt_rsi(btc['current_rsi'])}")
    if btc["direction"] not in {"BUY", "SELL"}:
        print("DIAGNOSTIC BTC_FILTER: no BTC WaveTrend direction; all altcoin alerts will be blocked.")

    alt_symbols = [s for s in symbols if s not in BTC_SYMBOLS]
    print(f"Tradable perpetual contracts={len(symbols)} | BTC reference={btc_symbol} | other contracts={len(alt_symbols)}")

    records: list[dict[str, Any]] = []
    errors = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(evaluate_symbol, symbol, now_ms): symbol for symbol in alt_symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                records.append(future.result())
            except requests.HTTPError as exc:
                errors += 1
                status = exc.response.status_code if exc.response is not None else "unknown"
                records.append({"symbol": symbol, "status": f"HTTP_ERROR_{status}"})
                print(f"HTTP error {symbol}: {exc}")
                if status == 429:
                    raise
            except Exception as exc:
                errors += 1
                records.append({"symbol": symbol, "status": f"ENGINE_ERROR_{type(exc).__name__}"})
                print(f"Skipped {symbol}: {type(exc).__name__}: {exc}")

    new_alerts: list[dict[str, Any]] = []
    cancellations: list[dict[str, Any]] = []
    candidate_count = 0
    confirmed_before_btc = 0
    rejected_opposite_count = 0
    no_match_count = 0
    wt_recent_count = 0
    for record in records:
        if record.get("status") != "OK":
            continue
        symbol = record["symbol"]
        analysis = record["analysis"]
        last_index = len(analysis["candles"]) - 1
        candidates = _candidate_for_current_window(symbol, analysis, last_index)
        candidate_count += len(candidates)
        diag = analysis.get("diagnostics", {})
        print(
            f"DIAGNOSTIC {symbol}: WT_buy={diag.get('wt_buy_total', 0)} "
            f"WT_sell={diag.get('wt_sell_total', 0)} "
            f"pivot_low/high={diag.get('pivot_low_confirmed_total', 0)}/{diag.get('pivot_high_confirmed_total', 0)} "
            f"OB_created={diag.get('ob_created_total', 0)} FVG_created={diag.get('fvg_created_total', 0)} "
            f"OB_deleted={diag.get('ob_deleted_total', 0)} FVG_deleted={diag.get('fvg_deleted_total', 0)} "
            f"active_OB(B/S)={diag.get('active_bull_ob_latest', 0)}/{diag.get('active_bear_ob_latest', 0)} "
            f"active_FVG(B/S)={diag.get('active_bull_fvg_latest', 0)}/{diag.get('active_bear_fvg_latest', 0)} "
            f"recent_candidates={len(candidates)}"
        )
        wt_recent_count += len(candidates)
        for item in candidates:
            key = _signal_key(symbol, item)
            if item["status"] == "REJECTED_OPPOSITE_TOUCH":
                rejected_opposite_count += 1
                print(f"DIAGNOSTIC {symbol} {item['direction']} {item['signal_time']}: REJECTED — opposite OB/FVG touch ({item['opposite_touch_count']})")
                if key in state["sent"] and key not in state["cancelled"]:
                    item["dedupe_key"] = key
                    cancellations.append(item)
                continue
            if item["status"] == "CONFIRMED":
                confirmed_before_btc += 1
                print(f"DIAGNOSTIC {symbol} {item['direction']} {item['signal_time']}: OB/FVG confirmed by {item.get('match_kind', 'OB/FVG')}; same_touches={item['same_touch_count']}; opposite_touches={item['opposite_touch_count']}; RSI={_fmt_rsi(item['rsi'])}; priority={item['priority']}")
                if key not in state["sent"]:
                    item["dedupe_key"] = key
                    new_alerts.append(item)
            elif item["status"] == "PENDING_NO_MATCHING_TOUCH":
                no_match_count += 1
                print(f"DIAGNOSTIC {symbol} {item['direction']} {item['signal_time']}: no matching OB/FVG touch in latest seven closed candles")

    # BTC direction is the FINAL signal filter, not merely an alert sort key.
    # No initialized BTC WaveTrend direction => fail closed and emit no altcoin signal.
    pre_filter_alerts = list(new_alerts)
    new_alerts, btc_direction_rejected = filter_btc_aligned_alerts(new_alerts, btc["direction"])
    for item in pre_filter_alerts:
        if item["direction"] != btc["direction"] or btc["direction"] not in {"BUY", "SELL"}:
            why = "BTC direction unavailable" if btc["direction"] not in {"BUY", "SELL"} else f"opposite current BTC direction ({btc['direction']})"
            print(f"DIAGNOSTIC {item['symbol']} {item['direction']}: BLOCKED — {why}")
    # Only the surviving BTC-aligned signals are sorted by RSI priority.
    new_alerts.sort(key=lambda x: (x["priority"], x["timestamp_ms"], x["symbol"]))
    cancellations.sort(key=lambda x: (x["timestamp_ms"], x["symbol"]))
    print(
        f"DIAGNOSTIC SUMMARY: recent_WT_candidates={wt_recent_count} "
        f"OB_FVG_confirmed={confirmed_before_btc} opposite_box_rejected={rejected_opposite_count} "
        f"no_matching_box={no_match_count} BTC_direction_blocked={btc_direction_rejected} "
        f"final_BTC_aligned={len(new_alerts)}"
    )

    sent_count = 0
    for item in cancellations:
        try:
            telegram_send(make_cancellation(item))
            state["cancelled"].add(item["dedupe_key"])
            save_state(state)
            print(f"CANCELLED {item['symbol']} {item['direction']} {item['signal_time']} due to opposite box touch")
        except Exception as exc:
            print(f"Cancellation notice failed {item['symbol']}: {type(exc).__name__}: {exc}")

    for item in new_alerts:
        try:
            telegram_send(make_alert(item, btc))
            state["sent"].add(item["dedupe_key"])
            save_state(state)
            sent_count += 1
            print(f"ALERT priority={item['priority']} {item['symbol']} {item['direction']} RSI={_fmt_rsi(item['rsi'])} {item['signal_time']}")
        except Exception as exc:
            print(f"Telegram send failed {item['symbol']}: {type(exc).__name__}: {exc}")

    save_state(state)
    ok_count = sum(r.get("status") == "OK" for r in records)
    print(f"Finished: fetched={ok_count}/{len(alt_symbols)} candidates_in_window={candidate_count} alerts={sent_count} cancellations={len(cancellations)} errors={errors}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
