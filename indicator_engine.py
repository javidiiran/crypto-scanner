"""Faithful Python port of the two supplied TradingView indicators.

1) Wave Trend With Signals [ChuckBanger], Pine v4
   - defaults: n1=10, n2=21, OB2=53, OS2=-53, Laguerre=true, gamma=0.02
   - only signal1 (buy cross) and signal2 (sell cross) are used; divergences are ignored.

2) Order Blocks Zones with Signals [Uncle_the_shooter], Pine v6
   - defaults preserved: pivot=10, wicks=true, size period=20,
     FVG multiplier=0.5, OB multiplier=0.1, FVG visible bars=10.
   - OB bullish/bearish = green/red. FVG bullish/bearish = blue/orange.
   - Zone touch is candle-range intersection (high/low), not close-in-zone.

The scanner confirmation window is exactly four candles before the WT signal
and two candles after it. The signal candle itself is not part of the touch
window. Any opposite-direction touch anywhere in that six-candle window
rejects the candidate, even if a matching touch also occurred.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WaveTrendSettings:
    channel_length: int = 10
    average_length: int = 21
    overbought_level_2: float = 53.0
    oversold_level_2: float = -53.0
    use_laguerre: bool = True
    gamma: float = 0.02


@dataclass(frozen=True)
class OrderBlockSettings:
    pivot_length: int = 10
    use_wicks: bool = True
    candle_size_period: int = 20
    fvg_size_multiplier: float = 0.5
    ob_size_multiplier: float = 0.1
    show_fvg: bool = True
    fvg_max_bars: int = 10
    max_boxes: int = 500
    max_age_bars: int = 500


def _validate_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close"}
    out = frame.copy()
    out.columns = [str(col).lower() for col in out.columns]
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")
    for col in ("open", "high", "low", "close"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def _pine_ema(values: pd.Series, length: int) -> pd.Series:
    # Pine ta.ema uses alpha=2/(length+1), recursively seeded from first value.
    return values.astype(float).ewm(span=length, adjust=False, min_periods=1).mean()


def _laguerre(values: pd.Series, gamma: float) -> pd.Series:
    src = values.to_numpy(dtype=float)
    n = len(src)
    l0 = np.zeros(n, dtype=float)
    l1 = np.zeros(n, dtype=float)
    l2 = np.zeros(n, dtype=float)
    l3 = np.zeros(n, dtype=float)
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        p = src[i]
        if not np.isfinite(p):
            # Pine's nz() is applied to prior recursive values, not to source.
            # A na source propagates na through this bar; state resumes next bar.
            continue
        p0 = l0[i - 1] if i else 0.0
        p1 = l1[i - 1] if i else 0.0
        p2 = l2[i - 1] if i else 0.0
        p3 = l3[i - 1] if i else 0.0
        l0[i] = (1.0 - gamma) * p + gamma * p0
        l1[i] = -gamma * l0[i] + p0 + gamma * p1
        l2[i] = -gamma * l1[i] + p1 + gamma * p2
        l3[i] = -gamma * l2[i] + p2 + gamma * p3
        out[i] = (l0[i] + 2.0 * l1[i] + 2.0 * l2[i] + l3[i]) / 6.0
    return pd.Series(out, index=values.index)


def wave_trend(frame: pd.DataFrame, settings: WaveTrendSettings = WaveTrendSettings()) -> pd.DataFrame:
    """Port of the supplied WTWS CB formulas and signal1/signal2 conditions."""
    d = _validate_ohlcv(frame)
    ap = (d.high + d.low + d.close) / 3.0
    esa = _pine_ema(ap, settings.channel_length)
    abs_diff = (ap - esa).abs()
    d_ = _pine_ema(abs_diff, settings.channel_length)
    denom = 0.015 * d_
    ci = (ap - esa) / denom.replace(0.0, np.nan)
    tci = _pine_ema(ci, settings.average_length)
    wt1 = _laguerre(tci, settings.gamma) if settings.use_laguerre else tci
    wt2 = wt1.rolling(4, min_periods=4).mean()

    cross_up = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    cross_down = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))
    buy = cross_up & (wt1 < settings.oversold_level_2)
    sell = cross_down & (wt1 > settings.overbought_level_2)

    return pd.DataFrame({
        "wt1": wt1,
        "wt2": wt2,
        "buy_signal": buy.fillna(False).astype(bool),
        "sell_signal": sell.fillna(False).astype(bool),
    }, index=d.index)


def rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    """TradingView-style RSI using Wilder RMA (SMA seed, then alpha=1/length)."""
    delta = close.astype(float).diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    def rma(series: pd.Series) -> pd.Series:
        arr = series.to_numpy(dtype=float)
        out = np.full(len(arr), np.nan, dtype=float)
        finite = np.flatnonzero(np.isfinite(arr))
        if len(finite) < length:
            return pd.Series(out, index=series.index)
        # Wilder seed: mean of the first `length` changes.
        seed_end = finite[length - 1]
        seed_start = finite[0]
        seed = arr[seed_start:seed_end + 1]
        if len(seed) != length or not np.isfinite(seed).all():
            return pd.Series(out, index=series.index)
        out[seed_end] = float(seed.mean())
        alpha = 1.0 / length
        for i in range(seed_end + 1, len(arr)):
            if np.isfinite(arr[i]):
                out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
        return pd.Series(out, index=series.index)

    avg_gain = rma(gain)
    avg_loss = rma(loss)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    result = 100.0 - (100.0 / (1.0 + rs))
    result = result.mask(avg_loss == 0.0, 100.0)
    result = result.mask(avg_gain == 0.0, 0.0)
    # Flat market (both averages zero) is conventionally 50 in TradingView RSI.
    result = result.mask((avg_gain == 0.0) & (avg_loss == 0.0), 50.0)
    return result


def _bar_touches(low: float, high: float, zone: dict[str, Any]) -> bool:
    return low <= float(zone["top"]) and high >= float(zone["bot"])


def order_block_fvg_touches(
    frame: pd.DataFrame,
    ob_settings: OrderBlockSettings = OrderBlockSettings(),
) -> pd.DataFrame:
    """Replay OB and FVG zone creation, lifetime, invalidation, and candle touches.

    Returned touch columns:
      bullish_touch: active green OB OR active blue FVG touched
      bearish_touch: active red OB OR active orange FVG touched
    Each candle is tested only against zones active on that candle. OB invalidation
    and FVG break-removal happen before that candle's touch is recorded, matching
    the supplied script's ordering.
    """
    d = _validate_ohlcv(frame)
    n = len(d)
    o = d.open.to_numpy(float)
    h = d.high.to_numpy(float)
    l = d.low.to_numpy(float)
    c = d.close.to_numpy(float)
    body = np.abs(c - o)
    avg_body = pd.Series(body).shift(1).rolling(ob_settings.candle_size_period).mean().to_numpy()

    ob_zones: list[dict[str, Any]] = []
    fvg_zones: list[dict[str, Any]] = []
    bull_ob_touch = np.zeros(n, dtype=bool)
    bear_ob_touch = np.zeros(n, dtype=bool)
    bull_fvg_touch = np.zeros(n, dtype=bool)
    bear_fvg_touch = np.zeros(n, dtype=bool)
    active_bull_ob_count = np.zeros(n, dtype=int)
    active_bear_ob_count = np.zeros(n, dtype=int)
    active_bull_fvg_count = np.zeros(n, dtype=int)
    active_bear_fvg_count = np.zeros(n, dtype=int)
    pivot_low_confirmed = np.zeros(n, dtype=bool)
    pivot_high_confirmed = np.zeros(n, dtype=bool)
    ob_created = np.zeros(n, dtype=int)
    fvg_created = np.zeros(n, dtype=int)
    ob_deleted = np.zeros(n, dtype=int)
    fvg_deleted = np.zeros(n, dtype=int)

    p = ob_settings.pivot_length
    for t in range(n):
        center = t - p
        if center >= p and t >= 2 * p and np.isfinite(avg_body[t]):
            left, right = center - p, center + p + 1
            # Pine ta.pivotlow/high(source, p, p) confirms the pivot p bars
            # after its candle. Chronological Python indices therefore map as:
            # Pine [p] -> center, [p+1] (older) -> center-1,
            # Pine [p-1] (newer) -> center+1.
            # ta.pivotlow/high resolves equal extrema toward the most recent
            # occurrence (the candidate at offset `p`). Thus the candidate may
            # tie older bars, but must be strictly more extreme than every
            # newer/right-side bar. Plain equality-to-window-extreme creates
            # repeated false pivots on flat/equal-price candles.
            older_lows = l[left:center]
            newer_lows = l[center + 1:right]
            older_highs = h[left:center]
            newer_highs = h[center + 1:right]
            pivot_low = bool(l[center] <= np.min(older_lows) and l[center] < np.min(newer_lows))
            pivot_high = bool(h[center] >= np.max(older_highs) and h[center] > np.max(newer_highs))
            pivot_low_confirmed[t] = pivot_low
            pivot_high_confirmed[t] = pivot_high

            def green(k: int) -> bool:
                return c[k] > o[k]

            if pivot_low:
                pivot_green = green(center)
                prev_green = green(center - 1)
                next_green = green(center + 1)
                source_idx: int | None = None
                is_bull = False
                if (not pivot_green) and next_green:
                    source_idx, is_bull = center, True
                elif pivot_green and (not prev_green):
                    source_idx, is_bull = center - 1, True
                elif pivot_green and prev_green and next_green:
                    source_idx, is_bull = center - 1, False
                if source_idx is not None:
                    top = h[source_idx] if ob_settings.use_wicks else max(o[source_idx], c[source_idx])
                    bot = l[source_idx] if ob_settings.use_wicks else min(o[source_idx], c[source_idx])
                    if (top - bot) >= avg_body[t] * ob_settings.ob_size_multiplier:
                        ob_zones.append({"top": top, "bot": bot, "created": source_idx, "bull": is_bull, "active": True})
                        ob_created[t] += 1

            if pivot_high:
                pivot_green = green(center)
                prev_green = green(center - 1)
                next_green = green(center + 1)
                source_idx = None
                is_bull = False
                if pivot_green and (not next_green):
                    source_idx, is_bull = center, False
                elif (not pivot_green) and prev_green:
                    source_idx, is_bull = center - 1, False
                elif (not pivot_green) and (not prev_green) and (not next_green):
                    source_idx, is_bull = center - 1, True
                if source_idx is not None:
                    top = h[source_idx] if ob_settings.use_wicks else max(o[source_idx], c[source_idx])
                    bot = l[source_idx] if ob_settings.use_wicks else min(o[source_idx], c[source_idx])
                    if (top - bot) >= avg_body[t] * ob_settings.ob_size_multiplier:
                        ob_zones.append({"top": top, "bot": bot, "created": source_idx, "bull": is_bull, "active": True})
                        ob_created[t] += 1

        # FVG creation: bullish blue (low > high[2]); bearish orange (high < low[2]).
        if ob_settings.show_fvg and t >= 2 and np.isfinite(avg_body[t]):
            if l[t] > h[t - 2]:
                top, bot = l[t], h[t - 2]
                if (top - bot) >= avg_body[t] * ob_settings.fvg_size_multiplier:
                    fvg_zones.append({"top": top, "bot": bot, "left": t - 2,
                                      "right": (t - 2) + ob_settings.fvg_max_bars + 1,
                                      "bull": True, "active": True})
                    fvg_created[t] += 1
            if h[t] < l[t - 2]:
                top, bot = l[t - 2], h[t]
                if (top - bot) >= avg_body[t] * ob_settings.fvg_size_multiplier:
                    fvg_zones.append({"top": top, "bot": bot, "left": t - 2,
                                      "right": (t - 2) + ob_settings.fvg_max_bars + 1,
                                      "bull": False, "active": True})
                    fvg_created[t] += 1
        # Pine array/box cap is 500. Oldest is removed first.
        if len(ob_zones) > ob_settings.max_boxes:
            removed = len(ob_zones) - ob_settings.max_boxes
            ob_deleted[t] += removed
            ob_zones = ob_zones[-ob_settings.max_boxes:]
        if len(fvg_zones) > ob_settings.max_boxes:
            removed = len(fvg_zones) - ob_settings.max_boxes
            fvg_deleted[t] += removed
            fvg_zones = fvg_zones[-ob_settings.max_boxes:]

        # Break removal is applied before touches on this candle, as in Pine.
        for z in ob_zones:
            if not z["active"]:
                continue
            if z["bull"] and c[t] < z["bot"]:
                z["active"] = False
                ob_deleted[t] += 1
            elif (not z["bull"]) and c[t] > z["top"]:
                z["active"] = False
                ob_deleted[t] += 1
        for z in fvg_zones:
            if not z["active"]:
                continue
            if z["bull"] and c[t] < z["bot"]:
                z["active"] = False
                fvg_deleted[t] += 1
            elif (not z["bull"]) and c[t] > z["top"]:
                z["active"] = False
                fvg_deleted[t] += 1

        active_bull_ob = [z for z in ob_zones if z["active"] and z["bull"]]
        active_bear_ob = [z for z in ob_zones if z["active"] and not z["bull"]]
        # The Pine source gives FVG boxes a finite visual right edge, but does
        # NOT deactivate/delete them there. They remain in fvg_active until a
        # close breaks the zone, the 500-bar cleanup runs, or the 500-box cap
        # removes the oldest box. The user's rule treats every still-active box
        # as valid, even after its drawn right edge.
        active_bull_fvg = [z for z in fvg_zones if z["active"] and z["bull"]]
        active_bear_fvg = [z for z in fvg_zones if z["active"] and not z["bull"]]

        bull_ob_touch[t] = any(_bar_touches(l[t], h[t], z) for z in active_bull_ob)
        bear_ob_touch[t] = any(_bar_touches(l[t], h[t], z) for z in active_bear_ob)
        bull_fvg_touch[t] = any(_bar_touches(l[t], h[t], z) for z in active_bull_fvg)
        bear_fvg_touch[t] = any(_bar_touches(l[t], h[t], z) for z in active_bear_fvg)
        active_bull_ob_count[t] = len(active_bull_ob)
        active_bear_ob_count[t] = len(active_bear_ob)
        active_bull_fvg_count[t] = len(active_bull_fvg)
        active_bear_fvg_count[t] = len(active_bear_fvg)

        # Pine runs the 500-bar cleanup at the very end of the bar, after its
        # signal/touch logic. Keep a zone available on the exact cleanup bar.
        kept_ob_zones = [z for z in ob_zones if (t - z["created"]) <= ob_settings.max_age_bars]
        kept_fvg_zones = [z for z in fvg_zones if (t - z["left"]) <= ob_settings.max_age_bars]
        ob_deleted[t] += len(ob_zones) - len(kept_ob_zones)
        fvg_deleted[t] += len(fvg_zones) - len(kept_fvg_zones)
        ob_zones = kept_ob_zones
        fvg_zones = kept_fvg_zones

    return pd.DataFrame({
        "bull_ob_touch": bull_ob_touch,
        "bear_ob_touch": bear_ob_touch,
        "bull_fvg_touch": bull_fvg_touch,
        "bear_fvg_touch": bear_fvg_touch,
        "bullish_touch": bull_ob_touch | bull_fvg_touch,
        "bearish_touch": bear_ob_touch | bear_fvg_touch,
        "active_bull_ob_count": active_bull_ob_count,
        "active_bear_ob_count": active_bear_ob_count,
        "active_bull_fvg_count": active_bull_fvg_count,
        "active_bear_fvg_count": active_bear_fvg_count,
        "pivot_low_confirmed": pivot_low_confirmed,
        "pivot_high_confirmed": pivot_high_confirmed,
        "ob_created": ob_created,
        "fvg_created": fvg_created,
        "ob_deleted": ob_deleted,
        "fvg_deleted": fvg_deleted,
    }, index=d.index)


def confirmation_window(signal_idx: int, length: int) -> tuple[int, int]:
    """The active scan window is the latest six CLOSED candles, inclusive."""
    return max(0, length - 6), max(0, length - 1)


def evaluate_candidates(
    frame: pd.DataFrame,
    wt: pd.DataFrame | None = None,
    touches: pd.DataFrame | None = None,
    rsi: pd.Series | None = None,
) -> list[dict[str, Any]]:
    """Evaluate WT events against the six most recent closed candles.

    Actionable candidates must have a WT event in one of the latest three
    closed candles. The confirmation scan uses the same latest-six-candle
    window for every such candidate (including the WT candle). Any opposite
    box touch rejects the candidate, even if a matching touch also exists.
    Older candidates are retained only as historical BTC-direction context,
    evaluated with the original four-prior/two-after window.
    """
    d = _validate_ohlcv(frame)
    wt = wt if wt is not None else wave_trend(d)
    touches = touches if touches is not None else order_block_fvg_touches(d)
    rsi = rsi if rsi is not None else rsi_wilder(d.close, 14)
    n = len(d)
    results: list[dict[str, Any]] = []
    global_start, global_end = confirmation_window(n - 1, n)

    for i in range(n):
        is_buy = bool(wt.buy_signal.iloc[i])
        is_sell = bool(wt.sell_signal.iloc[i])
        if not (is_buy or is_sell):
            continue
        direction = "BUY" if is_buy else "SELL"
        same_col = "bullish_touch" if is_buy else "bearish_touch"
        opp_col = "bearish_touch" if is_buy else "bullish_touch"
        recent = i >= max(0, n - 3)
        if recent:
            start, end = global_start, global_end
            idxs = list(range(start, end + 1))
        else:
            start = max(0, i - 4)
            end = min(n - 1, i + 2)
            idxs = list(range(start, min(i, n))) + list(range(i + 1, end + 1))
        same_indices = [j for j in idxs if bool(touches[same_col].iloc[j])]
        opp_indices = [j for j in idxs if bool(touches[opp_col].iloc[j])]
        if opp_indices:
            status = "REJECTED_OPPOSITE_TOUCH"
            reason = "OPPOSITE_BOX_TOUCH"
        elif same_indices:
            status = "CONFIRMED"
            reason = "MATCHING_OB_OR_FVG_TOUCH"
        elif recent:
            status = "PENDING_NO_MATCHING_TOUCH"
            reason = "NO_MATCHING_TOUCH_IN_LATEST_6"
        elif n - 1 >= i + 2:
            status = "EXPIRED_NO_MATCHING_TOUCH"
            reason = "NO_MATCHING_TOUCH_IN_CONFIRMATION_WINDOW"
        else:
            status = "PENDING_CONFIRMATION"
            reason = "WAITING_FOR_CLOSED_CANDLES"

        # RSI is priority only; it never creates, rejects, or reverses a signal.
        confirmation_idx = max(i, min(same_indices)) if same_indices else i
        signal_rsi = float(rsi.iloc[confirmation_idx]) if pd.notna(rsi.iloc[confirmation_idx]) else float("nan")
        aligned = (direction == "BUY" and np.isfinite(signal_rsi) and signal_rsi > 50.0) or (
            direction == "SELL" and np.isfinite(signal_rsi) and signal_rsi < 50.0
        )
        results.append({
            "signal_index": i,
            "timestamp": d.index[i],
            "timestamp_ms": int(pd.Timestamp(d.index[i]).timestamp() * 1000),
            "direction": direction,
            "status": status,
            "diagnostic_reason": reason,
            "recent_actionable": recent,
            "rsi": signal_rsi,
            "rsi_index": confirmation_idx,
            "confirmation_timestamp": d.index[confirmation_idx],
            "priority": 1 if aligned else 2,
            "rsi_aligned": bool(aligned),
            "first_match_index": min(same_indices) if same_indices else None,
            "first_opposite_index": min(opp_indices) if opp_indices else None,
            "same_touch_count": len(same_indices),
            "opposite_touch_count": len(opp_indices),
            "window_start_index": start,
            "window_end_index": end,
            "window_candles": len(idxs),
            "price": float(d.close.iloc[i]),
        })
    return results

def analyze_market(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute both indicator ports and RSI once for a symbol's closed candles."""
    d = _validate_ohlcv(frame)
    wt = wave_trend(d)
    zones = order_block_fvg_touches(d)
    rsi = rsi_wilder(d.close, 14)
    candidates = evaluate_candidates(d, wt=wt, touches=zones, rsi=rsi)
    return {
        "candles": d,
        "wave_trend": wt,
        "touches": zones,
        "rsi": rsi,
        "candidates": candidates,
        "latest_rsi": float(rsi.iloc[-1]) if len(rsi) and pd.notna(rsi.iloc[-1]) else float("nan"),
        "diagnostics": {
            "wt_buy_total": int(wt.buy_signal.sum()),
            "wt_sell_total": int(wt.sell_signal.sum()),
            "wt_buy_recent_3": int(wt.buy_signal.tail(3).sum()),
            "wt_sell_recent_3": int(wt.sell_signal.tail(3).sum()),
            "pivot_low_confirmed_total": int(zones.pivot_low_confirmed.sum()),
            "pivot_high_confirmed_total": int(zones.pivot_high_confirmed.sum()),
            "ob_created_total": int(zones.ob_created.sum()),
            "fvg_created_total": int(zones.fvg_created.sum()),
            "ob_deleted_total": int(zones.ob_deleted.sum()),
            "fvg_deleted_total": int(zones.fvg_deleted.sum()),
            "active_bull_ob_latest": int(zones.active_bull_ob_count.iloc[-1]),
            "active_bear_ob_latest": int(zones.active_bear_ob_count.iloc[-1]),
            "active_bull_fvg_latest": int(zones.active_bull_fvg_count.iloc[-1]),
            "active_bear_fvg_latest": int(zones.active_bear_fvg_count.iloc[-1]),
            "candidates_confirmed": sum(c["status"] == "CONFIRMED" for c in candidates),
            "candidates_rejected_opposite": sum(c["status"] == "REJECTED_OPPOSITE_TOUCH" for c in candidates),
        },
    }
