"""AB engine for Kraken Futures Scanner 2.
CNV 3-candle signal + informational RSI(14) + Pine-equivalent OB/FVG confirmation.
Only closed 1h candles are supplied by scanner.py.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def validate_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    d.columns = [str(c).lower() for c in d.columns]
    required = {'open', 'high', 'low', 'close', 'volume'}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    for c in required:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.dropna(subset=list(required))
    d = d[~d.index.duplicated(keep='last')].sort_index()
    return d


def rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.astype(float).diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)

    def rma(x: pd.Series) -> pd.Series:
        a = x.to_numpy(float)
        out = np.full(len(a), np.nan)
        if len(a) <= length:
            return pd.Series(out, index=x.index)
        seed = a[1:length + 1]
        if not np.isfinite(seed).all():
            return pd.Series(out, index=x.index)
        out[length] = seed.mean()
        for k in range(length + 1, len(a)):
            if np.isfinite(a[k]) and np.isfinite(out[k - 1]):
                out[k] = (out[k - 1] * (length - 1) + a[k]) / length
        return pd.Series(out, index=x.index)

    avg_gain, avg_loss = rma(gain), rma(loss)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    value = 100 - 100 / (1 + rs)
    value = value.mask((avg_loss == 0) & (avg_gain > 0), 100)
    value = value.mask((avg_gain == 0) & (avg_loss > 0), 0)
    value = value.mask((avg_gain == 0) & (avg_loss == 0), 50)
    return value


def cnv(frame: pd.DataFrame) -> pd.DataFrame:
    d = validate_ohlcv(frame)
    change = d.close.diff()
    signed_volume = pd.Series(
        np.where(change > 0, d.volume, np.where(change < 0, -d.volume, 0.0)),
        index=d.index,
    )
    cumulative = signed_volume.cumsum()
    baseline = cumulative.rolling(20, min_periods=20).mean()
    delta = cumulative - baseline
    blue = delta > 0
    return pd.DataFrame({
        'cnv': cumulative, 'cnv_sma20': baseline, 'cnv_tb': delta,
        'blue': blue.fillna(False).astype(bool),
        'pink': (~blue).astype(bool),
    }, index=d.index)


def _pivot_at(values: np.ndarray, center: int, length: int, low_pivot: bool) -> bool:
    left, right = center - length, center + length
    if left < 0 or right >= len(values):
        return False
    window = values[left:right + 1]
    target = np.nanmin(window) if low_pivot else np.nanmax(window)
    # Pine pivots select the pivot at the center; requiring a unique extremum
    # avoids fabricating pivots on flat/equal-price plateaus.
    return bool(np.isfinite(target) and values[center] == target and np.count_nonzero(window == target) == 1)


def ob_fvg_confirmations(frame: pd.DataFrame, pivot_length: int = 10,
                         use_wicks: bool = True, candle_size_period: int = 20,
                         fvg_size_multiplier: float = 0.5,
                         ob_size_multiplier: float = 0.1,
                         max_age_bars: int = 500) -> list[dict | None]:
    """Replay the supplied Pine OB/FVG lifecycle and return latest-zone status per bar.

    OB/FVG colors are direction-equivalent: bullish OB or bullish FVG = BUY;
    bearish OB or bearish FVG = SELL. A zone is removed on the same close-break
    conditions as the supplied Pine source, or after 500 bars of age.
    """
    d = validate_ohlcv(frame)
    o, h, l, c = (d[x].to_numpy(float) for x in ('open', 'high', 'low', 'close'))
    n = len(d)
    green = c > o
    zones: list[dict] = []
    latest_by_bar: list[dict | None] = [None] * n
    serial = 0

    for i in range(n):
        # Pine's avg_body uses abs(close[1]-open[1]) over the prior N bars.
        avg_body = float(np.mean(np.abs(c[i-candle_size_period:i] - o[i-candle_size_period:i]))) if i >= candle_size_period else np.nan

        # Pivot-low confirmation: pivot center is pivot_length bars behind current bar.
        center = i - pivot_length
        if center >= pivot_length and np.isfinite(avg_body):
            if _pivot_at(l, center, pivot_length, True):
                idx = pivot_length
                pivot_green, prev_green, next_green = green[center], green[center - 1], green[center + 1]
                source = None
                is_bull = False
                if not pivot_green and next_green:
                    source, is_bull = center, True
                elif pivot_green and not prev_green:
                    source, is_bull = center + 1, True
                elif pivot_green and prev_green and next_green:
                    source, is_bull = center + 1, False
                if source is not None:
                    top = h[source] if use_wicks else max(o[source], c[source])
                    bot = l[source] if use_wicks else min(o[source], c[source])
                    if top - bot >= avg_body * ob_size_multiplier:
                        serial += 1
                        zones.append({'kind': 'OB', 'direction': 'BUY' if is_bull else 'SELL',
                                      'top': float(top), 'bot': float(bot), 'created': i,
                                      'source': source, 'serial': serial, 'active': True})

        # Pivot-high confirmation.
        if center >= pivot_length and np.isfinite(avg_body):
            if _pivot_at(h, center, pivot_length, False):
                pivot_green, prev_green, next_green = green[center], green[center - 1], green[center + 1]
                source = None
                is_bull = False
                if pivot_green and not next_green:
                    source, is_bull = center, False
                elif not pivot_green and prev_green:
                    source, is_bull = center + 1, False
                elif not pivot_green and not prev_green and not next_green:
                    source, is_bull = center + 1, True
                if source is not None:
                    top = h[source] if use_wicks else max(o[source], c[source])
                    bot = l[source] if use_wicks else min(o[source], c[source])
                    if top - bot >= avg_body * ob_size_multiplier:
                        serial += 1
                        zones.append({'kind': 'OB', 'direction': 'BUY' if is_bull else 'SELL',
                                      'top': float(top), 'bot': float(bot), 'created': i,
                                      'source': source, 'serial': serial, 'active': True})

        # FVG creation (same 3-candle gap conditions as supplied Pine).
        if i >= 2 and np.isfinite(avg_body):
            if l[i] > h[i - 2] and l[i] - h[i - 2] >= avg_body * fvg_size_multiplier:
                serial += 1
                zones.append({'kind': 'FVG', 'direction': 'BUY', 'top': float(l[i]),
                              'bot': float(h[i - 2]), 'created': i, 'source': i - 2,
                              'serial': serial, 'active': True})
            if h[i] < l[i - 2] and l[i - 2] - h[i] >= avg_body * fvg_size_multiplier:
                serial += 1
                zones.append({'kind': 'FVG', 'direction': 'SELL', 'top': float(l[i - 2]),
                              'bot': float(h[i]), 'created': i, 'source': i - 2,
                              'serial': serial, 'active': True})

        # Remove invalid zones on close-break; then apply Pine's 500-bar cleanup.
        for z in zones:
            if not z['active']:
                continue
            broken = (z['direction'] == 'BUY' and c[i] < z['bot']) or \
                     (z['direction'] == 'SELL' and c[i] > z['top'])
            too_old = i - z['source'] > max_age_bars
            if broken or too_old:
                z['active'] = False

        active = [z for z in zones if z['active']]
        if not active:
            continue
        latest = max(active, key=lambda z: (z['created'], z['serial']))
        # Any overlap of the candle's full high-low range counts as a touch.
        first = max(0, i - 4, int(latest['created']))
        touched = any(h[k] >= latest['bot'] and l[k] <= latest['top'] for k in range(first, i + 1))
        latest_by_bar[i] = {
            'kind': latest['kind'], 'direction': latest['direction'],
            'top': latest['top'], 'bot': latest['bot'], 'created_index': latest['created'],
            'age_bars': i - latest['created'], 'touched_last_5': bool(touched),
            'confirmed': bool(touched),
        }
    return latest_by_bar


def analyze_market(frame: pd.DataFrame) -> dict:
    d = validate_ohlcv(frame)
    colors = cnv(d)
    rsi = rsi_wilder(d.close, 14)
    confirmations = ob_fvg_confirmations(d)
    signals = []

    # RSI is informational only: it is deliberately absent from all conditions.
    for i in range(2, len(d)):
        pattern = [bool(colors.blue.iloc[j]) for j in (i - 2, i - 1, i)]
        if pattern == [False, True, True]:
            direction = 'BUY'
        elif pattern == [True, False, False]:
            direction = 'SELL'
        else:
            continue

        zone = confirmations[i]
        if zone is None or zone['direction'] != direction or not zone['touched_last_5']:
            continue
        stamp = d.index[i]
        signals.append({
            'direction': direction, 'signal_index': i, 'timestamp': stamp,
            'timestamp_ms': int(pd.Timestamp(stamp).timestamp() * 1000),
            'signal_time': pd.Timestamp(stamp).strftime('%Y-%m-%d %H:%M UTC'),
            'price': float(d.close.iloc[i]),
            'rsi': float(rsi.iloc[i]) if pd.notna(rsi.iloc[i]) else float('nan'),
            'cnv_colors': ['BLUE' if x else 'PINK' for x in pattern],
            'confirmation_zone': dict(zone),
        })

    return {
        'candles': d, 'cnv': colors, 'rsi': rsi, 'signals': signals,
        'latest_rsi': float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else float('nan'),
        'latest_cnv_color': 'BLUE' if bool(colors.blue.iloc[-1]) else 'PINK',
        'confirmations': confirmations,
    }
