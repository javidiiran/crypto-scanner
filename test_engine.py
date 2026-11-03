import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from scanner import percent_change_24h, sort_signals_by_priority
from ob_sqzmom_engine import (
    ZoneSettings, analyze_market, build_zones, confirm_with_latest_zone,
    rsi_wilder,
)


class EngineTests(unittest.TestCase):
    def frame(self, n=60):
        idx = pd.date_range('2026-01-01', periods=n, freq='h', tz='UTC')
        close = np.linspace(100, 130, n)
        return pd.DataFrame({
            'open': close - 0.2,
            'high': close + 1.0,
            'low': close - 1.0,
            'close': close,
            'volume': np.full(n, 100.0),
        }, index=idx)

    def test_rsi_range_and_wilder_uptrend(self):
        x = pd.Series(np.arange(50, dtype=float))
        r = rsi_wilder(x, 14)
        self.assertAlmostEqual(float(r.iloc[-1]), 100.0)

    def test_cnv_patterns_are_not_filtered_by_rsi(self):
        d = self.frame(6)
        blue = pd.Series([False, False, False, True, True, True], index=d.index)
        cnv_frame = pd.DataFrame({'blue': blue}, index=d.index)
        rsi = pd.Series([80.0, 80.0, 80.0, 80.0, 80.0, 80.0], index=d.index)
        with patch('ob_sqzmom_engine.cnv', return_value=cnv_frame), \
             patch('ob_sqzmom_engine.rsi_wilder', return_value=rsi):
            result = analyze_market(d)
        self.assertEqual(result['signals'][-1]['direction'], 'BUY')
        self.assertEqual(result['signals'][-1]['rsi'], 80.0)

    def test_buy_accepts_fvg_blue_when_wick_touches(self):
        d = self.frame(8)
        # Candle low enters the zone while its close remains above the zone.
        d.iloc[-1, d.columns.get_loc('low')] = 109.0
        zone = {'kind': 'FVG', 'direction': 'BUY', 'top': 110.0,
                'bottom': 108.0, 'origin_index': 5, 'created_index': 6,
                'created_order': 4, 'active': True}
        with patch('ob_sqzmom_engine.build_zones', return_value=[zone]):
            result = confirm_with_latest_zone(d, 'BUY')
        self.assertTrue(result['confirmed'])
        self.assertEqual(result['zone']['kind'], 'FVG')

    def test_sell_accepts_bearish_fvg_on_any_wick_touch(self):
        d = self.frame(8)
        d.iloc[-2, d.columns.get_loc('low')] = 119.0
        d.iloc[-2, d.columns.get_loc('high')] = 120.0
        zone = {'kind': 'FVG', 'direction': 'SELL', 'top': 120.5,
                'bottom': 118.0, 'origin_index': 5, 'created_index': 6,
                'created_order': 5, 'active': True}
        with patch('ob_sqzmom_engine.build_zones', return_value=[zone]):
            result = confirm_with_latest_zone(d, 'SELL')
        self.assertTrue(result['confirmed'])

    def test_only_latest_active_zone_is_checked(self):
        d = self.frame(8)
        older_buy = {'kind': 'OB', 'direction': 'BUY', 'top': 110.0,
                     'bottom': 108.0, 'origin_index': 4, 'created_index': 5,
                     'created_order': 1, 'active': True}
        latest_sell = {'kind': 'FVG', 'direction': 'SELL', 'top': 130.0,
                       'bottom': 128.0, 'origin_index': 6, 'created_index': 7,
                       'created_order': 2, 'active': True}
        with patch('ob_sqzmom_engine.build_zones', return_value=[older_buy, latest_sell]):
            result = confirm_with_latest_zone(d, 'BUY')
        self.assertFalse(result['confirmed'])
        self.assertEqual(result['reason'], 'LATEST_ZONE_OPPOSITE')

    def test_close_need_not_be_inside_zone(self):
        d = self.frame(8)
        d.iloc[-1, d.columns.get_loc('low')] = 109.0
        d.iloc[-1, d.columns.get_loc('close')] = 115.0
        zone = {'kind': 'OB', 'direction': 'BUY', 'top': 110.0,
                'bottom': 108.0, 'origin_index': 4, 'created_index': 5,
                'created_order': 3, 'active': True}
        with patch('ob_sqzmom_engine.build_zones', return_value=[zone]):
            result = confirm_with_latest_zone(d, 'BUY')
        self.assertTrue(result['confirmed'])

    def test_empty_active_zone_rejects(self):
        with patch('ob_sqzmom_engine.build_zones', return_value=[]):
            result = confirm_with_latest_zone(self.frame(), 'BUY')
        self.assertFalse(result['confirmed'])
        self.assertEqual(result['reason'], 'NO_ACTIVE_ZONE')


    def test_sell_pattern_is_not_filtered_by_low_rsi(self):
        d = self.frame(6)
        blue = pd.Series([True, True, True, False, False, False], index=d.index)
        cnv_frame = pd.DataFrame({'blue': blue}, index=d.index)
        rsi = pd.Series([20.0] * 6, index=d.index)
        with patch('ob_sqzmom_engine.cnv', return_value=cnv_frame), \
             patch('ob_sqzmom_engine.rsi_wilder', return_value=rsi):
            result = analyze_market(d)
        self.assertEqual(result['signals'][-1]['direction'], 'SELL')
        self.assertEqual(result['signals'][-1]['rsi'], 20.0)

    def test_fvg_right_edge_is_respected(self):
        d = self.frame(12)
        zone = {'kind': 'FVG', 'direction': 'BUY', 'top': 120.0,
                'bottom': 118.0, 'origin_index': 0, 'created_index': 1,
                'created_order': 1, 'right_index': 3, 'active': True}
        with patch('ob_sqzmom_engine.build_zones', return_value=[zone]):
            result = confirm_with_latest_zone(d, 'BUY')
        self.assertFalse(result['confirmed'])
        self.assertEqual(result['reason'], 'NO_TOUCH_LAST_5')

    def test_only_five_latest_closed_candles_are_used(self):
        d = self.frame(10)
        d.iloc[2, d.columns.get_loc('low')] = 109.0
        zone = {'kind': 'OB', 'direction': 'BUY', 'top': 110.0,
                'bottom': 108.0, 'origin_index': 0, 'created_index': 1,
                'created_order': 1, 'active': True}
        with patch('ob_sqzmom_engine.build_zones', return_value=[zone]):
            result = confirm_with_latest_zone(d, 'BUY')
        self.assertFalse(result['confirmed'])

    def test_zone_replay_returns_only_active_boxes(self):
        result = build_zones(self.frame(), ZoneSettings())
        self.assertIsInstance(result, list)
        self.assertTrue(all(z['active'] for z in result))
        self.assertTrue(all(z['kind'] in ('OB', 'FVG') for z in result))

    def test_latest_two_cnv_colors_are_exposed(self):
        d = self.frame(6)
        blue = pd.Series([False, False, False, True, False, True], index=d.index)
        with patch('ob_sqzmom_engine.cnv', return_value=pd.DataFrame({'blue': blue}, index=d.index)):
            result = analyze_market(d)
        self.assertEqual(result['last_two_cnv_colors'], ['PINK', 'BLUE'])

    def test_24h_change_uses_24_hour_close_distance(self):
        d = self.frame(30)
        d.iloc[-25, d.columns.get_loc('close')] = 100.0
        d.iloc[-1, d.columns.get_loc('close')] = 110.0
        self.assertAlmostEqual(percent_change_24h(d), 10.0)

    def test_priority_buys_then_largest_buy_move_then_largest_sell_drop(self):
        items = [
            {'symbol': 'SLOWSELL', 'direction': 'SELL', 'change_24h_pct': -2.0},
            {'symbol': 'SMALLBUY', 'direction': 'BUY', 'change_24h_pct': 3.0},
            {'symbol': 'BIGSELL', 'direction': 'SELL', 'change_24h_pct': -15.0},
            {'symbol': 'BIGBUY', 'direction': 'BUY', 'change_24h_pct': 12.0},
        ]
        ordered = sort_signals_by_priority(items)
        self.assertEqual([x['symbol'] for x in ordered], ['BIGBUY', 'SMALLBUY', 'BIGSELL', 'SLOWSELL'])


if __name__ == '__main__':
    unittest.main()
