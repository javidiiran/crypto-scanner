"""Offline tests for SQZMOM trigger + SMC-only confirmation."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from indicator_engine import (
    evaluate_signals, reversal_probability_zones, smart_money_zones,
    squeeze_momentum, latest_confirmed_btc_pivot, _candle_touches_zone, _zone_contains,
)


class EngineSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        n = 700
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.006, n)))
        open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.0015, n))
        high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.008, n))
        low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.008, n))
        volume = rng.uniform(100, 10000, n)
        cls.df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=pd.date_range("2026-10-01", periods=n, freq="30min", tz="UTC"),
        )

    def test_sqzmom_returns_series(self):
        result = squeeze_momentum(self.df)
        self.assertEqual(len(result), len(self.df))
        self.assertTrue(result.notna().any())

    def test_reversal_zone_series_length(self):
        result = reversal_probability_zones(self.df)
        self.assertEqual(len(result), len(self.df))
        self.assertTrue(any(x is not None for x in result))

    def test_smc_all_three_price_zones_exist(self):
        result = smart_money_zones(self.df)
        self.assertEqual(len(result), len(self.df))
        kinds = {z["kind"] for bar in result for z in bar}
        self.assertTrue({"premium", "discount", "equilibrium"}.issubset(kinds))

    def test_full_engine_output(self):
        result = evaluate_signals(self.df)
        self.assertEqual(len(result), len(self.df))
        self.assertIn("signal", result.columns)
        self.assertIn("reversal_color", result.columns)
        self.assertTrue(set(result.signal.unique()).issubset({"", "BUY", "SELL"}))

    def test_smc_zone_confirmation_uses_wick_touch(self):
        zone = {"bottom": 100.0, "top": 105.0}
        self.assertTrue(_candle_touches_zone(103.0, 108.0, zone))
        self.assertFalse(_zone_contains(108.0, zone))
        self.assertFalse(_candle_touches_zone(106.0, 108.0, zone))

    def test_reversal_does_not_block_sqzmom_signal_when_no_reversal_box(self):
        df = pd.DataFrame({
            "open": [101., 101., 101.], "high": [102., 102., 102.],
            "low": [99., 99., 99.], "close": [100., 100., 100.],
            "volume": [10., 10., 10.],
        }, index=pd.date_range("2026-11-01", periods=3, freq="30min", tz="UTC"))
        hist = pd.Series([1., 3., 1.], index=df.index, name="sqzmom")
        zones = [[], [], [{"kind": "internal_bearish_ob", "direction": "SELL", "top": 101.5, "bottom": 100.5}]]
        with patch("indicator_engine.squeeze_momentum", return_value=hist), \
             patch("indicator_engine.reversal_probability_zones", return_value=[None, None, None]), \
             patch("indicator_engine.smart_money_zones", return_value=zones):
            result = evaluate_signals(df)
        self.assertEqual(result.iloc[-1].signal, "SELL")
        self.assertEqual(result.iloc[-1].reversal_color, "بدون باکس / خنثی")

    def test_opposite_reversal_color_does_not_block_signal(self):
        df = pd.DataFrame({
            "open": [101., 101., 101.], "high": [102., 102., 102.],
            "low": [99., 99., 99.], "close": [100., 100., 100.],
            "volume": [10., 10., 10.],
        }, index=pd.date_range("2026-11-01", periods=3, freq="30min", tz="UTC"))
        hist = pd.Series([1., 3., 1.], index=df.index, name="sqzmom")
        zones = [[], [], [{"kind": "internal_bearish_ob", "direction": "SELL", "top": 101.5, "bottom": 100.5}]]
        opposite_reversal = [None, None, {"direction": "BUY", "top": 98., "bottom": 95.}]
        with patch("indicator_engine.squeeze_momentum", return_value=hist), \
             patch("indicator_engine.reversal_probability_zones", return_value=opposite_reversal), \
             patch("indicator_engine.smart_money_zones", return_value=zones):
            result = evaluate_signals(df)
        self.assertEqual(result.iloc[-1].signal, "SELL")
        self.assertEqual(result.iloc[-1].reversal_color, "سبز")

    def test_btc_peak_requires_two_closed_bars_after_peak_without_strength_multiplier(self):
        hist = pd.Series([0.2, 0.5, 1.0, 0.8, 0.6],
                         index=pd.date_range("2026-11-01", periods=5, freq="30min", tz="UTC"))
        event = latest_confirmed_btc_pivot(hist)
        self.assertIsNotNone(event)
        self.assertEqual(event["signal"], "SELL")
        self.assertEqual(event["pivot_position"], 2)

    def test_btc_trough_requires_two_closed_bars_after_trough_without_strength_multiplier(self):
        hist = pd.Series([-0.2, -0.5, -1.0, -0.8, -0.6],
                         index=pd.date_range("2026-11-01", periods=5, freq="30min", tz="UTC"))
        event = latest_confirmed_btc_pivot(hist)
        self.assertIsNotNone(event)
        self.assertEqual(event["signal"], "BUY")
        self.assertEqual(event["pivot_position"], 2)

    def test_btc_pivot_without_two_confirmation_bars_is_not_emitted(self):
        hist = pd.Series([0.2, 0.5, 1.0, 0.8],
                         index=pd.date_range("2026-11-01", periods=4, freq="30min", tz="UTC"))
        self.assertIsNone(latest_confirmed_btc_pivot(hist))


if __name__ == "__main__":
    unittest.main(verbosity=2)
