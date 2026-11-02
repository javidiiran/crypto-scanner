"""Offline smoke tests for the scanner calculation module."""
import unittest
import numpy as np
import pandas as pd

from indicator_engine import (
    evaluate_signals,
    reversal_probability_zones,
    smart_money_zones,
    squeeze_momentum,
    _candle_touches_zone,
    _zone_contains,
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
        self.assertTrue(set(result.signal.unique()).issubset({"", "BUY", "SELL"}))

    def test_smc_zone_confirmation_uses_wick_touch(self):
        zone = {"bottom": 100.0, "top": 105.0}
        # Close is outside, but the candle wick enters the zone.
        self.assertTrue(_candle_touches_zone(103.0, 108.0, zone))
        self.assertFalse(_zone_contains(108.0, zone))
        self.assertFalse(_candle_touches_zone(106.0, 108.0, zone))

    def test_reversal_close_confirmation_still_requires_close_inside(self):
        zone = {"bottom": 100.0, "top": 105.0}
        self.assertTrue(_zone_contains(102.5, zone))
        self.assertFalse(_zone_contains(99.9, zone))


if __name__ == "__main__":
    unittest.main(verbosity=2)
