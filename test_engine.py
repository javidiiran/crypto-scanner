"""Offline smoke tests for the scanner calculation module."""
import unittest
import numpy as np
import pandas as pd

from indicator_engine import (
    evaluate_signals,
    reversal_probability_zones,
    smart_money_zones,
    squeeze_momentum,
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
