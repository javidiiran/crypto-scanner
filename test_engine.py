"""Regression tests for three-closed-bar SQZMOM + original colored OB logic."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from indicator_engine import (
    Settings, evaluate_signals, latest_confirmed_sqzmom_pivot,
    order_block_zones, squeeze_momentum,
)


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        n = 800
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.006, n)))
        open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.0015, n))
        high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.008, n))
        low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.008, n))
        cls.df = pd.DataFrame({"open": open_, "high": high, "low": low,
                               "close": close, "volume": rng.uniform(100, 10000, n)},
                              index=pd.date_range("2026-10-01", periods=n, freq="30min", tz="UTC"))

    def _patched_engine(self, hist, bull, bear):
        n = len(hist)
        ob = pd.DataFrame({"bull_ob_inside": bull, "bear_ob_inside": bear,
                           "bull_ob_count": np.array(bull, dtype=int),
                           "bear_ob_count": np.array(bear, dtype=int)}, index=self.df.index[:n])
        sqz = pd.DataFrame({"sqzmom": hist}, index=self.df.index[:n])
        with patch("indicator_engine.squeeze_momentum", return_value=sqz), \
             patch("indicator_engine.order_block_zones", return_value=ob):
            return evaluate_signals(self.df.iloc[:n])

    def test_sell_peak_requires_three_positive_closed_bars_and_red_ob(self):
        result = self._patched_engine([0.1, 0.3, 0.2], [False]*3, [False, True, False])
        self.assertEqual(result.iloc[-1].signal, "SELL")
        self.assertTrue(result.iloc[-1].sell_ob_confirm)

    def test_buy_trough_requires_three_negative_closed_bars_and_green_ob(self):
        result = self._patched_engine([-0.1, -0.4, -0.2], [False, True, False], [False]*3)
        self.assertEqual(result.iloc[-1].signal, "BUY")

    def test_signal_rejected_when_all_three_bars_outside_colored_ob(self):
        result = self._patched_engine([0.1, 0.3, 0.2], [False]*3, [False]*3)
        self.assertEqual(result.iloc[-1].signal, "")

    def test_ob_direction_must_match_pivot_direction(self):
        result = self._patched_engine([0.1, 0.3, 0.2], [True, False, False], [False]*3)
        self.assertEqual(result.iloc[-1].signal, "")

    def test_latest_closed_bar_waits_for_third_candle(self):
        result = self._patched_engine([0.1, 0.3], [False, True], [False, False])
        self.assertEqual(result.iloc[-1].signal, "")

    def test_no_simultaneous_buy_and_sell(self):
        result = self._patched_engine([-0.1, -0.3, -0.2], [True]*3, [True]*3)
        self.assertFalse(bool(result.iloc[-1].BUY and result.iloc[-1].SELL))

    def test_btc_latest_peak_sets_sell_bias_and_trough_sets_buy_bias(self):
        hist = np.array([np.nan, 0.1, 0.4, 0.2, -0.1, -0.5, -0.2])
        fake = pd.DataFrame({"sqzmom": hist}, index=self.df.index[:len(hist)])
        with patch("indicator_engine.squeeze_momentum", return_value=fake):
            pivot = latest_confirmed_sqzmom_pivot(self.df.iloc[:len(hist)])
        self.assertEqual(pivot["bias"], "BUY")
        self.assertEqual(pivot["confirm_index"], 6)

    def test_btc_bias_persists_until_opposite_confirmed_pivot(self):
        hist = np.array([np.nan, 0.1, 0.4, 0.2, 0.15, 0.1, -0.1])
        fake = pd.DataFrame({"sqzmom": hist}, index=self.df.index[:len(hist)])
        with patch("indicator_engine.squeeze_momentum", return_value=fake):
            pivot = latest_confirmed_sqzmom_pivot(self.df.iloc[:len(hist)])
        self.assertEqual(pivot["bias"], "SELL")

    def test_real_indicator_functions_run(self):
        sqz = squeeze_momentum(self.df)
        ob = order_block_zones(self.df)
        self.assertEqual(len(sqz), len(self.df))
        self.assertEqual(len(ob), len(self.df))
        self.assertTrue({"bull_ob_inside", "bear_ob_inside"}.issubset(ob.columns))

    def test_full_engine_has_no_dual_direction(self):
        result = evaluate_signals(self.df)
        self.assertFalse((result.BUY & result.SELL).any())
        self.assertTrue(set(result.signal.unique()).issubset({"", "BUY", "SELL"}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
