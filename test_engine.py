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

    def _patched_engine(self, hist, bull_touch, bear_touch, bull_inside=None, bear_inside=None):
        n = len(hist)
        bull_inside = bull_inside if bull_inside is not None else [False] * n
        bear_inside = bear_inside if bear_inside is not None else [False] * n
        ob = pd.DataFrame({"bull_ob_inside": bull_inside, "bear_ob_inside": bear_inside,
                           "bull_ob_touch": [False] * n, "bear_ob_touch": [False] * n,
                           "bull_ob_touch_5": bull_touch, "bear_ob_touch_5": bear_touch,
                           "bull_ob_count": np.array(bull_inside, dtype=int),
                           "bear_ob_count": np.array(bear_inside, dtype=int)}, index=self.df.index[:n])
        sqz = pd.DataFrame({"sqzmom": hist}, index=self.df.index[:n])
        with patch("indicator_engine.squeeze_momentum", return_value=sqz), \
             patch("indicator_engine.order_block_zones", return_value=ob):
            return evaluate_signals(self.df.iloc[:n])

    def test_sell_peak_and_red_ob_touch_in_latest_five_candles(self):
        result = self._patched_engine(
            [0.0, 0.1, 0.2, 0.4, 0.3],
            [False] * 5, [False, False, False, False, True]
        )
        self.assertEqual(result.iloc[-1].signal, "SELL")
        self.assertTrue(result.iloc[-1].sell_ob_confirm)

    def test_buy_trough_and_green_ob_touch_in_latest_five_candles(self):
        result = self._patched_engine(
            [0.0, -0.1, -0.2, -0.4, -0.3],
            [False, False, False, False, True], [False] * 5
        )
        self.assertEqual(result.iloc[-1].signal, "BUY")

    def test_close_inside_ob_is_not_required_when_touch_is_confirmed(self):
        result = self._patched_engine(
            [0.0, 0.1, 0.2, 0.4, 0.3],
            [False] * 5, [False, False, False, False, True],
            bear_inside=[False] * 5
        )
        self.assertTrue(result.iloc[-1].sell_ob_confirm)
        self.assertEqual(result.iloc[-1].signal, "SELL")

    def test_no_matching_ob_touch_in_five_candle_window_rejects(self):
        result = self._patched_engine(
            [0.0, 0.1, 0.2, 0.4, 0.3],
            [False] * 5, [False] * 5
        )
        self.assertEqual(result.iloc[-1].signal, "")

    def test_wrong_color_ob_touch_does_not_confirm(self):
        result = self._patched_engine(
            [0.0, 0.1, 0.2, 0.4, 0.3],
            [True, False, False, False, False], [False] * 5
        )
        self.assertEqual(result.iloc[-1].signal, "")

    def test_latest_five_touch_window_not_limited_to_pivot_three_bars(self):
        # Touch on the oldest of the five recent candles still confirms the pivot.
        result = self._patched_engine(
            [0.0, 0.1, 0.2, 0.4, 0.3],
            [False] * 5, [False, False, False, False, True]
        )
        self.assertTrue(result.iloc[-1].SELL)

    def test_latest_closed_bar_waits_for_third_candle(self):
        result = self._patched_engine([0.1, 0.3], [False, True], [False, False])
        self.assertEqual(result.iloc[-1].signal, "")

    def test_no_simultaneous_buy_and_sell(self):
        result = self._patched_engine([-0.1, -0.3, -0.2, -0.1, -0.05], [True]*5, [True]*5)
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
        self.assertTrue({"bull_ob_inside", "bear_ob_inside", "bull_ob_touch", "bear_ob_touch", "bull_ob_touch_5", "bear_ob_touch_5"}.issubset(ob.columns))

    def test_full_engine_has_no_dual_direction(self):
        result = evaluate_signals(self.df)
        self.assertFalse((result.BUY & result.SELL).any())
        self.assertTrue(set(result.signal.unique()).issubset({"", "BUY", "SELL"}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
