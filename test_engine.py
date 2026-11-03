import unittest

import numpy as np
import pandas as pd

from indicator_engine import (
    OrderBlockSettings,
    confirmation_window,
    evaluate_candidates,
    rsi_wilder,
    wave_trend,
    order_block_fvg_touches,
)


class EngineTests(unittest.TestCase):
    def setUp(self):
        idx = pd.date_range("2026-01-01", periods=80, freq="30min", tz="UTC")
        close = 100 + np.sin(np.arange(80) / 3) * 4 + np.arange(80) * 0.03
        self.df = pd.DataFrame({
            "open": close - 0.3,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1000,
        }, index=idx)

    def test_wave_trend_outputs_original_signal_columns(self):
        out = wave_trend(self.df)
        self.assertIn("wt1", out)
        self.assertIn("wt2", out)
        self.assertEqual(out.buy_signal.dtype, bool)
        self.assertEqual(out.sell_signal.dtype, bool)
        self.assertFalse((out.buy_signal & out.sell_signal).any())

    def test_rsi_range(self):
        rsi = rsi_wilder(self.df.close)
        finite = rsi.dropna()
        self.assertGreater(len(finite), 20)
        self.assertTrue(((finite >= 0) & (finite <= 100)).all())

    def test_rsi_all_up_and_flat_series(self):
        rising = pd.Series(np.arange(40, dtype=float) + 100)
        flat = pd.Series(np.full(40, 100.0))
        self.assertEqual(float(rsi_wilder(rising).dropna().iloc[-1]), 100.0)
        self.assertEqual(float(rsi_wilder(flat).dropna().iloc[-1]), 50.0)

    def test_fvg_blue_and_orange_map_to_bullish_and_bearish_touch(self):
        idx = pd.date_range("2026-01-01", periods=60, freq="30min", tz="UTC")
        close = np.full(60, 100.0)
        opn = close - 0.1
        high = close + 0.2
        low = close - 0.2
        # Bullish FVG at bar 40: current low > high two bars earlier.
        high[38] = 102.0
        opn[40], close[40], low[40], high[40] = 105.2, 105.5, 105.0, 105.7
        # Bearish FVG at bar 45: current high < low two bars earlier.
        low[43] = 108.0
        opn[45], close[45], high[45], low[45] = 104.8, 104.5, 105.0, 104.2
        frame = pd.DataFrame({"open": opn, "high": high, "low": low, "close": close}, index=idx)
        touch = order_block_fvg_touches(frame)
        self.assertTrue(bool(touch.bull_fvg_touch.iloc[40]))
        self.assertTrue(bool(touch.bear_fvg_touch.iloc[45]))
        self.assertTrue(bool(touch.bullish_touch.iloc[40]))
        self.assertTrue(bool(touch.bearish_touch.iloc[45]))

    def test_confirmation_window_is_four_before_and_two_after_excluding_signal(self):
        self.assertEqual(confirmation_window(10, 30), (6, 12))

    def test_signal_candidate_direction_and_confirmation_logic(self):
        # Build a deterministic synthetic WT event at bar 10 and explicit touches.
        idx = self.df.index
        wt = pd.DataFrame({"buy_signal": False, "sell_signal": False}, index=idx)
        wt.loc[idx[10], "buy_signal"] = True
        touches = pd.DataFrame({
            "bullish_touch": False,
            "bearish_touch": False,
            "bull_ob_touch": False,
            "bear_ob_touch": False,
            "bull_fvg_touch": False,
            "bear_fvg_touch": False,
        }, index=idx)
        touches.loc[idx[8], "bullish_touch"] = True
        rsi = pd.Series(55.0, index=idx)
        candidates = evaluate_candidates(self.df, wt=wt, touches=touches, rsi=rsi)
        candidate = next(c for c in candidates if c["signal_index"] == 10)
        self.assertEqual(candidate["status"], "CONFIRMED")
        self.assertEqual(candidate["priority"], 1)

    def test_any_opposite_touch_rejects_even_if_matching_touch_exists(self):
        idx = self.df.index
        wt = pd.DataFrame({"buy_signal": False, "sell_signal": False}, index=idx)
        wt.loc[idx[10], "buy_signal"] = True
        touches = pd.DataFrame({
            "bullish_touch": False,
            "bearish_touch": False,
            "bull_ob_touch": False,
            "bear_ob_touch": False,
            "bull_fvg_touch": False,
            "bear_fvg_touch": False,
        }, index=idx)
        touches.loc[idx[7], "bullish_touch"] = True
        touches.loc[idx[12], "bearish_touch"] = True
        result = evaluate_candidates(self.df, wt=wt, touches=touches, rsi=pd.Series(52.0, index=idx))
        candidate = next(c for c in result if c["signal_index"] == 10)
        self.assertEqual(candidate["status"], "REJECTED_OPPOSITE_TOUCH")

    def test_later_opposite_touch_rejects_an_earlier_matching_confirmation(self):
        idx = self.df.index
        wt = pd.DataFrame({"buy_signal": False, "sell_signal": False}, index=idx)
        wt.loc[idx[10], "buy_signal"] = True
        touches = pd.DataFrame({
            "bullish_touch": False,
            "bearish_touch": False,
            "bull_ob_touch": False,
            "bear_ob_touch": False,
            "bull_fvg_touch": False,
            "bear_fvg_touch": False,
        }, index=idx)
        touches.loc[idx[8], "bullish_touch"] = True
        touches.loc[idx[12], "bearish_touch"] = True
        rsi = pd.Series(54.0, index=idx)
        early = evaluate_candidates(self.df.iloc[:12], wt=wt.iloc[:12], touches=touches.iloc[:12], rsi=rsi.iloc[:12])
        early_candidate = next(c for c in early if c["signal_index"] == 10)
        self.assertEqual(early_candidate["status"], "CONFIRMED")
        final = evaluate_candidates(self.df, wt=wt, touches=touches, rsi=rsi)
        final_candidate = next(c for c in final if c["signal_index"] == 10)
        self.assertEqual(final_candidate["status"], "REJECTED_OPPOSITE_TOUCH")

    def test_no_touch_expires_after_two_following_closed_candles(self):
        idx = self.df.index
        wt = pd.DataFrame({"buy_signal": False, "sell_signal": False}, index=idx)
        wt.loc[idx[10], "sell_signal"] = True
        touches = pd.DataFrame({
            "bullish_touch": False,
            "bearish_touch": False,
            "bull_ob_touch": False,
            "bear_ob_touch": False,
            "bull_fvg_touch": False,
            "bear_fvg_touch": False,
        }, index=idx)
        result = evaluate_candidates(self.df, wt=wt, touches=touches, rsi=pd.Series(45.0, index=idx))
        candidate = next(c for c in result if c["signal_index"] == 10)
        self.assertEqual(candidate["status"], "EXPIRED_NO_MATCHING_TOUCH")

    def test_rsi_priority_sell_below_50_is_first_group(self):
        idx = self.df.index
        wt = pd.DataFrame({"buy_signal": False, "sell_signal": False}, index=idx)
        wt.loc[idx[10], "sell_signal"] = True
        touches = pd.DataFrame({
            "bullish_touch": False,
            "bearish_touch": False,
            "bull_ob_touch": False,
            "bear_ob_touch": False,
            "bull_fvg_touch": False,
            "bear_fvg_touch": False,
        }, index=idx)
        touches.loc[idx[9], "bearish_touch"] = True
        result = evaluate_candidates(self.df, wt=wt, touches=touches, rsi=pd.Series(49.9, index=idx))
        candidate = next(c for c in result if c["signal_index"] == 10)
        self.assertEqual(candidate["priority"], 1)


if __name__ == "__main__":
    unittest.main()
