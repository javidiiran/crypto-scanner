import importlib.util
import unittest
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scanner.py"
spec = importlib.util.spec_from_file_location("scanner4", MODULE_PATH)
scanner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = scanner
spec.loader.exec_module(scanner)


class Scanner4LogicTests(unittest.TestCase):
    def test_rsi_is_defined_and_returns_values_after_warmup(self):
        values = [float(x) for x in range(40)]
        result = scanner.rsi(values, 14)
        self.assertIsNone(result[13])
        self.assertAlmostEqual(result[14], 100.0)
        self.assertAlmostEqual(result[-1], 100.0)

    def test_color_transition_on_latest_step_buy(self):
        self.assertEqual(scanner.bxt_color_change(["RED", "RED", "GREEN"]), "BUY")

    def test_color_transition_on_second_step_buy(self):
        self.assertEqual(scanner.bxt_color_change(["RED", "GREEN", "GREEN"]), "BUY")

    def test_color_transition_on_second_step_sell(self):
        self.assertEqual(scanner.bxt_color_change(["GREEN", "RED", "RED"]), "SELL")

    def test_latest_transition_wins_when_both_steps_change(self):
        self.assertEqual(scanner.bxt_color_change(["GREEN", "RED", "GREEN"]), "BUY")

    def test_no_transition_returns_none(self):
        self.assertIsNone(scanner.bxt_color_change(["GREEN", "GREEN", "GREEN"]))

    def test_bxt_color_series_runs_without_missing_rsi(self):
        candles = []
        price = 100.0
        for i in range(220):
            price += (0.8 if (i // 8) % 2 == 0 else -0.6) + (i % 3) * 0.03
            candles.append(scanner.Candle(i, price - 0.1, price + 0.3, price - 0.3, price, 1000.0))
        colors = scanner.bxt_color_series(candles)
        self.assertEqual(len(colors), len(candles))
        self.assertIn(colors[-1], ("GREEN", "RED"))

    def test_delta_zone_confirmation_is_boolean_for_full_history(self):
        candles = []
        price = 100.0
        for i in range(220):
            move = ((i % 17) - 8) * 0.08
            old = price
            price = max(1.0, price + move)
            candles.append(scanner.Candle(i, old, max(old, price) + 0.15,
                                          min(old, price) - 0.15, price, 100.0 + (i % 9) * 5))
        result = scanner.delta_zone_confirmation(candles, "BUY", lookback=6, tick_size=0.01)
        self.assertIsInstance(result, bool)

    def test_one_hour_change_uses_four_15m_intervals(self):
        candles = [scanner.Candle(i, 0, 0, 0, close, 1) for i, close in enumerate([100, 101, 102, 103, 104])]
        self.assertAlmostEqual(scanner.ticker_change_1h(candles), 4.0)

    def test_report_btc_signal_then_btc_colors_then_ranked_signals(self):
        btc = {"symbol": "PF_XBTUSD", "direction": "BUY", "change24h": 2.5, "change1h": 0.4}
        signals = [
            {"symbol": "SELL_SMALL", "direction": "SELL", "change24h": -1.2, "change1h": -0.2},
            {"symbol": "BUY_SMALL", "direction": "BUY", "change24h": 3.0, "change1h": 0.1},
            {"symbol": "SELL_LARGE", "direction": "SELL", "change24h": -8.0, "change1h": -1.0},
            {"symbol": "BUY_LARGE", "direction": "BUY", "change24h": 9.0, "change1h": 2.0},
        ]
        report = scanner.build_report_messages(btc, ["RED", "GREEN"], signals)
        self.assertIn("PF_XBTUSD", report[0])
        self.assertIn("RED → GREEN", report[1])
        self.assertIn("BUY_LARGE", report[2])
        self.assertIn("BUY_SMALL", report[3])
        self.assertIn("SELL_LARGE", report[4])
        self.assertIn("SELL_SMALL", report[5])
        self.assertEqual(len(report[2].split("|")), 3)

    def test_report_without_btc_signal_starts_with_btc_colors(self):
        report = scanner.build_report_messages(None, ["GREEN", "RED"], [])
        self.assertEqual(len(report), 1)
        self.assertIn("BTC B-Xtrender Color", report[0])
        self.assertIn("GREEN → RED", report[0])


if __name__ == "__main__":
    unittest.main()
