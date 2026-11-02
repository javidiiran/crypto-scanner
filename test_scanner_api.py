"""Offline tests for Kraken perpetual filtering and candle closure handling."""
import unittest
from unittest.mock import patch
import scanner


class KrakenAdapterTests(unittest.TestCase):
    def test_only_perpetual_contracts_are_selected(self):
        payload = {"instruments": [
            {"symbol": "PF_XBTUSD", "tradeable": True, "isExpired": False},
            {"symbol": "PI_XBTUSD", "tradeable": True, "isExpired": False},
            {"symbol": "FF_XBTUSD_261225", "tradeable": True, "isExpired": False},
            {"symbol": "PF_OLDUSD", "tradeable": True, "isExpired": True},
        ]}
        with patch.object(scanner, "get_json", return_value=payload):
            self.assertEqual(scanner.fetch_symbols(), ["PF_XBTUSD", "PI_XBTUSD"])

    def test_closed_candles_excludes_forming_bar(self):
        now_ms = 1_800_000_000_000
        start_ms = now_ms - 300 * scanner.INTERVAL_MS
        rows = [{"time": start_ms + i * scanner.INTERVAL_MS, "open": "100", "high": "102",
                 "low": "99", "close": "101", "volume": 123} for i in range(260)]
        forming_start = (now_ms // scanner.INTERVAL_MS) * scanner.INTERVAL_MS
        rows.append({"time": forming_start, "open": "101", "high": "103", "low": "100", "close": "102", "volume": 50})
        with patch.object(scanner, "get_json", return_value={"candles": rows}):
            frame = scanner.fetch_closed_candles("PF_XBTUSD", now_ms)
        self.assertEqual(len(frame), 260)
        self.assertLess(int(frame.index[-1].timestamp() * 1000) + scanner.INTERVAL_MS, now_ms)

    def test_signal_must_match_btc_bias(self):
        import pandas as pd
        idx = pd.date_range("2026-11-02", periods=260, freq="30min", tz="UTC")
        candles = pd.DataFrame({"open": 1., "high": 2., "low": .5, "close": 1.5, "volume": 1.}, index=idx)
        result = pd.DataFrame({"signal": [""]*259+["BUY"], "BUY": [False]*259+[True],
                               "SELL": [False]*260, "buy_trough": [False]*259+[True],
                               "sell_peak": [False]*260, "close": [1.5]*260,
                               "sqzmom": [0.2]*260, "sqzmom_peak_trough": [0.3]*260,
                               "bull_ob_inside": [True]*260, "bear_ob_inside": [False]*260,
                               "bull_ob_count": [1]*260, "bear_ob_count": [0]*260,
                               "buy_ob_confirm": [True]*260, "sell_ob_confirm": [False]*260}, index=idx)
        stamp = int(idx[-1].timestamp()*1000)
        with patch.object(scanner, "fetch_closed_candles", return_value=candles), \
             patch.object(scanner, "evaluate_signals", return_value=result):
            rejected = scanner.evaluate_symbol("PF_ETHUSD", stamp, "SELL", stamp)
            item = scanner.evaluate_symbol("PF_ETHUSD", stamp, "BUY", stamp)
        self.assertEqual(rejected["status"], "BUY_BLOCKED_BY_BTC_BIAS")
        self.assertEqual(item["status"], "SIGNAL_READY")
        self.assertEqual(item["raw_signal"], "BUY")


if __name__ == "__main__":
    unittest.main(verbosity=2)
