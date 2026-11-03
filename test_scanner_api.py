import unittest
from unittest.mock import patch

import pandas as pd

import scanner


class ScannerApiTests(unittest.TestCase):
    def test_only_kraken_perpetual_contracts_are_selected(self):
        payload = {"instruments": [
            {"symbol": "PF_XBTUSD", "tradeable": True, "status": "online"},
            {"symbol": "PI_ETHUSD", "tradeable": True, "status": "trading"},
            {"symbol": "FI_ETHUSD_OLD", "tradeable": True, "status": "online"},
            {"symbol": "PF_BAD", "tradeable": False, "status": "online"},
            {"symbol": "PF_EXPIRED", "tradeable": True, "isExpired": True, "status": "online"},
            {"symbol": "BTCUSDT", "tradeable": True, "status": "online"},
        ]}
        with patch.object(scanner, "get_json", return_value=payload):
            self.assertEqual(scanner.fetch_symbols(), ["PF_XBTUSD", "PI_ETHUSD"])

    def test_candle_fetch_discards_forming_bar(self):
        now_ms = 1_800_000_000_000
        closed = now_ms - (now_ms % scanner.INTERVAL_MS) - scanner.INTERVAL_MS
        forming = closed + scanner.INTERVAL_MS
        payload = {"candles": [
            {"time": closed, "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10},
            {"time": forming, "open": 1.5, "high": 2, "low": 1, "close": 1.8, "volume": 5},
        ]}
        # The fetcher requires 100 candles; lower threshold for this isolated unit test.
        with patch.object(scanner, "MIN_CANDLES", 1), patch.object(scanner, "get_json", return_value=payload):
            frame = scanner.fetch_closed_candles("PF_XBTUSD", now_ms=now_ms)
        self.assertEqual(len(frame), 1)
        self.assertEqual(float(frame.iloc[0].close), 1.5)

    def test_btc_context_uses_latest_confirmed_signal(self):
        idx = pd.date_range("2026-01-01", periods=3, freq="30min", tz="UTC")
        candidates = [
            {"status": "CONFIRMED", "direction": "BUY", "timestamp_ms": 1, "timestamp": idx[0], "rsi": 45},
            {"status": "REJECTED_OPPOSITE_TOUCH", "direction": "SELL", "timestamp_ms": 2, "timestamp": idx[1], "rsi": 55},
            {"status": "CONFIRMED", "direction": "SELL", "timestamp_ms": 3, "timestamp": idx[2], "rsi": 49},
        ]
        result = scanner._btc_context("PF_XBTUSD", {"candidates": candidates, "latest_rsi": 48.2})
        self.assertEqual(result["direction"], "SELL")
        self.assertEqual(result["current_rsi"], 48.2)

    def test_final_btc_filter_blocks_every_opposite_direction(self):
        rows = [
            {"symbol": "PI_ETHUSD", "direction": "BUY"},
            {"symbol": "PI_SOLUSD", "direction": "SELL"},
        ]
        kept, blocked = scanner.filter_btc_aligned_alerts(rows, "BUY")
        self.assertEqual([x["symbol"] for x in kept], ["PI_ETHUSD"])
        self.assertEqual(blocked, 1)
        self.assertEqual(kept[0]["btc_alignment"], "ALIGNED")

    def test_final_btc_filter_fails_closed_without_confirmed_btc_direction(self):
        rows = [{"symbol": "PI_ETHUSD", "direction": "BUY"}]
        kept, blocked = scanner.filter_btc_aligned_alerts(rows, "NO_CONFIRMED_SIGNAL")
        self.assertEqual(kept, [])
        self.assertEqual(blocked, 1)


if __name__ == "__main__":
    unittest.main()
