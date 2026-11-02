"""Offline tests for Kraken instrument filtering and candle closure handling."""
import unittest
from unittest.mock import patch

import scanner


class KrakenAdapterTests(unittest.TestCase):
    def test_only_tradeable_perpetual_prefixes_are_selected(self):
        payload = {"result": "success", "instruments": [
            {"symbol": "PF_XBTUSD", "tradeable": True, "isExpired": False},
            {"symbol": "PI_XBTUSD", "tradeable": True, "isExpired": False},
            {"symbol": "FF_XBTUSD_261225", "tradeable": True, "isExpired": False},
            {"symbol": "FI_XBTUSD_261225", "tradeable": True, "isExpired": False},
            {"symbol": "PF_ETHUSD", "tradeable": False, "isExpired": False},
            {"symbol": "PF_OLDUSD", "tradeable": True, "isExpired": True},
        ]}
        with patch.object(scanner, "get_json", return_value=payload):
            self.assertEqual(scanner.fetch_symbols(), ["PF_XBTUSD", "PI_XBTUSD"])

    def test_closed_candles_excludes_live_bar(self):
        now_ms = 1_800_000_000_000
        start_ms = now_ms - 300 * scanner.INTERVAL_MS
        rows = []
        for i in range(260):
            rows.append({"time": start_ms + i * scanner.INTERVAL_MS,
                         "open": "100", "high": "102", "low": "99",
                         "close": "101", "volume": 123})
        # A forming bar at the current interval start must not be included.
        forming_start = (now_ms // scanner.INTERVAL_MS) * scanner.INTERVAL_MS
        rows.append({"time": forming_start, "open": "101", "high": "103",
                     "low": "100", "close": "102", "volume": 50})
        with patch.object(scanner, "get_json", return_value={"candles": rows}):
            frame = scanner.fetch_closed_candles("PF_XBTUSD", now_ms)
        self.assertEqual(len(frame), 260)
        self.assertLess(int(frame.index[-1].timestamp() * 1000) + scanner.INTERVAL_MS, now_ms)


if __name__ == "__main__":
    unittest.main(verbosity=2)
