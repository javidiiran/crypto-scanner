"""Offline tests for top-300 ranking, Kraken market mapping and candle handling."""
import unittest
from unittest.mock import patch

import pandas as pd
import scanner


class KrakenAdapterTests(unittest.TestCase):
    def test_market_cap_fetch_uses_two_250_item_pages_and_keeps_top_300(self):
        page1 = [{"id": f"coin-{i}", "name": f"Coin {i}", "symbol": f"c{i}",
                  "market_cap_rank": i, "market_cap": 1000-i} for i in range(1, 251)]
        page2 = [{"id": f"coin-{i}", "name": f"Coin {i}", "symbol": f"c{i}",
                  "market_cap_rank": i, "market_cap": 1000-i} for i in range(251, 501)]
        with patch.object(scanner, "get_json", side_effect=[page1, page2]) as mocked, \
             patch.object(scanner.time, "sleep"):
            result = scanner.fetch_top_market_cap()
        self.assertEqual(len(result), 300)
        self.assertEqual(result[0]["rank"], 1)
        self.assertEqual(result[-1]["rank"], 300)
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(mocked.call_args_list[0].args[1]["per_page"], 250)
        self.assertEqual(mocked.call_args_list[1].args[1]["page"], 2)

    def test_coin_symbols_map_to_kraken_tickers(self):
        self.assertEqual(scanner.normalize_symbol("BTC"), "XBT")
        self.assertEqual(scanner.normalize_symbol("DOGE"), "XDG")
        coins = [
            {"symbol": "BTC", "rank": 1, "name": "Bitcoin"},
            {"symbol": "WBTC", "rank": 15, "name": "Wrapped Bitcoin"},
            {"symbol": "ETH", "rank": 2, "name": "Ethereum"},
        ]
        markets = {"XBT": {"symbol": "PF_XBTUSD", "pair": "PF_XBTUSD", "base": "XBT"},
                   "ETH": {"symbol": "ETHUSD", "pair": "ETHUSD", "base": "ETH"}}
        mapped = scanner.map_top300_to_kraken(coins, markets)
        self.assertEqual([x["name"] for x in mapped], ["Bitcoin", "Ethereum"])

    def test_candle_normalizer_excludes_live_bar(self):
        now_ms = 1_800_000_000_000
        start_ms = now_ms - 300 * scanner.INTERVAL_MS
        rows = []
        for i in range(260):
            rows.append({"time": start_ms + i * scanner.INTERVAL_MS,
                         "open": "100", "high": "102", "low": "99",
                         "close": "101", "volume": 123})
        forming_start = (now_ms // scanner.INTERVAL_MS) * scanner.INTERVAL_MS
        rows.append({"time": forming_start, "open": "101", "high": "103",
                     "low": "100", "close": "102", "volume": 50})
        frame = scanner._normalize_candle_frame(rows, now_ms, "futures")
        self.assertEqual(len(frame), 260)
        self.assertLess(int(frame.index[-1].timestamp() * 1000) + scanner.INTERVAL_MS, now_ms)

    def test_spot_ohlc_row_normalization(self):
        now_ms = 1_800_000_000_000
        start_ms = now_ms - 300 * scanner.INTERVAL_MS
        rows = [[(start_ms + i * scanner.INTERVAL_MS) // 1000,
                 "100", "102", "99", "101", "100.5", "123", 10] for i in range(260)]
        frame = scanner._normalize_candle_frame(rows, now_ms, "spot")
        self.assertEqual(len(frame), 260)
        self.assertEqual(float(frame.iloc[-1].volume), 123.0)

    def test_kraken_market_map_prefers_perpetual_when_quote_priority_matches(self):
        instruments = {"instruments": [
            {"symbol": "PF_XBTUSD", "tradeable": True, "isExpired": False, "status": "online"},
            {"symbol": "PF_ETHUSD", "tradeable": True, "isExpired": False, "status": "online"},
        ]}
        pairs = {"result": {
            "XXBTZUSD": {"altname": "XBTUSD", "wsname": "XBT/USD", "base": "XXBT", "quote": "ZUSD"},
            "XETHZUSD": {"altname": "ETHUSD", "wsname": "ETH/USD", "base": "XETH", "quote": "ZUSD"},
        }}
        with patch.object(scanner, "get_json", side_effect=[instruments, pairs]):
            markets = scanner.fetch_kraken_market_maps()
        self.assertEqual(markets["XBT"]["market_type"], "Futures Perpetual")
        self.assertEqual(markets["ETH"]["market_type"], "Futures Perpetual")


if __name__ == "__main__":
    unittest.main(verbosity=2)
