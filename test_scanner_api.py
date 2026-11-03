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


    def test_stale_chart_response_is_retried_and_never_used(self):
        now_ms = 1_800_000_000_000
        latest_closed = now_ms - (now_ms % scanner.INTERVAL_MS) - scanner.INTERVAL_MS

        def candles(last_stamp):
            return {"candles": [
                {"time": last_stamp - (59 - i) * scanner.INTERVAL_MS,
                 "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10}
                for i in range(60)
            ]}

        stale = candles(latest_closed - 30 * scanner.INTERVAL_MS)
        fresh = candles(latest_closed)
        with patch.object(scanner, "MIN_CANDLES", 60), \
             patch.object(scanner, "get_json", side_effect=[stale, fresh]) as mocked, \
             patch.object(scanner.time, "sleep"):
            frame = scanner.fetch_closed_candles("PF_XBTUSD", now_ms=now_ms)
        self.assertEqual(len(frame), 60)
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(mocked.call_args_list[1].args[1], {"count": scanner.CANDLE_LIMIT})
        self.assertEqual(frame.index[-1].value // 1_000_000, latest_closed)

    def test_stale_chart_data_fails_instead_of_scanning_old_candles(self):
        now_ms = 1_800_000_000_000
        latest_closed = now_ms - (now_ms % scanner.INTERVAL_MS) - scanner.INTERVAL_MS
        stale_stamp = latest_closed - 20 * scanner.INTERVAL_MS
        payload = {"candles": [
            {"time": stale_stamp - (59 - i) * scanner.INTERVAL_MS,
             "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10}
            for i in range(60)
        ]}
        with patch.object(scanner, "MIN_CANDLES", 60), \
             patch.object(scanner, "get_json", return_value=payload), \
             patch.object(scanner.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "STALE_KRAKEN_CHART_DATA"):
                scanner.fetch_closed_candles("PF_XBTUSD", now_ms=now_ms)

    def test_first_btc_direction_uses_latest_wt_without_ob_fvg(self):
        idx = pd.date_range("2026-01-01", periods=3, freq="30min", tz="UTC")
        candidates = [
            {"status": "CONFIRMED", "direction": "BUY", "timestamp_ms": 1, "timestamp": idx[0], "rsi": 45},
            {"status": "REJECTED_OPPOSITE_TOUCH", "direction": "SELL", "timestamp_ms": 2, "timestamp": idx[1], "rsi": 55},
        ]
        state = {"btc_direction": None, "btc_signal_timestamp_ms": 0, "btc_signal_time": "—", "btc_signal_rsi": float("nan"), "btc_source": None}
        result = scanner._btc_context("PF_XBTUSD", {"candidates": candidates, "latest_rsi": 48.2}, state)
        self.assertEqual(result["direction"], "SELL")
        self.assertEqual(result["source"], "INITIAL_WAVETREND")
        self.assertEqual(result["current_rsi"], 48.2)

    def test_later_btc_direction_change_requires_confirmed_box_touch(self):
        idx = pd.date_range("2026-01-01", periods=3, freq="30min", tz="UTC")
        state = {"btc_direction": "BUY", "btc_signal_timestamp_ms": 1, "btc_signal_time": "old", "btc_signal_rsi": 52.0, "btc_source": "INITIAL_WAVETREND"}
        pending = {"status": "PENDING_NO_MATCHING_TOUCH", "direction": "SELL", "timestamp_ms": 2, "timestamp": idx[1], "rsi": 48, "recent_actionable": True}
        result = scanner._btc_context("PF_XBTUSD", {"candidates": [pending], "latest_rsi": 48.2}, state)
        self.assertEqual(result["direction"], "BUY")
        confirmed = dict(pending, status="CONFIRMED")
        result = scanner._btc_context("PF_XBTUSD", {"candidates": [confirmed], "latest_rsi": 48.2}, state)
        self.assertEqual(result["direction"], "SELL")
        self.assertEqual(result["source"], "OB_FVG_CONFIRMED")


    def test_btc_state_survives_json_save_and_load(self):
        import tempfile
        from pathlib import Path
        state = {
            "sent": {"one"}, "cancelled": set(), "btc_direction": "BUY",
            "btc_signal_timestamp_ms": 123456, "btc_signal_time": "2026-01-01 00:00 UTC",
            "btc_signal_rsi": 54.2, "btc_source": "INITIAL_WAVETREND",
        }
        with tempfile.TemporaryDirectory() as tmp, patch.object(scanner, "STATE_FILE", Path(tmp) / "state.json"):
            scanner.save_state(state)
            loaded = scanner.load_state()
        self.assertEqual(loaded["btc_direction"], "BUY")
        self.assertEqual(loaded["btc_signal_timestamp_ms"], 123456)
        self.assertAlmostEqual(loaded["btc_signal_rsi"], 54.2)
        self.assertIn("one", loaded["sent"])

    def test_alert_contains_asset_and_btc_signal_and_current_rsi(self):
        item = {
            "direction": "BUY", "priority": 1, "symbol": "PI_ETHUSD", "price": 2000,
            "rsi": 56.1, "current_rsi": 58.4, "signal_time": "2026-01-01 00:00 UTC",
            "btc_alignment": "ALIGNED", "match_kind": "Order Block",
        }
        btc = {
            "symbol": "PF_XBTUSD", "direction": "BUY", "signal_time": "2026-01-01 00:00 UTC",
            "signal_rsi": 53.2, "current_rsi": 55.7, "source": "INITIAL_WAVETREND",
        }
        message = scanner.make_alert(item, btc)
        self.assertIn("RSI(14) فعلی ارز: 58.4", message)
        self.assertIn("RSI(14) فعلی بیت‌کوین", message)
        self.assertIn("بدون نیاز به OB/FVG", message)
        self.assertIn("هفت کندل بسته اخیر", message)

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
