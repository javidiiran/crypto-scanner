"""Tests for the AB engine module (CNV + RSI)."""
import unittest
import numpy as np
import pandas as pd
from ob_sqzmom_engine import analyze_market, rsi_wilder

class ABEngineTests(unittest.TestCase):
    def test_ab_module_imports_and_calculates(self):
        idx = pd.date_range('2026-01-01', periods=60, freq='h', tz='UTC')
        close = np.linspace(100, 130, 60)
        frame = pd.DataFrame({'open': close, 'high': close+1, 'low': close-1,
                              'close': close, 'volume': np.full(60, 100.0)}, index=idx)
        result = analyze_market(frame)
        self.assertEqual(len(result['candles']), 60)
        self.assertIn(result['latest_cnv_color'], ('BLUE', 'PINK'))
        self.assertEqual(len(rsi_wilder(frame.close, 14)), 60)

if __name__ == '__main__':
    unittest.main()
