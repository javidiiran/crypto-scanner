"""AB engine regression tests: CNV, OB/FVG confirmation and informational RSI."""
import unittest
import numpy as np
import pandas as pd
from ob_sqzmom_engine import analyze_market, rsi_wilder, ob_fvg_confirmations

class ABEngineTests(unittest.TestCase):
    def test_ab_module_imports_and_calculates(self):
        idx = pd.date_range('2026-01-01', periods=80, freq='h', tz='UTC')
        close = np.linspace(100, 130, 80)
        frame = pd.DataFrame({'open': close-0.1, 'high': close+1,
                              'low': close-1, 'close': close,
                              'volume': np.full(80, 100.0)}, index=idx)
        result = analyze_market(frame)
        self.assertEqual(len(result['candles']), 80)
        self.assertIn(result['latest_cnv_color'], ('BLUE', 'PINK'))
        self.assertEqual(len(rsi_wilder(frame.close, 14)), 80)
        self.assertEqual(len(result['confirmations']), 80)
        self.assertEqual(len(ob_fvg_confirmations(frame)), 80)

if __name__ == '__main__':
    unittest.main()
