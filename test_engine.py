import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import ob_sqzmom_engine as eng

class EngineTests(unittest.TestCase):
    def frame(self, n=60):
        idx = pd.date_range('2026-01-01', periods=n, freq='h', tz='UTC')
        close = np.linspace(100, 160, n)
        return pd.DataFrame({'open': close-0.2, 'high': close+1,
                             'low': close-1, 'close': close,
                             'volume': np.full(n, 100.0)}, index=idx)

    def test_rsi_wilder_range(self):
        x = pd.Series(np.arange(40, dtype=float))
        r = eng.rsi_wilder(x, 14)
        self.assertAlmostEqual(float(r.iloc[-1]), 100.0)

    def test_cnv_output_shape(self):
        d = self.frame()
        c = eng.cnv(d)
        self.assertEqual(len(c), len(d))
        self.assertTrue(set(c.blue.unique()).issubset({True, False}))

    def test_rsi_is_not_a_signal_filter(self):
        d = self.frame()
        # Monotonically rising prices produce RSI > 50, yet BUY must pass
        # when CNV pattern and OB/FVG confirmation pass.
        blue = pd.Series(False, index=d.index)
        blue.iloc[-2:] = True
        colors = pd.DataFrame({'blue': blue, 'pink': ~blue}, index=d.index)
        zone = {'kind':'FVG', 'direction':'BUY', 'top':150.0, 'bot':149.0,
                'created_index':len(d)-2, 'age_bars':1,
                'touched_last_5':True, 'confirmed':True}
        confirmations = [None] * len(d)
        confirmations[-1] = zone
        with patch.object(eng, 'cnv', return_value=colors), \
             patch.object(eng, 'ob_fvg_confirmations', return_value=confirmations):
            result = eng.analyze_market(d)
        self.assertEqual(len(result['signals']), 1)
        self.assertEqual(result['signals'][0]['direction'], 'BUY')
        self.assertGreater(result['signals'][0]['rsi'], 50)
        self.assertNotIn('rsi_condition_value', result['signals'][0])

    def test_bullish_fvg_can_confirm_after_touch(self):
        n = 40
        idx = pd.date_range('2026-01-01', periods=n, freq='h', tz='UTC')
        op = np.full(n, 100.0); cl = np.full(n, 100.0)
        hi = np.full(n, 101.0); lo = np.full(n, 99.0)
        # A bullish gap forms at bar 25; following bars stay above its invalidation.
        for i in range(25, n):
            op[i] = 103.5; cl[i] = 104.0; hi[i] = 105.0; lo[i] = 102.0
        d = pd.DataFrame({'open':op,'high':hi,'low':lo,'close':cl,
                          'volume':np.full(n,100.0)}, index=idx)
        result = eng.ob_fvg_confirmations(d)
        self.assertIsNotNone(result[-1])
        self.assertEqual(result[-1]['kind'], 'FVG')
        self.assertEqual(result[-1]['direction'], 'BUY')
        self.assertTrue(result[-1]['touched_last_5'])

if __name__ == '__main__':
    unittest.main()
