import unittest
import numpy as np, pandas as pd
from indicator_engine import analyze_market, rsi_wilder
class EngineTests(unittest.TestCase):
 def frame(self, closes, vols=None):
  n=len(closes); idx=pd.date_range('2026-01-01',periods=n,freq='h',tz='UTC')
  return pd.DataFrame({'open':closes,'high':np.array(closes)+1,'low':np.array(closes)-1,'close':closes,'volume':vols or [100]*n},index=idx)
 def test_rsi_range(self):
  x=pd.Series(np.arange(40,dtype=float)); r=rsi_wilder(x,14); self.assertAlmostEqual(float(r.iloc[-1]),100.0)
 def test_cnv_color_and_structure(self):
  a=analyze_market(self.frame(list(range(1,60))))
  self.assertIn(a['latest_cnv_color'],('BLUE','PINK')); self.assertEqual(len(a['candles']),59)
 def test_only_closed_input_passed_to_engine(self):
  a=analyze_market(self.frame(list(range(1,45))))
  self.assertEqual(len(a['signals']),len([s for s in a['signals'] if s['signal_index']<44]))
if __name__=='__main__': unittest.main()
