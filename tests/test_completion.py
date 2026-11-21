from pathlib import Path
from unittest.mock import patch
import sys,unittest,tempfile,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from external_prepare import clock,event_rows,minute_labels
from completion import alarm_counts
import control
class ExternalContractTests(unittest.TestCase):
 def test_midnight_and_hypopnea(self):
  e=event_rows('00:00:10 HYP-C 20 90\n00:01:00 APNEA-O PB 10 89','23.59.00',300)
  self.assertEqual(e[0][:2],(70,90));self.assertEqual(minute_labels(e,300).tolist(),[0,1,1,0,0])
 def test_exact_end_boundary(self):self.assertEqual(minute_labels([(0,60,'APNEA-O')],120).tolist(),[1,0])
 def test_unknown_type_blocks(self):
  with self.assertRaises(ValueError):event_rows('01:00:00 UNKNOWN 20','00.00.00',8000)
 def test_outside_duration_blocks(self):
  with self.assertRaises(ValueError):event_rows('00:10:00 APNEA-O 20','00.00.00',100)
 def test_empty_optional_status(self):
  with tempfile.TemporaryDirectory() as t,patch.object(control,'ROOT',Path(t)):
   result=control.status();self.assertFalse(result['runner_lock_held']);self.assertEqual(result['registry_counts'],{})
 def test_alarm_suppression(self):
  y=np.array([0,0,0,1,0,0]);p=np.array([1,1,0,1,0,1]);r=alarm_counts(y,p,np.full(6,.5),np.zeros(6),np.arange(6));self.assertEqual(r['alarms'],2);self.assertEqual(r['false_alarm_starts'],1)

class SavedPredictionTests(unittest.TestCase):
 def test_legacy_AHI_groups_need_no_pickle(self):
  import completion,pandas as pd
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);(root/'private').mkdir();pd.DataFrame({'AHI':[4.,2.],'group':['g0','g1']}).to_csv(root/'private/reference_grouped.csv',index=False)
   p=root/'pred.npz';np.savez(p,y=np.array([2.,4.]),p=np.array([3.,3.]),record=np.array([1,0]),group=np.array(['g1','g0'],dtype=object))
   with patch.object(completion,'ROOT',root):
    d=completion.load_ahi_prediction(p);self.assertEqual(d['group'].tolist(),['g1','g0']);np.testing.assert_array_equal(d['y'],[2.,4.])

class MetricUncertaintyTests(unittest.TestCase):
 def test_saved_prediction_bootstrap(self):
  import completion,json
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);(root/'private/predictions').mkdir(parents=True);(root/'results').mkdir();rng=np.random.default_rng(14);prob=rng.uniform(.1,.9,100);y=rng.binomial(1,prob)
   np.savez(root/'private/predictions/detect_fixture_OOF.npz',y=y,p=prob,group=np.repeat(np.arange(10),10),threshold=np.full(100,.5))
   with patch.object(completion,'ROOT',root):completion.metric_uncertainty()
   d=json.loads((root/'results/ADDITIONAL_METRIC_UNCERTAINTY.json').read_text())[0];self.assertEqual(d['valid_bootstrap_counts']['AP'],500);self.assertEqual(len(d['CI95']['calibration_slope']),2);self.assertFalse(d['intervals_include_training_variability'])

class ManuscriptSmokeTests(unittest.TestCase):
 def test_empty_incomplete_program_stays_incomplete(self):
  import completion,report,json,zipfile,xml.etree.ElementTree as ET
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder)/'study';root.mkdir()
   for directory in ['results','private/predictions','figures','manuscript','reports','logs']:(root/directory).mkdir(parents=True,exist_ok=True)
   with patch.object(completion,'ROOT',root),patch.object(report,'ROOT',root),patch.object(report,'log',lambda s:None):completion.manuscript_and_package()
   receipt=json.loads((root/'results/COMPLETION_ASSESSMENT.json').read_text());self.assertEqual(receipt['status'],'incomplete')
   with zipfile.ZipFile(root/'manuscript/FINAL_MANUSCRIPT.docx') as z:ET.fromstring(z.read('word/document.xml'))
   with zipfile.ZipFile(root.parent/'OSA_ECG_Multimethod_v1_Review_Package.zip') as z:self.assertFalse(any('/private/' in n for n in z.namelist()))
