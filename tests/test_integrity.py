import sys,tempfile,unittest,time
from pathlib import Path
from unittest.mock import patch
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data import features,split_groups
from residual import bounds,valid_geometry,bouts,STATE
from neural import Net,Autoencoder,fit_net,seed
import neural
class IntegrityTests(unittest.TestCase):
 def test_group_separation(self):
  g=np.repeat(np.arange(12),10);y=np.tile([0,1],60)
  for a,b in split_groups(y,g):self.assertFalse(set(g[a])&set(g[b]))
 def test_window_geometry(self):
  self.assertEqual(bounds(10,13,2,1),(7,9,14,16));y=np.zeros(30,int);y[10:13]=1;self.assertTrue(valid_geometry(y,10,13,2,1));y[8]=1;self.assertFalse(valid_geometry(y,10,13,2,1))
 def test_bout_boundaries(self):self.assertEqual(bouts(np.array([0,1,1,0,1])),[(1,3),(4,5)])
 def test_causal_crop(self):
  # An arbitrary change in the future cannot enter the bounded feature function.
  rng=np.random.default_rng(2);x=rng.normal(size=18000);a=features(x[:6000])[0];x[6000:]*=100;np.testing.assert_equal(a,features(x[:6000])[0])
 def test_flat_qc(self):self.assertTrue(np.isnan(features(np.zeros(6000))[0]).all())
 def test_state_excludes_RMSSD(self):self.assertNotIn(3,STATE)
 def test_models_and_losses(self):
  x=torch.randn(2,1,1500)
  for kind in ['cnn','resnet','tcn','gru']:
   m=Net(kind);self.assertEqual(tuple(m(x).shape),(2,));self.assertLess(sum(p.numel() for p in m.parameters()),200000)
  for kind in ['ae','vae','masked']:
   m=Autoencoder(kind);a,b=m(x);self.assertEqual(a.shape,x.shape);loss=(a-x).square().mean()+b*.001;loss.backward();self.assertTrue(torch.isfinite(loss))
 def test_resumption(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);(root/'checkpoints_private').mkdir();(root/'results').mkdir();x=np.random.default_rng(2).normal(size=(8,1500)).astype('float32');y=np.array([0,1]*4);a=np.arange(6);b=np.arange(6,8)
   with patch.object(neural,'ROOT',root),patch.object(neural,'log',lambda s:None),patch.object(neural,'device',lambda:torch.device('cpu')),patch.object(neural,'stopped',return_value=False):
    seed(4);m=Net();full,info=fit_net(m,x,y,a,b,'full',time.time()+60,epochs=2,fixed=True,seed_value=4)
    seed(4);m=Net()
    with patch.object(neural,'stopped',side_effect=[False,True]):part,info=fit_net(m,x,y,a,b,'resume',time.time()+60,epochs=2,fixed=True,seed_value=4)
    self.assertIsNone(part);m=Net();resumed,info=fit_net(m,x,y,a,b,'resume',time.time()+60,epochs=2,fixed=True,seed_value=4)
    for k,v in full.state_dict().items():torch.testing.assert_close(v,resumed.state_dict()[k],rtol=0,atol=0)
if __name__=='__main__':unittest.main()

class ForecastContractTests(unittest.TestCase):
 def test_transition_eligibility_does_not_condition_on_intervening_N(self):
  import data
  import pandas as pd
  with tempfile.TemporaryDirectory() as folder:
   r=Path(folder);(r/'private/cache').mkdir(parents=True)
   pd.DataFrame([dict(record='fixture',group='g')]).to_csv(r/'private/reference_grouped.csv',index=False)
   pd.DataFrame([dict(samples=36000)]).to_csv(r/'private/record_audit.csv',index=False)
   y=np.array([0,0,1,0,1,0]);F=np.ones((6,2,12),np.float32);W=np.ones((6,1500),np.float32)
   p=r/'private/cache/fixture.npz';np.savez(p,F=F,W=W,y=y)
   with patch.object(data,'ROOT',r):
    x,target,g,rec,t,w=data.load('forecast_transition',1);self.assertEqual(t.tolist(),[0,1,3]);self.assertEqual(target.tolist(),[1,0,0])
    y[1]=1;np.savez(p,F=F,W=W,y=y)
    x,target,g,rec,t,w=data.load('forecast_transition',1);self.assertIn(0,t);self.assertEqual(int(target[np.flatnonzero(t==0)[0]]),0)
 def test_incomplete_target_excluded(self):
  import data
  import pandas as pd
  with tempfile.TemporaryDirectory() as folder:
   r=Path(folder);(r/'private/cache').mkdir(parents=True)
   pd.DataFrame([dict(record='fixture',group='g')]).to_csv(r/'private/reference_grouped.csv',index=False)
   pd.DataFrame([dict(samples=33000)]).to_csv(r/'private/record_audit.csv',index=False)
   np.savez(r/'private/cache/fixture.npz',F=np.ones((6,2,12),np.float32),W=np.ones((6,1500),np.float32),y=np.array([0,0,1,0,1,0]))
   with patch.object(data,'ROOT',r):
    *_,t,w=data.load('forecast',1);self.assertEqual(t.tolist(),[0,1,2])
