import numpy as np,pandas as pd,wfdb
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from common import *
from data import features
STATE=[0,1,2,4,5,6,7,8,9,10] # exclude contemporaneous RMSSD and QC fraction

def bouts(y):
 d=np.diff(np.r_[0,y,0]);return list(zip(np.flatnonzero(d==1),np.flatnonzero(d==-1)))
def bounds(a,n,L,g):return (a-g-L,a-g,n+g,n+g+L)
def valid_geometry(y,a,n,L,g):
 ps,pe,qs,qe=bounds(a,n,L,g)
 return ps>=0 and qe<=len(y) and not np.any(y[ps:a]) and not np.any(y[n:qe])
def prepare(deadline):
 ref=pd.read_csv(ROOT/'private/reference_grouped.csv');source=Path(CFG['source'])
 for L,g in CFG['residual_geometry']:
  folder=ROOT/'private'/f'residual_L{L}_g{g}';folder.mkdir(exist_ok=True)
  for ri,row in ref.iterrows():
   p=folder/f'{row.record}.npz'
   if p.exists():continue
   if stopped(deadline):return False
   y=np.load(ROOT/'private/cache'/f'{row.record}.npz')['y'];x=wfdb.rdrecord(str(source/row.record)).p_signal[:,0];y=y[:len(x)//6000];cache={};rows=[];used=set()
   def ff(s,e):
    if (s,e) not in cache:cache[s,e]=features(x[s*6000:e*6000])[0]
    return cache[s,e]
   for a,n in bouts(y):
    if not valid_geometry(y,a,n,L,g):continue
    ps,pe,qs,qe=bounds(a,n,L,g)
    if used.intersection(range(ps,qe)):continue
    pre=ff(ps,pe);post=ff(qs,qe)
    if not np.isfinite(pre).all() or not np.isfinite(post).all():continue
    # Deterministic, within-record, pre-state/time matching; post features do not choose controls.
    candidates=[]
    for b in range(L+g,len(y)-(n-a)-g-L, L):
     nn=b+n-a;s,e,u,v=bounds(b,nn,L,g)
     if used.intersection(range(s,v)) or y[s:v].any() or set(range(ps,qe)).intersection(range(s,v)):continue
     bp=ff(s,e)
     if not np.isfinite(bp).all():continue
     distance=abs(bp[0]-pre[0])/10+abs(bp[7]-pre[7])/max(abs(pre[7]),.1)+abs(b-a)/max(len(y),1);candidates.append((distance,b,nn,bp))
    if not candidates:continue
    _,b,nn,bp=min(candidates,key=lambda c:(c[0],c[1]));s,e,u,v=bounds(b,nn,L,g);bpost=ff(u,v)
    if not np.isfinite(bpost).all():continue # no alternate control chosen using post outcomes
    used.update(range(ps,qe));used.update(range(s,v));rows.append((pre,post,bp,bpost,a,n,b))
   if rows:
    np.savez_compressed(p,pre=np.stack([z[0] for z in rows]),post=np.stack([z[1] for z in rows]),control_pre=np.stack([z[2] for z in rows]),control_post=np.stack([z[3] for z in rows]),a=np.array([z[4] for z in rows]),n=np.array([z[5] for z in rows]),control_a=np.array([z[6] for z in rows]))
   else:np.savez_compressed(p,pre=np.empty((0,12)),post=np.empty((0,12)),control_pre=np.empty((0,12)),control_post=np.empty((0,12)),a=[],n=[],control_a=[])
  log(f'Residual geometry L{L}/g{g}: preparation complete')
 return True

def run(L,g,name,deadline):
 tag=f'residual_L{L}_g{g}_{name}';out=ROOT/'results'/f'{tag}.json'
 if out.exists() or stopped(deadline):return
 ref=pd.read_csv(ROOT/'private/reference_grouped.csv');xs=[];ys=[];cs=[];ds=[];gs=[];dur=[]
 for _,r in ref.iterrows():
  z=np.load(ROOT/'private'/f'residual_L{L}_g{g}'/f'{r.record}.npz');n=len(z['a'])
  if not n:continue
  xs.append(z['pre'][:,STATE]);ys.append(z['post'][:,STATE]);cs.append(z['control_pre'][:,STATE]);ds.append(z['control_post'][:,STATE]);gs.extend([r.group]*n);dur.extend(z['n']-z['a']+2*g)
 if not xs:atomic(out,dict(status='infeasible',reason='no eligible matched events'));return
 x,y,c,d=map(np.concatenate,[xs,ys,cs,ds]);gs=np.array(gs);du=np.asarray(dur);rx=np.full(len(x),np.nan);rc=rx.copy();baseline=rx.copy();normerr=[]
 if len(np.unique(gs))<6:atomic(out,dict(status='infeasible',reason='fewer than six supported proxy groups',episodes=len(x)));return
 def estimator():return make_pipeline(StandardScaler(),Ridge(alpha=10)) if name=='ridge' else make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(16,),activation='relu',alpha=.1,max_iter=300,early_stopping=False,random_state=CFG['seed']))
 for tr,te in GroupKFold(3).split(x,groups=gs):
  if stopped(deadline):return
  mu=d[tr].mean(0);sd=np.maximum(d[tr].std(0),1e-6);inp=np.c_[c,du];real=np.c_[x,du];target=(d-mu)/sd;cf=np.zeros((len(tr),len(STATE)))
  for a,b in GroupKFold(3).split(tr,groups=gs[tr]):
   m=estimator().fit(inp[tr[a]],target[tr[a]]);cf[b]=target[tr[b]]-m.predict(inp[tr[b]])
  center=cf.mean(0);scale=np.maximum(cf.std(0),.05);m=estimator().fit(inp[tr],target[tr]);rr=((y[te]-mu)/sd-m.predict(real[te])-center)/scale;cr=(target[te]-m.predict(inp[te])-center)/scale
  rx[te]=np.sqrt(np.mean(rr*rr,axis=1));rc[te]=np.sqrt(np.mean(cr*cr,axis=1));baseline[te]=np.sqrt(np.mean(((y[te]-x[te])/sd)**2,axis=1))
  pc=PCA(n_components=1).fit((cf-center)/scale);normerr.append(dict(heldout_normal_coordinate_mean=np.mean(cr,0).tolist(),heldout_normal_coordinate_SD=np.std(cr,0).tolist(),normal_training_PC1_variance=float(pc.explained_variance_ratio_[0])))
 diff=rx-rc;u=np.unique(gs);rng=np.random.default_rng(CFG['seed']);means=np.array([diff[gs==a].mean() for a in u]);boot=np.mean(rng.choice(means,(CFG['bootstrap'],len(means)),replace=True),1)
 atomic(out,dict(status='complete',model=name,L_minutes=L,g_minutes=g,matched_episodes=len(x),proxy_groups=len(u),effect_equal_group_real_minus_control_residual_norm=float(means.mean()),CI95=np.quantile(boot,[.025,.975]).tolist(),real_residual_norm_mean=float(rx.mean()),pseudo_residual_norm_mean=float(rc.mean()),ordinary_change_norm_mean=float(baseline.mean()),heldout_normal_calibration=normerr,RMSSD_excluded_from_state=True,true_annotation_defined_explanatory_analysis=True,causal_injury_claim=False,recovery='not completed; distinct fixed-horizon analysis remains pending'))
 np.savez_compressed(ROOT/'private'/f'{tag}.npz',real=rx,control=rc,group=gs,difference=diff)
