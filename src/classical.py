import time
import numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,HistGradientBoostingClassifier,HistGradientBoostingRegressor
from sklearn.svm import SVC,SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold,GroupKFold
from sklearn.metrics import roc_auc_score,mean_absolute_error
from common import *
from data import load,split_groups
from evaluate import summarize,threshold

def model(name,param=0,reg=False):
 if name in ['linear','pca','clinical','combined','waveform','mean','median']:
  m=Ridge(alpha=[10,100][param]) if reg else LogisticRegression(C=[.1,1][param],max_iter=300)
 elif name=='forest':m=RandomForestRegressor(n_estimators=100,max_depth=[4,8][param],min_samples_leaf=10,random_state=CFG['seed'],n_jobs=CFG['threads']) if reg else RandomForestClassifier(n_estimators=100,max_depth=[4,8][param],min_samples_leaf=30,random_state=CFG['seed'],n_jobs=CFG['threads'])
 elif name=='boost':m=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=[7,15][param],l2_regularization=10,random_state=CFG['seed']) if reg else HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=[7,15][param],l2_regularization=10,random_state=CFG['seed'])
 elif name=='svm':m=SVR(C=[1,10][param]) if reg else SVC(C=[.1,1][param],probability=False,random_state=CFG['seed'])
 elif name=='tree':m=DecisionTreeClassifier(max_depth=3,min_samples_leaf=100,random_state=CFG['seed'])
 else:raise ValueError(name)
 steps=[SimpleImputer(strategy='median',add_indicator=True),StandardScaler()]
 if name=='pca':steps.append(PCA(n_components=5,random_state=CFG['seed']))
 return make_pipeline(*steps,m)
def predict(m,x,reg):return m.predict(x) if reg else m.predict_proba(x)[:,1]
def fit(m,x,y,name,groups=None):
 if name=='svm' and len(y)>4000:
  ix=np.sort(np.random.default_rng(CFG['seed']).choice(len(y),4000,replace=False));x=x[ix];y=y[ix];groups=groups[ix] if groups is not None else None
 if name=='svm' and isinstance(m[-1],SVC):
  from sklearn.calibration import CalibratedClassifierCV
  if groups is None:raise RuntimeError('SVM probability calibration requires training groups')
  cv=list(StratifiedGroupKFold(2,shuffle=True,random_state=CFG['seed']).split(x,y,groups))
  if any(np.unique(y[a]).size<2 or np.unique(y[b]).size<2 for a,b in cv):raise RuntimeError('SVM training-only group calibration lacks support')
  m=CalibratedClassifierCV(m,cv=cv,method='sigmoid')
 m.fit(x,y)
 return m

def ahi_data():
 ref=pd.read_csv(ROOT/'private/reference_grouped.csv');xx=[]
 for _,r in ref.iterrows():
  f=np.load(ROOT/'private/cache'/f'{r.record}.npz')['F'][:,0];f=f[np.isfinite(f).all(1)]
  xx.append(np.r_[np.mean(f,0),np.std(f,0),np.quantile(f,.1,axis=0),np.quantile(f,.9,axis=0)])
 return np.asarray(xx),ref.AHI.values,ref.group.values,ref[['age','sex_numeric','BMI']].values

def run(task,name,duration,deadline):
 tag=f'{task}_{duration}m_{name}';out=ROOT/'results'/f'{tag}.json'
 if out.exists():return
 reg=task=='ahi'
 if reg:
  x,y,g,clinical=ahi_data();r=np.arange(len(y));times=np.zeros(len(y));splits=list(GroupKFold(CFG['outer_folds']).split(x,y,g))
  if name=='clinical':x=clinical
  elif name=='combined':x=np.c_[clinical,x]
  elif name=='waveform':x=x[:,[7,8,9,10,19,20,21,22,31,32,33,34,43,44,45,46]]
 else:
  x,y,g,r,times,_=load(task,duration);splits=split_groups(y,g)
  if duration!=1:
   # Reuse the immutable one-minute outer group assignments, not a new row-dependent split.
   reference=np.load(ROOT/'private/predictions'/f'{task}_1m_linear_OOF.npz')
   mapping={}
   for group in np.unique(reference['group']):
    values=np.unique(reference['fold'][reference['group']==group]);assert len(values)==1;mapping[group]=int(values[0])
   assigned=np.array([mapping[group] for group in g]);splits=[(np.flatnonzero(assigned!=fi),np.flatnonzero(assigned==fi)) for fi in range(CFG['outer_folds'])]
  if name=='waveform':x=x[:,7:11]
  elif name=='rhythm':x=x[:,:7]
  elif name=='pca':pass
 fitname='linear' if name=='rhythm' else name
 allp=np.full(len(y),np.nan);allth=np.full(len(y),np.nan);fold=np.full(len(y),-1);choices=[]
 for fi,(tr,te) in enumerate(splits):
  if stopped(deadline):return
  saved=ROOT/'private/predictions'/f'{tag}_fold{fi}.npz';saved.parent.mkdir(exist_ok=True)
  if saved.exists():
   z=np.load(saved);assert np.array_equal(z['indices'],te);allp[te]=z['p'];allth[te]=z['threshold'];fold[te]=fi;continue
  start=time.time()
  inner=list(GroupKFold(CFG['inner_folds']).split(x[tr],y[tr],g[tr])) if reg else list(StratifiedGroupKFold(CFG['inner_folds'],shuffle=True,random_state=CFG['seed']+fi).split(x[tr],y[tr],g[tr]))
  if fitname in ['mean','median']:
   p=np.full(len(te),np.mean(y[tr]) if fitname=='mean' else np.median(y[tr]));best=0;t=.5
  else:
   candidates=range(1 if fitname=='tree' else 2);bestloss=np.inf;best=0;bestinner=None
   for par in candidates:
    inn=np.full(len(tr),np.nan)
    for a,b in inner:
     m=fit(model(fitname,par,reg),x[tr[a]],y[tr[a]],fitname,g[tr[a]]);inn[b]=predict(m,x[tr[b]],reg)
    loss=mean_absolute_error(y[tr],inn) if reg else -roc_auc_score(y[tr],inn)
    if loss<bestloss:bestloss=loss;best=par;bestinner=inn
   t=.5 if reg else threshold(y[tr],bestinner)
   m=fit(model(fitname,best,reg),x[tr],y[tr],fitname,g[tr]);p=predict(m,x[te],reg)
  allp[te]=p;allth[te]=t;fold[te]=fi
  np.savez_compressed(saved,indices=te,p=p,threshold=t,configuration=best)
  choices.append(dict(fold=fi,configuration=best,train_groups=len(np.unique(g[tr])),evaluation_groups=len(np.unique(g[te])),seconds=time.time()-start))
  atomic(ROOT/'results'/f'{tag}_progress.json',dict(completed_folds=fi+1,total_folds=len(splits),choices=choices));log(f'{tag}: outer fold {fi+1}/3 saved')
 assert np.isfinite(allp).all() and (fold>=0).all()
 np.savez_compressed(ROOT/'private/predictions'/f'{tag}_OOF.npz',y=y,p=allp,group=g,record=r,minute=times,threshold=allth,fold=fold)
 metrics=summarize(y,allp,g,task,allth);atomic(out,dict(task=task,model=name,duration_minutes=duration,validation='3 outer / 2 inner proxy-group nested; all historically exposed',metrics=metrics,folds=choices,selection_outcomes_used=False))
