import numpy as np
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,mean_absolute_error,mean_squared_error
from sklearn.linear_model import LogisticRegression,LinearRegression
from common import *
def classification(y,p,threshold=.5,weights=None):
 y=np.asarray(y);p=np.clip(p,1e-7,1-1e-7);w=np.ones(len(y)) if weights is None else weights
 if len(np.unique(y[w>0]))<2:return None
 pred=p>=threshold
 pos=(y==1);neg=~pos
 try:
  cal=LogisticRegression(C=1e6).fit(logit(p).reshape(-1,1),y,sample_weight=w);inter=float(cal.intercept_[0]);slope=float(cal.coef_[0,0])
 except Exception:inter=None;slope=None
 return dict(AUROC=float(roc_auc_score(y,p,sample_weight=w)),AP=float(average_precision_score(y,p,sample_weight=w)),prevalence=float(np.average(y,weights=w)),Brier=float(np.average((y-p)**2,weights=w)),sensitivity=float(np.average(pred[pos],weights=w[pos])),specificity=float(np.average(~pred[neg],weights=w[neg])),calibration_intercept=inter,calibration_slope=slope)
def regression(y,p):
 c=LinearRegression().fit(np.asarray(p).reshape(-1,1),y)
 return dict(MAE=float(mean_absolute_error(y,p)),RMSE=float(np.sqrt(mean_squared_error(y,p))),bias=float(np.mean(p-y)),calibration_intercept=float(c.intercept_),calibration_slope=float(c.coef_[0]))
def summarize(y,p,g,task,thresholds=None,seed=None):
 unique,gi=np.unique(g,return_inverse=True);rng=np.random.default_rng(CFG['seed'] if seed is None else seed)
 if task=='ahi':
  out=regression(y,p);key='MAE';vals=[]
  for _ in range(CFG['bootstrap']):
   draw=rng.integers(0,len(unique),len(unique));ix=np.concatenate([np.flatnonzero(gi==i) for i in draw]);vals.append(mean_absolute_error(y[ix],p[ix]))
  out['MAE_CI95']=np.quantile(vals,[.025,.975]).tolist();out['severity_thresholds']={str(t):classification((y>=t).astype(int),np.clip((p-t)/100+.5,0,1)) for t in [5,15,30]};out['severity_true_counts']=np.bincount(np.digitize(y,[5,15,30]),minlength=4).tolist()
 else:
  # Saved fold-specific threshold-selected classes; probabilities remain unmodified for discrimination/calibration.
  out=classification(y,p);binary=p>=thresholds if thresholds is not None else p>=.5
  out['sensitivity']=float(binary[y==1].mean());out['specificity']=float((~binary[y==0]).mean())
  counts=np.bincount(gi);weights=1/counts[gi];out['equal_group']=classification(y,p,weights=weights);vals=[]
  for _ in range(CFG['bootstrap']):
   w=np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))[gi]
   if np.unique(y[w>0]).size==2:vals.append(roc_auc_score(y,p,sample_weight=w))
  out['AUROC_CI95']=np.quantile(vals,[.025,.975]).tolist() if vals else None
 out.update(evaluation_rows=len(y),proxy_groups=len(unique),intervals_include_training_variability=False,bootstrap_repetitions=CFG['bootstrap']);return out

def threshold(y,p):
 from sklearn.metrics import roc_curve
 fpr,tpr,t=roc_curve(y,p);finite=np.isfinite(t);ix=np.argmax(np.where(finite,tpr-fpr,-np.inf));return float(t[ix])
def paired(y,p,q,g,task):
 u,gi=np.unique(g,return_inverse=True);rng=np.random.default_rng(CFG['seed']+4);vals=[]
 for _ in range(CFG['bootstrap']):
  draw=rng.integers(0,len(u),len(u));w=np.bincount(draw,minlength=len(u))[gi]
  if task=='ahi':vals.append(np.average(abs(y-p)-abs(y-q),weights=w))
  elif np.unique(y[w>0]).size==2:vals.append(roc_auc_score(y,p,sample_weight=w)-roc_auc_score(y,q,sample_weight=w))
 effect=np.mean(abs(y-p)-abs(y-q)) if task=='ahi' else roc_auc_score(y,p)-roc_auc_score(y,q)
 return dict(effect=float(effect),CI95=np.quantile(vals,[.025,.975]).tolist(),metric='delta_MAE' if task=='ahi' else 'delta_AUROC',paired_proxy_group_bootstrap=True,training_variability_included=False)
