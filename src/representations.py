import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score,adjusted_rand_score
from common import *
from data import load,split_groups
from neural import Autoencoder,fit_net,infer,seed
from evaluate import summarize,threshold

def run(kind,deadline):
 tag=f'detect_1m_representation_{kind}'
 if (ROOT/'results'/f'{tag}.json').exists():return
 f,y,g,r,t,x=load('detect',1);p=np.full(len(y),np.nan);th=np.full(len(y),np.nan);cluster_rows=[]
 for fi,(tr,te) in enumerate(split_groups(y,g)):
  if stopped(deadline):return
  fp=ROOT/'private/predictions'/f'{tag}_fold{fi}.npz';cp=ROOT/'private'/f'{tag}_f{fi}_cluster.json'
  if fp.exists():
   z=np.load(fp);p[te]=z['p'];th[te]=z['threshold'];cluster_rows.extend(json.loads(cp.read_text()) if cp.exists() else []);continue
  a,b=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=CFG['seed']+fi).split(x[tr],groups=g[tr]));a=tr[a];b=tr[b];rng=np.random.default_rng(CFG['seed']);a=np.sort(rng.choice(a,min(len(a),6000),replace=False));fitix=np.sort(rng.choice(tr,min(len(tr),6000),replace=False))
  def embedding_train(indices,monitor,stage,epochs=40):
   if kind=='pca_wave':
    pipe=make_pipeline(StandardScaler(),PCA(n_components=16,svd_solver='randomized',random_state=CFG['seed']));pipe.fit(x[indices]);return pipe,None
   seed(CFG['seed']);net=Autoencoder(kind);return fit_net(net,x,np.zeros(len(y)),indices,monitor,f'{tag}_f{fi}_{stage}',deadline,epochs=epochs,reconstruct=True,fixed=stage=='refit')
  net,info=embedding_train(a,b,'screen')
  if net is None:return
  embed=lambda n,info,ix:n.transform(x[ix]) if kind=='pca_wave' else infer(n,x[ix],info,latent=True)
  za=embed(net,info,a);zb=embed(net,info,b);best=-np.inf;chosen=.1
  for C in [.1,1]:
   m=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=300));m.fit(za,y[a]);pb=m.predict_proba(zb)[:,1]
   from sklearn.metrics import roc_auc_score
   val=roc_auc_score(y[b],pb)
   if val>best:best=val;chosen=C;cut=threshold(y[b],pb)
  net,info=embedding_train(fitix,b,'refit',info['selected_epoch'] if info else 40)
  if net is None:return
  ztr=embed(net,info,tr);zte=embed(net,info,te);m=make_pipeline(StandardScaler(),LogisticRegression(C=chosen,max_iter=300));m.fit(ztr,y[tr]);p[te]=m.predict_proba(zte)[:,1];th[te]=cut
  # Clusters see fitting embeddings only; no apnea/AHI labels choose complexity.
  zsmall=ztr[rng.choice(len(tr),min(2000,len(tr)),replace=False)];norm=StandardScaler().fit(zsmall);zz=norm.transform(zsmall);rows=[]
  for method in ['kmeans','gmm']:
   bestsil=-np.inf;selected=None
   for k in [2,3]:
    cl=KMeans(k,n_init=10,random_state=CFG['seed']) if method=='kmeans' else GaussianMixture(k,covariance_type='diag',reg_covar=1e-4,random_state=CFG['seed'])
    lab=cl.fit_predict(zz);support=np.bincount(lab,minlength=k)
    if min(support)<.05*len(lab):continue
    sil=silhouette_score(zz,lab,sample_size=min(1000,len(zz)),random_state=CFG['seed'])
    if sil>bestsil:bestsil=sil;selected=(k,cl,lab,support)
   if selected is None:continue
   k,cl,lab,support=selected;ari=[]
   for s in range(3):
    idx=rng.choice(len(zz),len(zz),replace=True);other=KMeans(k,n_init=10,random_state=CFG['seed']+s+1) if method=='kmeans' else GaussianMixture(k,covariance_type='diag',reg_covar=1e-4,random_state=CFG['seed']+s+1);other.fit(zz[idx]);ari.append(adjusted_rand_score(lab,other.predict(zz)))
   ev=cl.predict(norm.transform(zte));rows.append(dict(fold=fi,method=method,k=k,training_silhouette=float(bestsil),training_support=support.tolist(),bootstrap_refit_ARI=ari,evaluation_support=np.bincount(ev,minlength=k).tolist(),evaluation_proxy_groups_per_cluster=[len(np.unique(g[te][ev==j])) for j in range(k)],evaluated_after_cluster_freeze=True,biological_subtype_claim=False))
   # All membership / example linkage stays private.
   np.savez_compressed(ROOT/'private'/f'{tag}_f{fi}_{method}_clusters.npz',indices=te,cluster=ev,embedding=zte)
  atomic(cp,rows);cluster_rows.extend(rows);np.savez_compressed(fp,p=p[te],threshold=cut)
  log(f'{tag} fold {fi+1}/3 completed')
 np.savez_compressed(ROOT/'private/predictions'/f'{tag}_OOF.npz',y=y,p=p,group=g,record=r,minute=t,threshold=th)
 atomic(ROOT/'results'/f'{tag}.json',dict(task='detect',model=kind,metrics=summarize(y,p,g,'detect',th),clusters=cluster_rows,representation_pretraining_outer_groups_excluded=True,learning_objective='waveform reconstruction or PCA, no apnea/AHI labels',readout='training-only logistic C0.1/1',prototype_clinical_review='pending'))
