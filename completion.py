"""Bounded postprocessing of frozen outputs. No neural training or new model search."""
from pathlib import Path
import os,sys,json,hashlib,csv,textwrap,zipfile,html,traceback,fcntl
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'private/matplotlib'))
sys.path.insert(0,str(ROOT/'src'))
from common import atomic,CFG,sha

def read(p,default=None):
 try:return json.loads(Path(p).read_text())
 except (FileNotFoundError,json.JSONDecodeError):return default

def load_ahi_prediction(path):
 # Legacy local AHI archives stored string groups as NumPy object arrays.
 # Do not unpickle them: reconstruct groups from the immutable recording assignment and verify targets.
 z=np.load(path,allow_pickle=False);d={key:z[key] for key in z.files if key!='group'}
 ref=pd.read_csv(ROOT/'private/reference_grouped.csv');indices=d['record'].astype(int)
 if np.any(indices<0) or np.any(indices>=len(ref)):raise ValueError('AHI recording alignment invalid')
 np.testing.assert_array_equal(d['y'],ref.AHI.to_numpy()[indices]);d['group']=ref.group.astype(str).to_numpy(dtype=str)[indices]
 return d

def alarm_counts(y,p,cut,record,minute):
 total=0;false=0
 for r in np.unique(record):
  ix=np.flatnonzero(record==r);ix=ix[np.argsort(minute[ix])];last=-100;previous=-100
  for i in ix:
   hit=p[i]>=cut[i]
   if hit and (minute[i]!=previous+1) and minute[i]-last>2:
    total+=1;false+=int(y[i]==0);last=minute[i]
   previous=minute[i] if hit else -100
 return {'alarms':total,'false_alarm_starts':false,'eligible_recorded_hours':len(y)/60,'false_alarms_per_eligible_recorded_hour':false/(len(y)/60),'suppression_minutes':2,'physiological_event_onset_accuracy_claimed':False}

def windows_and_alarms():
 from evaluate import paired
 rows=[];alarms=[]
 for task in ['detect','forecast','forecast_transition']:
  for model in ['linear','boost']:
   a=ROOT/'private/predictions'/f'{task}_1m_{model}_OOF.npz';b=ROOT/'private/predictions'/f'{task}_2m_{model}_OOF.npz'
   if a.exists() and b.exists():
    x=np.load(a);z=np.load(b);lookup={(int(r),int(t)):i for i,(r,t) in enumerate(zip(x['record'],x['minute']))};pairs=[(lookup[(int(r),int(t))],j) for j,(r,t) in enumerate(zip(z['record'],z['minute'])) if (int(r),int(t)) in lookup]
    ix,iz=np.array(pairs,int).T
    for key in ['y','group','fold']:np.testing.assert_array_equal(x[key][ix],z[key][iz])
    effect=paired(x['y'][ix],z['p'][iz],x['p'][ix],x['group'][ix],task);rows.append({'task':task,'model':model,'common_rows':len(ix),'proxy_groups':len(np.unique(x['group'][ix])),'comparison':'2minutes minus1minute','group_folds_identical':True,**effect})
 for p in (ROOT/'private/predictions').glob('detect_*_OOF.npz'):
  z=np.load(p)
  if {'threshold','minute','record'}<=set(z.files):alarms.append({'experiment':p.stem,**alarm_counts(z['y'],z['p'],np.broadcast_to(z['threshold'],z['p'].shape),z['record'],z['minute'])})
 atomic(ROOT/'results/COMMON_ELIGIBLE_WINDOW_COMPARISONS.json',rows);pd.DataFrame(alarms).to_csv(ROOT/'results/ALARM_SUMMARIES.csv',index=False)

def severity_and_flow():
 from sklearn.metrics import confusion_matrix
 from evaluate import paired
 rows=[];pairs=[];rng=np.random.default_rng(207043)
 for p in (ROOT/'private/predictions').glob('ahi_1m_*_OOF.npz'):
  z=load_ahi_prediction(p);y=z['y'];pred=z['p'];g=z['group'];u,gi=np.unique(g,return_inverse=True)
  for threshold_value in [5,15,30]:
   target=y>=threshold_value;positive=pred>=threshold_value;npos=int(target.sum());nneg=int((~target).sum());sens=float(positive[target].mean()) if npos else None;spec=float((~positive[~target]).mean()) if nneg else None;bs=[];bn=[]
   for _ in range(500):
    weights=np.bincount(rng.integers(0,len(u),len(u)),minlength=len(u))[gi]
    if weights[target].sum()>0:bs.append(float(np.average(positive[target],weights=weights[target])))
    if weights[~target].sum()>0:bn.append(float(np.average(~positive[~target],weights=weights[~target])))
   rows.append({'experiment':p.stem,'threshold_events_per_hour':threshold_value,'reference_positive_recordings':npos,'reference_negative_recordings':nneg,'positive_proxy_groups':len(np.unique(g[target])),'negative_proxy_groups':len(np.unique(g[~target])),'sensitivity':sens,'specificity':spec,'sensitivity_CI95':np.quantile(bs,[.025,.975]).tolist() if bs else None,'specificity_CI95':np.quantile(bn,[.025,.975]).tolist() if bn else None,'four_category_confusion':confusion_matrix(np.digitize(y,[5,15,30]),np.digitize(pred,[5,15,30]),labels=[0,1,2,3]).tolist(),'calibrated_severity_probability_claim':False})
 for model,reference in [('linear','clinical'),('combined','clinical'),('combined','linear')]:
  a=ROOT/'private/predictions'/f'ahi_1m_{model}_OOF.npz';b=ROOT/'private/predictions'/f'ahi_1m_{reference}_OOF.npz'
  if a.exists() and b.exists():
   x=load_ahi_prediction(a);z=load_ahi_prediction(b);np.testing.assert_array_equal(x['y'],z['y']);np.testing.assert_array_equal(x['group'],z['group']);pairs.append({'comparison':model+' minus '+reference,**paired(x['y'],x['p'],z['p'],x['group'],'ahi')})
 atomic(ROOT/'results/SEVERITY_THRESHOLD_RESULTS.json',rows);atomic(ROOT/'results/PAIRED_AHI_COMPARISONS.json',pairs)
 # Do not interpret the original artificial (AHI-threshold)/100+.5 transform as a probability.
 atomic(ROOT/'results/SEVERITY_REPORTING_CORRECTION.json',{'original_records_preserved':True,'superseding_results':'SEVERITY_THRESHOLD_RESULTS.json','reason':'Original severity helper used an arbitrary monotone transform of AHI; its Brier/calibration fields are not validated probabilities. Report corrected hard-threshold metrics, raw-AHI discrimination only if specified, and no calibrated severity probability claims.'})
 ref=pd.read_csv(ROOT/'private/reference_grouped.csv');audit=pd.read_csv(ROOT/'private/record_audit.csv');flow=[]
 for j,L in enumerate(CFG['durations_minutes']):
  assigned=valid=partial=short=other=0;excluded_positive=eligible_positive=0
  for i,r in ref.iterrows():
   z=np.load(ROOT/'private/cache'/f'{r.record}.npz');f=z['F'][:,j];y=z['y'];ok=np.isfinite(f).all(1);t=np.arange(len(y));first=t+1<L;last=(t+1)*6000>int(audit.iloc[i]['samples']);bad=~ok&~first&~last
   assigned+=len(y);valid+=int(ok.sum());partial+=int(last.sum());short+=int(first.sum());other+=int(bad.sum());excluded_positive+=int(y[~ok].sum());eligible_positive+=int(y[ok].sum())
  flow.append({'lookback_minutes':L,'annotated_minutes':assigned,'eligible_windows':valid,'incomplete_final_observation':partial,'insufficient_lookback':short,'beat_or_feature_QC_failure':other,'eligible_A_minutes':eligible_positive,'excluded_A_minutes':excluded_positive,'QC_reason_subtypes_not_separately_logged':True})
 atomic(ROOT/'results/QC_FLOW_DETAIL.json',flow)

def metric_uncertainty():
 from sklearn.metrics import average_precision_score
 from scipy.special import expit,logit
 rows=[];rng=np.random.default_rng(207044)
 for path in sorted((ROOT/'private/predictions').glob('*_OOF.npz')):
  reg=path.name.startswith('ahi_');z=load_ahi_prediction(path) if reg else np.load(path);y=z['y'];p=z['p'];g=z['group'];u,gi=np.unique(g,return_inverse=True);values={}
  def add(key,v):
   if np.isfinite(v):values.setdefault(key,[]).append(float(v))
  if not reg:
   cut=np.broadcast_to(z['threshold'],p.shape);positive=p>=cut;lp=logit(np.clip(p,1e-7,1-1e-7));X=np.c_[np.ones(len(y)),lp]
  for _ in range(500):
   w=np.bincount(rng.integers(0,len(u),len(u)),minlength=len(u))[gi].astype(float)
   if reg:
    error=p-y;add('RMSE',np.sqrt(np.average(error**2,weights=w)));add('bias',np.average(error,weights=w));mp=np.average(p,weights=w);my=np.average(y,weights=w);den=np.average((p-mp)**2,weights=w)
    if den>1e-12:
     slope=np.average((p-mp)*(y-my),weights=w)/den;add('calibration_slope',slope);add('calibration_intercept',my-slope*mp)
   elif len(np.unique(y[w>0]))==2:
    add('AP',average_precision_score(y,p,sample_weight=w));add('Brier',np.average((y-p)**2,weights=w));add('sensitivity',np.average(positive[y==1],weights=w[y==1]));add('specificity',np.average(~positive[y==0],weights=w[y==0]))
    # Evaluation-only calibration fit; never used to change the model probabilities.
    beta=np.array([0.,1.]);valid=True
    for step in range(25):
     prob=expit(X@beta);v=np.maximum(prob*(1-prob),1e-10)*w;H=X.T@(X*v[:,None]);gradient=X.T@(w*(y-prob))
     try:delta=np.linalg.solve(H+np.eye(2)*1e-8,gradient)
     except np.linalg.LinAlgError:valid=False;break
     delta=np.clip(delta,-5,5);beta+=delta
     if np.max(abs(delta))<1e-6:break
    else:valid=False
    if valid:add('calibration_intercept',beta[0]);add('calibration_slope',beta[1])
  rows.append({'experiment':path.stem,'group_bootstrap_repetitions':500,'groups':len(u),'intervals_include_training_variability':False,'CI95':{k:np.quantile(v,[.025,.975]).tolist() for k,v in values.items()},'valid_bootstrap_counts':{k:len(v) for k,v in values.items()},'calibration_fit_is_evaluation_only':True})
 atomic(ROOT/'results/ADDITIONAL_METRIC_UNCERTAINTY.json',rows)

def residual_stats():
 rows=[];keys_by_geometry={};values_by_geometry={};rng=np.random.default_rng(207041);ref=pd.read_csv(ROOT/'private/reference_grouped.csv')
 for L,g in CFG['residual_geometry']:
  folder=ROOT/'private'/f'residual_L{L}_g{g}';keys=[]
  if folder.exists():
   for _,r in ref.iterrows():
    p=folder/f'{r.record}.npz'
    if p.exists():
     z=np.load(p);keys.extend([(r.record,int(a),int(n)) for a,n in zip(z['a'],z['n'])])
  for model in ['ridge','mlp']:
   tag=f'residual_L{L}_g{g}_{model}';p=ROOT/'private'/f'{tag}.npz';receipt=read(ROOT/'results'/f'{tag}.json',{})
   if not p.exists():continue
   z=np.load(p);means=np.array([z['difference'][z['group']==v].mean() for v in np.unique(z['group'])]);obs=abs(means.mean());sim=(rng.choice([-1,1],size=(10000,len(means)))*means).mean(1);pv=float((1+(abs(sim)>=obs).sum())/10001)
   rows.append({'experiment':tag,'L':L,'g':g,'model':model,'episodes':len(z['real']),'proxy_groups':len(means),'equal_group_difference':float(means.mean()),'CI95':receipt.get('CI95'),'sign_flip_p':pv,'interpretation':'apnea-associated deviation, not injury'})
   if model=='ridge' and len(keys)==len(z['real']):keys_by_geometry[(L,g)]=keys;values_by_geometry[(L,g)]={k:(float(v),str(gr)) for k,v,gr in zip(keys,z['difference'],z['group'])}
 if rows:
  order=np.argsort([x['sign_flip_p'] for x in rows]);last=0
  for rank,i in enumerate(order):last=max(last,min(1,(len(rows)-rank)*rows[i]['sign_flip_p']));rows[i]['Holm_p_residual_family']=last
 atomic(ROOT/'results/RESIDUAL_MULTIPLICITY.json',rows)
 common=set.intersection(*(set(k) for k in keys_by_geometry.values())) if len(keys_by_geometry)==4 else set();commonrows=[]
 for geo,mapping in values_by_geometry.items():
  if not common:continue
  vals=np.array([mapping[k][0] for k in sorted(common)]);gr=np.array([mapping[k][1] for k in sorted(common)]);means=np.array([vals[gr==a].mean() for a in np.unique(gr)]);out={'L':geo[0],'g':geo[1],'common_matched_episodes':len(common),'proxy_groups':len(means),'equal_group_difference':float(means.mean()),'inferential_support':len(means)>=6}
  if len(means)>=6:out['CI95']=np.quantile(rng.choice(means,(500,len(means)),replace=True).mean(1),[.025,.975]).tolist()
  commonrows.append(out)
 atomic(ROOT/'results/COMMON_EPISODE_RESIDUAL_GEOMETRY.json',{'all_four_geometries_available':len(keys_by_geometry)==4,'common_episodes':len(common),'comparisons':commonrows})

def recovery():
 from data import features
 from residual import STATE
 from sklearn.model_selection import GroupKFold
 from sklearn.pipeline import make_pipeline
 from sklearn.preprocessing import StandardScaler
 from sklearn.linear_model import Ridge
 import wfdb
 ref=pd.read_csv(ROOT/'private/reference_grouped.csv');folder=ROOT/'private/residual_L2_g1'
 if not folder.exists():atomic(ROOT/'results/RECOVERY.json',{'status':'pending','reason':'primary matched geometry not yet prepared'});return
 rows=[];rng=np.random.default_rng(207042)
 for horizon in [0,2,4]:
  xx=[];yy=[];cc=[];dd=[];groups=[];dur=[];ident=[];record_count=0
  for _,r in ref.iterrows():
   p=folder/f'{r.record}.npz'
   if not p.exists():continue
   z=np.load(p);ann=np.load(ROOT/'private/cache'/f'{r.record}.npz')['y'];wave=None
   for j,(a,n,b) in enumerate(zip(z['a'],z['n'],z['control_a'])):
    a=int(a);n=int(n);b=int(b);nn=b+n-a;ps=n+1+horizon;pe=ps+2;cs=nn+1+horizon;ce=cs+2
    if pe>len(ann) or ce>len(ann) or ann[n:pe].any() or ann[b:ce].any():continue
    if horizon==0:post=z['post'][j];control=z['control_post'][j]
    else:
     if wave is None:wave=wfdb.rdrecord(str(Path(CFG['source'])/r.record)).p_signal[:,0]
     if pe*6000>len(wave) or ce*6000>len(wave):continue
     post=features(wave[ps*6000:pe*6000])[0];control=features(wave[cs*6000:ce*6000])[0]
    if not np.isfinite(post).all() or not np.isfinite(control).all():continue
    xx.append(z['pre'][j][STATE]);yy.append(post[STATE]);cc.append(z['control_pre'][j][STATE]);dd.append(control[STATE]);groups.append(r.group);dur.append(n-a+2+horizon);ident.append((r.record,a,n))
  if not xx or len(set(groups))<6:rows.append({'horizon_minutes':horizon,'eligible_matched_episodes':len(xx),'proxy_groups':len(set(groups)),'status':'insufficient_support'});continue
  x,y,c,d=map(np.asarray,[xx,yy,cc,dd]);gs=np.asarray(groups);du=np.asarray(dur);real=np.full(len(x),np.nan);ctrl=real.copy()
  def est():return make_pipeline(StandardScaler(),Ridge(alpha=10))
  for tr,te in GroupKFold(3).split(x,groups=gs):
   mu=d[tr].mean(0);sd=np.maximum(d[tr].std(0),1e-6);target=(d-mu)/sd;inp=np.c_[c,du];rx=np.c_[x,du];cf=np.zeros((len(tr),len(STATE)))
   for a,b in GroupKFold(3).split(tr,groups=gs[tr]):m=est().fit(inp[tr[a]],target[tr[a]]);cf[b]=target[tr[b]]-m.predict(inp[tr[b]])
   center=cf.mean(0);scale=np.maximum(cf.std(0),.05);m=est().fit(inp[tr],target[tr]);rr=((y[te]-mu)/sd-m.predict(rx[te])-center)/scale;cr=(target[te]-m.predict(inp[te])-center)/scale;real[te]=np.sqrt((rr*rr).mean(1));ctrl[te]=np.sqrt((cr*cr).mean(1))
  if horizon==0:
   baseline=ROOT/'private/residual_L2_g1_ridge.npz'
   if baseline.exists():
    old=np.load(baseline);np.testing.assert_allclose(real,old['real'],rtol=1e-7,atol=1e-7);np.testing.assert_allclose(ctrl,old['control'],rtol=1e-7,atol=1e-7)
  diff=real-ctrl;means=np.array([diff[gs==v].mean() for v in np.unique(gs)]);boot=rng.choice(means,(500,len(means)),replace=True).mean(1)
  rows.append({'horizon_minutes':horizon,'eligible_matched_episodes':len(x),'proxy_groups':len(means),'equal_group_real_minus_pseudo_norm':float(means.mean()),'CI95':np.quantile(boot,[.025,.975]).tolist(),'status':'complete','offset_zero_primary_agreement_checked':horizon==0,'interpretation':'changing eligible population; not a causal recovery trajectory'})
  np.savez_compressed(ROOT/'private'/f'recovery_h{horizon}.npz',real=real,control=ctrl,group=gs)
 atomic(ROOT/'results/RECOVERY.json',{'status':'complete_secondary','models':'ridge only, normal-transition training groups only','results':rows,'causal_recovery_claim':False})

def clusters():
 from data import load
 from sklearn.metrics import normalized_mutual_info_score
 ref=pd.read_csv(ROOT/'private/reference_grouped.csv');f,y,g,record,minute,w=load('detect',1);rows=[];examples=[]
 for p in sorted((ROOT/'private').glob('detect_1m_representation_*_f*_clusters.npz')):
  z=np.load(p);ix=z['indices'];lab=z['cluster'];latent=z['embedding']
  for k in np.unique(lab):
   sub=lab==k;ii=ix[sub];rec=record[ii];counts=np.unique(rec,return_counts=True)[1];row={'representation_file':p.stem,'cluster':int(k),'minutes':int(sub.sum()),'recordings':len(counts),'proxy_groups':len(np.unique(g[ii])),'largest_record_fraction':float(counts.max()/counts.sum()),'apnea_minute_fraction_descriptive':float(y[ii].mean()),'mean_HR':float(f[ii,0].mean()),'mean_rejected_RR_fraction':float(f[ii,11].mean()),'mean_age_record_weighted':float(ref.iloc[rec].age.mean()),'mean_BMI_record_weighted':float(ref.iloc[rec].BMI.mean()),'evaluation_only_after_cluster_freeze':True,'clinical_phenotype_claim':False};rows.append(row)
   if '_f0_kmeans_clusters' in p.stem and len(examples)<12:
    zz=latent[sub];j=np.flatnonzero(sub)[np.argmin(((zz-zz.mean(0))**2).sum(1))];i=ix[j];examples.append({'record_index':int(record[i]),'minute':int(minute[i]),'cluster':int(k),'representation':p.stem})
  rows.append({'representation_file':p.stem,'cluster':'all','record_identity_NMI':float(normalized_mutual_info_score(record[ix],lab)),'heart_rate_quartile_NMI':float(normalized_mutual_info_score(np.digitize(f[ix,0],np.quantile(f[ix,0],[.25,.5,.75])),lab))})
 atomic(ROOT/'private/CLUSTER_CONTEXT_FULL.json',rows)
 public_rows=[]
 for original in rows:
  row=original.copy()
  if row.get('proxy_groups',5)<5:
   for key in ['apnea_minute_fraction_descriptive','mean_HR','mean_rejected_RR_fraction','mean_age_record_weighted','mean_BMI_record_weighted']:row[key]=None
   row['small_group_descriptive_cells_suppressed']=True
  public_rows.append(row)
 pd.DataFrame(public_rows).to_csv(ROOT/'results/CLUSTER_CONTEXT.csv',index=False)
 if examples:
  import wfdb
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  from matplotlib.backends.backend_pdf import PdfPages
  with PdfPages(ROOT/'private/REPRESENTATIVE_ECGS_LOCAL_ONLY.pdf') as pdf:
   for i,e in enumerate(examples):
    name=ref.iloc[e['record_index']].record;start=e['minute']*6000+2500;wave=wfdb.rdrecord(str(Path(CFG['source'])/name),sampfrom=start,sampto=start+1000).p_signal[:,0]
    fig,ax=plt.subplots(figsize=(11,3));ax.plot(np.arange(1000)/100,wave,lw=.7);ax.set(title=f'Example {i+1}: actual ECG; outcome-blind illustrative selection',xlabel='Seconds',ylabel='ECG (mV)');fig.tight_layout();pdf.savefig(fig);plt.close(fig)
  atomic(ROOT/'private/REPRESENTATIVE_ECG_KEY.json',examples)
 atomic(ROOT/'results/PROTOTYPE_REVIEW_STATUS.json',{'actual_examples_prepared':len(examples),'clinical_reviews_completed':0,'echo_or_apnea_outcomes_used_to_choose_examples':False,'example_signals_public':False,'novel_phenotype_established':False})

def supplemental_figures():
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 exposure=[]
 for p in sorted((ROOT/'results').glob('*_learning.json')):
  d=read(p,{});history=d.get('history',[])
  if history:exposure.append({'experiment':p.stem,'epochs_completed':len(history),'done':d.get('done',False),'parameters':d.get('parameters'),'device':d.get('device'),'optimizer_updates':d.get('updates'),'examples_processed':d.get('examples'),'seconds':sum(v['seconds'] for v in history),'final_training_loss':history[-1]['training_loss'],'final_monitor_loss':history[-1]['validation_loss']})
 pd.DataFrame(exposure).to_csv(ROOT/'results/TRAINING_EXPOSURE.csv',index=False)
 windows=read(ROOT/'results/COMMON_ELIGIBLE_WINDOW_COMPARISONS.json',[])
 if windows:
  fig,ax=plt.subplots(figsize=(8,4));v=np.array([x['effect'] for x in windows]);lo=np.array([x['CI95'][0] for x in windows]);hi=np.array([x['CI95'][1] for x in windows]);ax.errorbar(v,np.arange(len(v)),xerr=[np.maximum(v-lo,0),np.maximum(hi-v,0)],fmt='o',capsize=3);ax.axvline(0,color='gray');ax.set_yticks(np.arange(len(v)),[x['task']+' / '+x['model'] for x in windows]);ax.set_xlabel('Delta AUROC:2-minute minus1-minute, common eligibility');fig.tight_layout()
  for ext in ['png','pdf']:fig.savefig(ROOT/'figures'/f'WINDOW_COMPARISON.{ext}',dpi=300)
  plt.close(fig)
 flow=read(ROOT/'results/COHORT_FLOW.json',[]);audit=read(ROOT/'results/DATA_AUDIT.json',{})
 if flow:
  fig,ax=plt.subplots(figsize=(8,5));ax.axis('off');labels=[f"{audit.get('recordings',0)} source recordings; {audit.get('proxy_groups',0)} proxy groups\nNot verified independent patients",f"{audit.get('minutes',0)} minute annotations\n{audit.get('incomplete_final_minutes_excluded',0)} partial final observation windows excluded"]
  labels += [f"{x['task']}: {x['eligible_windows']} eligible windows\n{x['positive_minutes']} positive labels" for x in flow if x['L']==1]
  for i,label in enumerate(labels):
   yy=.92-i*.19;ax.text(.5,yy,label,ha='center',va='center',bbox=dict(boxstyle='round',facecolor='#edf3f8',edgecolor='#567'))
   if i<len(labels)-1:ax.annotate('',xy=(.5,yy-.125),xytext=(.5,yy-.065),arrowprops={'arrowstyle':'->'})
  fig.tight_layout()
  for ext in ['png','pdf']:fig.savefig(ROOT/'figures'/f'COHORT_FLOW.{ext}',dpi=300)
  plt.close(fig)
 seedrows=[]
 for family in ['cnn','resnet']:
  mainchoices=[]
  for fold in range(3):
   val=[]
   for par in [0,1]:
    d=read(ROOT/'results'/f'detect_1m_neural_{family}_seed207031_f{fold}_screen{par}_learning.json',{});h=d.get('history',[]);val.append(min((v['validation_loss'] for v in h),default=float('inf')))
   mainchoices.append(int(np.argmin(val)) if min(val)<float('inf') else None)
  for seed in CFG['neural']['finalist_seeds']:
   d=read(ROOT/'results'/f'detect_1m_neural_{family}_seed{seed}.json',{})
   if d:seedrows.append({'family':family,'seed':seed,'AUROC':d['metrics']['AUROC'],'CI95':d['metrics']['AUROC_CI95'],'main_seed_selected_configuration_by_fold':mainchoices,'additional_seeds_fixed_configuration':0,'all_seeds_hyperparameters_match':all(v==0 for v in mainchoices),'main_vs_repeat_search_budget_equal':False})
 atomic(ROOT/'results/SEED_COMPARISON_AUDIT.json',seedrows)

def manuscript_and_package():
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.backends.backend_pdf import PdfPages
 q=read(ROOT/'results/EXPERIMENT_REGISTRY.json',[]);pending=[x['id'] for x in q if x.get('status') in ['pending','paused','running']];errors=[x['id'] for x in q if x.get('status')=='error'];core_done=bool(q) and not pending and not errors
 limitations=['True patient identities unavailable; validation uses20conservative proxy groups, with historical exposure.','External evaluation blocked by missing PSG release/restricted DNS and incomplete task-specific clock/stage/AHI gates.','Respiratory/SpO2 companion signals absent locally; no physiology comparison performed.','Repeated-seed main configuration can differ from fixed-config repeats; matched three-seed estimate not established.','Independent clinical interpretation has not been performed.']
 infeasible=[]
 for path in (ROOT/'results').glob('*.json'):
  result=read(path,{})
  if isinstance(result,dict) and result.get('status')=='infeasible':infeasible.append({'investigation':path.stem,'reason':result.get('reason','documented in result')})
 limitations.extend(v['investigation']+': '+v['reason'] for v in infeasible)
 post=read(ROOT/'results/POSTPROCESSING_STATUS.json',[]);post_ok=len(post)==7 and all(v['status']=='complete' for v in post)
 assessment={'status':'completed_development_queue_with_documented_limitations' if core_done and post_ok else 'incomplete','core_queue_complete':core_done,'postprocessing_complete':post_ok,'postprocessing':post,'complete_entries':sum(v.get('status')=='complete' for v in q),'pending':pending,'errors':errors,'specific_limitations':limitations,'external_confirmation_established':False,'new_phenotype_established':False}
 atomic(ROOT/'results/COMPLETION_ASSESSMENT.json',assessment)
 from report import report
 report()
 body=(ROOT/'manuscript/MANUSCRIPT.md').read_text();label='Completed development report.' if core_done and post_ok else 'Interim manuscript draft—registered investigations remain incomplete.'
 body=body.replace('Interim development report—experiments in progress.',label)
 # Replace raw fixed-width metric dumps with readable manuscript tables; full values stay in CSV/JSON.
 table_path=ROOT/'results/COMPLETED_RESULTS.csv'
 if table_path.exists():
  try:table=pd.read_csv(table_path)
  except pd.errors.EmptyDataError:table=pd.DataFrame()
  oldtable=table.to_string(index=False)
  body=body.replace(oldtable,'Complete prespecified model results are provided in COMPLETED_RESULTS.csv. No winner was chosen using these outer estimates.')
 extra=['\n## Final investigation reconciliation\n',f"Completed registry entries: {assessment['complete_entries']} of {len(q)}. Core queue complete: {core_done}. Postprocessing complete: {post_ok}. Status: {assessment['status']}.",'\n'.join('- '+v for v in limitations),'\n## Results of the fixed comparisons\n']
 # Model order is prespecified/alphabetical, never rank by best observed result.
 result_rows=[]
 for p in sorted((ROOT/'results').glob('*.json')):
  v=read(p,{})
  if isinstance(v,dict) and 'metrics' in v:
   m=v['metrics'];key='AUROC' if 'AUROC' in m else 'MAE';ci=m.get(key+'_CI95');effect=f"{m[key]:.3f} ({ci[0]:.3f}–{ci[1]:.3f})" if ci else f"{m[key]:.3f}"
   result_rows.append(f"| {p.stem} | {m.get('evaluation_rows','')} | {m.get('proxy_groups','')} | {key} {effect} |")
 extra+=['| Model/task configuration | Evaluation rows | Proxy groups | Estimate (95% CI) |','|---|---:|---:|---|',*result_rows]
 primary=read(ROOT/'results/PAIRED_PRIMARY_COMPARISONS.json',[])
 if primary:
  for v in primary:
   lo,hi=v['CI95'];interpret='The interval spans zero; superiority or equivalence is not established.' if lo<=0<=hi else 'This conditional estimate requires the stated exploration and grouping caveats.'
   extra.append(f"{v['task']}: boosting minus logistic AUROC {v['effect']:+.3f} (95% CI {lo:+.3f} to {hi:+.3f}). {interpret}")
 extra.append('\n### Common-eligible window sensitivities\n| Task / model | Common rows | Delta AUROC, 2min−1min (95% CI) |\n|---|---:|---|')
 for v in read(ROOT/'results/COMMON_ELIGIBLE_WINDOW_COMPARISONS.json',[]):extra.append(f"| {v['task']} / {v['model']} | {v['common_rows']} | {v['effect']:+.3f} ({v['CI95'][0]:+.3f} to {v['CI95'][1]:+.3f}) |")
 extra.append('\n### AHI incremental value\nHard severity-threshold metrics supersede any artificial probability-transformed severity Brier/calibration fields in the original result JSONs. See SEVERITY_THRESHOLD_RESULTS.json.')
 for v in read(ROOT/'results/PAIRED_AHI_COMPARISONS.json',[]):extra.append(f"AHI {v['comparison']}: paired delta MAE {v['effect']:+.3f} events/hour (95% CI {v['CI95'][0]:+.3f} to {v['CI95'][1]:+.3f}).")
 extra.append('\n### Conditional normal-transition deviations\n| Configuration | Matched episodes | Proxy groups | Real−pseudo norm (95% CI) | Holm P |\n|---|---:|---:|---|---:|')
 for v in read(ROOT/'results/RESIDUAL_MULTIPLICITY.json',[]):
  ci=v.get('CI95');txt=f"{v['equal_group_difference']:+.3f} ({ci[0]:+.3f} to {ci[1]:+.3f})" if ci else str(v['equal_group_difference'])
  extra.append(f"| {v['experiment']} | {v['episodes']} | {v['proxy_groups']} | {txt} | {v['Holm_p_residual_family']:.4f} |")
 rec=read(ROOT/'results/RECOVERY.json',{})
 extra.append('\n### Fixed secondary recovery comparison\nHorizon-specific eligibility changes; these are not causal recovery-time estimates.')
 for v in rec.get('results',[]):extra.append(f"Offset {v['horizon_minutes']} min: {v.get('eligible_matched_episodes',0)} matched episodes, {v.get('proxy_groups',0)} proxy groups; estimate {v.get('equal_group_real_minus_pseudo_norm','not estimable')}, CI {v.get('CI95','not estimable')}.")
 extra.append('\n### Reconstruction versus downstream prediction\nReconstruction accuracy is audited separately from apnea readout discrimination; global error/correlation does not establish clinically faithful morphology.')
 for v in read(ROOT/'results/RECONSTRUCTION_AUDIT.json',[]):
  correlation_text=f"{v['median_per_window_temporal_Pearson']:.3f}" if v.get('median_per_window_temporal_Pearson') is not None else 'not estimable (constant reconstruction)'
  extra.append(f"{v['representation']}: held-out equal-group MSE {v['equal_group_reconstruction_MSE_mV_squared']:.5g} mV²; ratio to training-mean-wave baseline {v['relative_MSE_to_training_mean_wave']:.3f}; median temporal correlation {correlation_text}.")
 proto=read(ROOT/'results/PROTOTYPE_REVIEW_STATUS.json',{});extra.append(f"Actual ECG examples prepared for local review: {proto.get('actual_examples_prepared',0)}; independent clinical reviews: {proto.get('clinical_reviews_completed',0)}. No biological subtype is established. Cluster context and identity concentration are in CLUSTER_CONTEXT.csv; fold-local cluster labels are not independently transferable phenotype definitions.")
 extra.append('External confirmation: not completed. The documented missing PSG/network and clinical-harmonization requirements remain. Neither internal cross-validation nor software tests substitute for external validation.')
 extra.append('''\n## Focused related work and interpretation
Early single-lead ECG approaches evaluated apnea screening using rhythm and ECG-derived information (de Chazal et al.,2003; DOI10.1109/TBME.2003.812203). Later CNN–recurrent models illustrate waveform learning, but reported benchmark performance is not directly comparable across patient grouping, exposure and preprocessing (Zhang et al.,2021; DOI10.1155/2021/5594733). Yeh et al. (2022; DOI10.3390/s22020510) specifically examined subject-dependent versus subject-independent subband evaluation. Its published32-subject provenance motivates a remaining identity-mapping audit;20demographic/content proxy groups are not verified subject IDs. We preserve the fixed splits but do not call their results proven patient-independent validation. Our25Hz inputs also cannot test high-frequency subband hypotheses above12.5Hz.
Verified source links: https://pmc.ncbi.nlm.nih.gov/articles/PMC8009718/ ; https://pmc.ncbi.nlm.nih.gov/articles/PMC8777653/ ; original2003paper citation is provider/literature-confirmed, publisher full text not accessed. No numerical superiority to these different protocols is claimed.
''')
 body+='\n'.join(extra);(ROOT/'manuscript/FINAL_MANUSCRIPT.md').write_text(body);(ROOT/'manuscript/FINAL_MANUSCRIPT.html').write_text('<!doctype html><meta charset="utf-8"><pre style="white-space:pre-wrap">'+html.escape(body)+'</pre>')
 with PdfPages(ROOT/'manuscript/FINAL_MANUSCRIPT.pdf') as pdf:
  lines=[]
  for line in body.splitlines():lines.extend(textwrap.wrap(line,100) or [''])
  for i in range(0,len(lines),58):
   fig=plt.figure(figsize=(8.27,11.69));fig.text(.07,.95,'\n'.join(lines[i:i+58]),fontsize=8,va='top');pdf.savefig(fig);plt.close(fig)
 from xml.sax.saxutils import escape
 paras=''.join('<w:p><w:r><w:t xml:space="preserve">'+escape(s)+'</w:t></w:r></w:p>' for s in body.splitlines())
 with zipfile.ZipFile(ROOT/'manuscript/FINAL_MANUSCRIPT.docx','w',zipfile.ZIP_DEFLATED) as z:
  z.writestr('[Content_Types].xml','<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
  z.writestr('_rels/.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
  z.writestr('word/document.xml','<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'+paras+'</w:body></w:document>')
 # Supplemental figures come solely from completed outputs.
 for name,xkey,ykey in [('RECOVERY','horizon_minutes','equal_group_real_minus_pseudo_norm'),('RESIDUAL_MULTIPLICITY','experiment','equal_group_difference')]:
  d=read(ROOT/'results'/f'{name}.json',[]);d=d.get('results',[]) if isinstance(d,dict) else d;d=[v for v in d if ykey in v]
  if not d:continue
  fig,ax=plt.subplots(figsize=(9,4));xx=np.arange(len(d));vals=np.array([v[ykey] for v in d]);lo=np.array([v.get('CI95',[v[ykey],v[ykey]])[0] for v in d]);hi=np.array([v.get('CI95',[v[ykey],v[ykey]])[1] for v in d]);ax.errorbar(xx,vals,yerr=[np.maximum(vals-lo,0),np.maximum(hi-vals,0)],fmt='o',capsize=3);ax.axhline(0,color='gray');ax.set_xticks(xx,[str(v[xkey]) for v in d],rotation=30,ha='right');ax.set_ylabel('Equal-group real minus pseudo residual norm');fig.tight_layout()
  for ext in ['png','pdf']:fig.savefig(ROOT/'figures'/f'{name}.{ext}',dpi=300)
  plt.close(fig)
 allowed={'.py','.md','.json','.csv','.pdf','.png','.html','.docx','.txt','.sh'};files=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and p.suffix in allowed and not set(p.relative_to(ROOT).parts)&{'private','checkpoints_private','corrections','__pycache__','logs'} and p.name!='AGGREGATE_PACKAGE_MANIFEST.json']
 manifest={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files};atomic(ROOT/'AGGREGATE_PACKAGE_MANIFEST.json',manifest);files.append(ROOT/'AGGREGATE_PACKAGE_MANIFEST.json')
 with zipfile.ZipFile(ROOT.parent/'OSA_ECG_Multimethod_v1_Review_Package.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in files:z.write(p,'osa_ecg_multimethod_v1/'+str(p.relative_to(ROOT)))

if __name__=='__main__':
 command=sys.argv[1] if len(sys.argv)>1 else 'prepare'
 if command=='prepare':
  from external_prepare import prepare
  prepare()
 elif command=='finalize':
  from control import held
  if held('RUNNER.lock'):raise SystemExit('Training runner remains active; no concurrent finalization')
  lock=read(ROOT/'private/COMPLETION_CODE_LOCK.json',{})
  if not lock or any(sha(ROOT/name)!=value for name,value in lock.items()):raise SystemExit('Completion source/plan lock mismatch; version correction required')
  from reconstruction_audit import reconstruction_audit
  phases=[('windows_and_alarms',windows_and_alarms),('severity_and_flow',severity_and_flow),('metric_uncertainty',metric_uncertainty),('residual_stats',residual_stats),('recovery',recovery),('reconstruction_audit',reconstruction_audit),('clusters',clusters)]
  post_status=[]
  for name,fn in phases:
   try:fn();post_status.append({'phase':name,'status':'complete'})
   except Exception as e:
    post_status.append({'phase':name,'status':'incomplete','reason':str(e)})
    atomic(ROOT/'results'/f'COMPLETION_ERROR_{name}.json',{'status':'incomplete','reason':str(e),'exception':type(e).__name__});(ROOT/'logs'/f'completion_{name}.log').write_text(traceback.format_exc())
  atomic(ROOT/'results/POSTPROCESSING_STATUS.json',post_status)
  from external_prepare import prepare
  prepare();supplemental_figures();manuscript_and_package()
 else:raise SystemExit('Use prepare or finalize')
