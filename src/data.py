import json,re,time
from pathlib import Path
import numpy as np,pandas as pd,wfdb
from scipy.signal import butter,sosfilt,find_peaks,resample_poly
from common import *
FEATURES=['mean_HR','mean_RR','SDNN','RMSSD','pNN50','RR_IQR','RR_slope','RMS','peak_to_peak','roughness','band_5_15_fraction','RR_rejected_fraction']
def features(x,fs=100):
 x=np.asarray(x,float);out=np.full(len(FEATURES),np.nan)
 if len(x)<fs*30 or np.isfinite(x).mean()<.95:return out,None
 if not np.isfinite(x).all():return out,None
 if np.ptp(x)<1e-6:return out,None
 # Only the observed crop enters filtering/detection; never future samples.
 filt=sosfilt(butter(3,[5,20],btype='bandpass',fs=fs,output='sos'),x)
 energy=np.convolve(filt*filt,np.ones(9)/9,'full')[:len(x)]
 med=np.median(energy);span=np.quantile(energy,.95)-med
 peaks=find_peaks(energy,distance=int(.3*fs),height=med+.2*span,prominence=.15*span)[0]
 rr=np.diff(peaks)/fs
 valid=(rr>=.3)&(rr<=2)
 if len(rr)>1:valid[1:]&=np.abs(np.diff(rr))/np.maximum(rr[:-1],1e-6)<=.3
 reject=1-valid.mean() if len(valid) else 1
 if valid.sum()<5 or reject>.2:return out,None
 r=rr[valid];d=np.diff(rr);pair=valid[1:]&valid[:-1];d=d[pair]
 out[:7]=[60/r.mean(),r.mean()*1000,r.std(ddof=1)*1000,np.sqrt(np.mean(d*d))*1000 if len(d) else np.nan,np.mean(abs(d)>.05) if len(d) else np.nan,(np.quantile(r,.75)-np.quantile(r,.25))*1000,np.polyfit(np.flatnonzero(valid)/max(len(rr),1),r,1)[0]*1000]
 fft=abs(np.fft.rfft(x-x.mean()))**2;freq=np.fft.rfftfreq(len(x),1/fs)
 out[7:]=[np.sqrt(np.mean(x*x)),np.ptp(x),np.sqrt(np.mean(np.diff(x)**2)),fft[(freq>=5)&(freq<=15)].sum()/max(fft.sum(),1e-12),reject]
 wave=resample_poly(x,1,4).astype('float32') # no samples beyond observed crop
 return out,wave

def audit():
 if (ROOT/'results/DATA_AUDIT.json').exists():
  for entry in json.loads((ROOT/'private/FILE_MANIFEST.json').read_text()):
   name=entry['record']+'.'+entry['extension']
   if sha(Path(CFG['source'])/name)!=entry['sha256']:raise RuntimeError('Original data changed; resume blocked')
   if entry['copy_sha256'] is not None and sha(Path(CFG['copy'])/name)!=entry['copy_sha256']:raise RuntimeError('Comparison data changed; resume blocked')
  return
 source=Path(CFG['source']);copy=Path(CFG['copy']);ref=pd.read_csv(ROOT/'private/reference.csv');names=ref.record.tolist()
 inventories=[];rec=[];parent={n:n for n in names}
 def find(a):
  while parent[a]!=a:a=parent[a]
  return a
 def join(a,b):parent[find(b)]=find(a)
 # Conservative demographic blocking is NOT established subject identity.
 for _,g in ref.groupby(['age','sex','height']):
  for n in g.record:join(g.record.iloc[0],n)
 join('c05','c06');fingerprints={};shared=0
 for i,n in enumerate(names):
  for ext in ['hea','dat','apn','qrs']:
   a=source/f'{n}.{ext}';b=copy/f'{n}.{ext}'
   if not a.exists():raise ValueError('Required primary component missing: '+ext)
   h=sha(a);h2=sha(b) if b.exists() else None
   inventories.append(dict(record=n,extension=ext,bytes=a.stat().st_size,sha256=h,copy_sha256=h2,agree=h==h2))
   if h2 is not None and h!=h2:raise ValueError('Copy checksum mismatch; official investigation required')
  h=wfdb.rdheader(str(source/n));s=wfdb.rdrecord(str(source/n),physical=False);a=wfdb.rdann(str(source/n),'apn');q=wfdb.rdann(str(source/n),'qrs')
  if h.fs!=100 or h.n_sig!=1 or h.sig_len<=0:raise ValueError('Unexpected header')
  if s.d_signal.shape!=(h.sig_len,1) or not np.isfinite(s.d_signal).all():raise ValueError('Invalid data')
  checksum_ok=[int(np.asarray(s.d_signal[:,j],np.int64).sum())%65536==int(h.checksum[j])%65536 for j in range(h.n_sig)]
  if not all(checksum_ok):raise ValueError('WFDB checksum failed')
  if not set(a.symbol)<=set('AN') or not np.all(np.diff(a.sample)==6000) or a.sample[0]!=0:raise ValueError('Annotation alignment failed')
  if np.any(a.sample>=h.sig_len):raise ValueError('Annotation starts outside ECG')
  complete=a.sample+6000<=h.sig_len
  # Detect exact common one-minute content; offsets other than minute-grid require separate known-overlap grouping.
  z=s.d_signal[:,0]
  for j in range(0,len(z)-6000,6000):
   f=hashlib.sha256(z[j:j+6000].tobytes()).digest()
   if f in fingerprints and fingerprints[f]!=n:join(fingerprints[f],n);shared+=1
   else:fingerprints[f]=n
  qrr=np.diff(q.sample)/h.fs
  rec.append(dict(record=n,samples=h.sig_len,fs=h.fs,units=h.units[0],gain=h.adc_gain[0],signal_name=h.sig_name[0],minutes=len(a.symbol),apnea_minutes=a.symbol.count('A'),qrs_count=len(q.sample),qrs_invalid_RR_fraction=float(np.mean((qrr<.3)|(qrr>2))),qrs_within_signal=bool(np.all((q.sample>=0)&(q.sample<h.sig_len))),checksum_pass=True,incomplete_final_minutes=int((~complete).sum()),complete_minutes=int(complete.sum())))
  log(f'Integrity verified {i+1}/70')
 ref['group']=[find(n) for n in names];ref['BMI']=ref.weight/(ref.height/100)**2;ref['sex_numeric']=(ref.sex=='M').astype(int)
 ref.to_csv(ROOT/'private/reference_grouped.csv',index=False);pd.DataFrame(rec).to_csv(ROOT/'private/record_audit.csv',index=False);atomic(ROOT/'private/FILE_MANIFEST.json',inventories)
 atomic(ROOT/'results/DATA_AUDIT.json',dict(recordings=len(names),copies_identical=all(d['agree'] for d in inventories),components_compared=len(inventories),proxy_groups=int(ref.group.nunique()),verified_patient_identities=False,documented_c05_c06_grouped=True,exact_shared_minute_blocks=shared,minutes=sum(d['minutes'] for d in rec),apnea_minutes=sum(d['apnea_minutes'] for d in rec),sampling_hz=100,units=sorted({d['units'] for d in rec}),wfdb_checksums_pass=True,all_70_annotations_present=True,incomplete_final_minutes_excluded=sum(d['incomplete_final_minutes'] for d in rec),AHI_source='PhysioNet additional-information.txt, public provider table; manual transcription requires independent verification',historically_explored=True))

def prepare(deadline):
 source=Path(CFG['source']);ref=pd.read_csv(ROOT/'private/reference_grouped.csv')
 for i,row in ref.iterrows():
  p=ROOT/'private/cache'/f'{row.record}.npz'
  if p.exists():continue
  if stopped(deadline):return False
  h=wfdb.rdrecord(str(source/row.record));x=h.p_signal[:,0];a=wfdb.rdann(str(source/row.record),'apn');y=np.array([s=='A' for s in a.symbol],np.int8);n=len(y)
  complete=a.sample+6000<=len(x)
  F=np.full((n,2,len(FEATURES)),np.nan,np.float32);W=np.zeros((n,1500),np.float32)
  for t in range(n):
   for j,L in enumerate(CFG['durations_minutes']):
    if t+1<L or not complete[t]:continue
    f,w=features(x[(t+1-L)*6000:(t+1)*6000]);F[t,j]=f
    if L==1 and w is not None:W[t]=w
  p.parent.mkdir(exist_ok=True);np.savez_compressed(p,F=F,W=W,y=y)
  log(f'Features cached {i+1}/70; valid one-minute windows {int(np.isfinite(F[:,0]).all(1).sum())}/{n}')
 return True

def load(task='detect',duration=1):
 ref=pd.read_csv(ROOT/'private/reference_grouped.csv');allx=[];ally=[];groups=[];records=[];times=[];waves=[]
 for ri,row in ref.iterrows():
  z=np.load(ROOT/'private/cache'/f'{row.record}.npz');f=z['F'][:,CFG['durations_minutes'].index(duration)];y=z['y'];w=z['W'];t=np.arange(len(y));valid=np.isfinite(f).all(1)
  if task.startswith('forecast'):
   valid&=t+2<len(y);valid&=(np.minimum(t+2,len(y)-1)+1)*6000<=int(pd.read_csv(ROOT/'private/record_audit.csv').iloc[ri]['samples']);target=y[np.minimum(t+2,len(y)-1)].copy()
   if task=='forecast_transition':
    valid&=y==0;target&=y[np.minimum(t+1,len(y)-1)]==0
  else:target=y
  allx.append(f[valid]);ally.append(target[valid]);groups.extend([row.group]*valid.sum());records.extend([ri]*valid.sum());times.extend(t[valid]);waves.append(w[valid])
 return np.concatenate(allx),np.concatenate(ally),np.asarray(groups),np.asarray(records),np.asarray(times),np.concatenate(waves)

def split_groups(y,g):
 from sklearn.model_selection import StratifiedGroupKFold
 return [(a,b) for a,b in StratifiedGroupKFold(CFG['outer_folds'],shuffle=True,random_state=CFG['seed']).split(np.zeros(len(y)),y,g)]
