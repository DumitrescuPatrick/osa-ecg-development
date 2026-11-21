"""Optional native-network public data preparation. Never overwrites existing raw files."""
from pathlib import Path
import argparse,urllib.request,time,hashlib,re,json,csv,os
import numpy as np,wfdb
BASE='https://physionet.org/files/apnea-ecg/1.0.0/'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def get(url):
 for attempt in range(3):
  try:
   with urllib.request.urlopen(url,timeout=30) as r:return r.read()
  except OSError:
   if attempt==2:raise
   time.sleep(1+attempt)
def file(url,path,expected=None):
 path=Path(path)
 if path.exists():
  if expected and digest(path)!=expected:raise RuntimeError('Existing source checksum differs; refusing overwrite')
  return
 partial=path.with_suffix(path.suffix+'.part');offset=partial.stat().st_size if partial.exists() else 0
 req=urllib.request.Request(url,headers={'Range':f'bytes={offset}-'} if offset else {})
 with urllib.request.urlopen(req,timeout=60) as r:
  if offset and r.status!=206:raise RuntimeError('Resume not honored; partial preserved')
  with partial.open('ab' if offset else 'wb') as f:
   for chunk in iter(lambda:r.read(2**20),b''):f.write(chunk)
 if expected and digest(partial)!=expected:raise RuntimeError('Checksum mismatch; partial preserved')
 partial.replace(path)
def prepare(raw,workspace):
 raw=Path(raw).expanduser().resolve();raw.mkdir(parents=True,exist_ok=True);workspace=Path(workspace).expanduser().resolve();(workspace/'private').mkdir(parents=True,exist_ok=True)
 manifest=get(BASE+'SHA256SUMS.txt').decode();hashes={v[1]:v[0] for line in manifest.splitlines() if len(v:=line.split())==2}
 text=get(BASE+'additional-information.txt').decode();rows=[];provider=[]
 for line in text.splitlines():
  v=line.split()
  if len(v)>=12 and re.fullmatch(r'[abcx]\d{2}',v[0]):
   rows.append(dict(record=v[0],AHI=float(v[7]),age=int(v[8]),sex=v[9],height=int(v[10]),weight=int(v[11])));provider.append(dict(record=v[0],minutes=int(v[1]),non_apnea_minutes=int(v[2]),apnea_minutes=int(v[3]),hours_with_apnea=int(v[4]),AI=float(v[5]),HI=float(v[6]),AHI=float(v[7]),age=int(v[8]),sex=v[9],height=int(v[10]),weight=int(v[11])))
 if len(rows)!=70:raise ValueError('Provider reference table schema changed')
 labels={};key=None
 for line in get(BASE+'test-dataset-annos.txt').decode().splitlines():
  if re.fullmatch(r'x\d{2}',line.strip()):key=line.strip();labels[key]=''
  elif key:labels[key]+=''.join(re.findall('[AN]',line))
 if len(labels)!=35:raise ValueError('Test label schema changed')
 for row in rows:
  name=row['record']
  for ext in ['hea','dat','qrs']+(['apn'] if not name.startswith('x') else []):file(BASE+name+'.'+ext,raw/(name+'.'+ext),hashes.get(name+'.'+ext))
  if name.startswith('x'):
   target=raw/(name+'.apn')
   if not target.exists():wfdb.wrann(name,'apn',np.arange(len(labels[name]))*6000,symbol=list(labels[name]),fs=100,write_dir=str(raw))
   a=wfdb.rdann(str(raw/name),'apn')
   if ''.join(a.symbol)!=labels[name] or not np.array_equal(a.sample,np.arange(len(labels[name]))*6000):raise ValueError('Existing test annotations differ; refusing overwrite')
 ref=workspace/'private/reference.csv'
 if ref.exists():raise RuntimeError('Existing reference file retained; use a new reproduction workspace')
 with ref.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=['record','AHI','age','sex','height','weight']);w.writeheader();w.writerows(rows)
 pub=workspace/'publication_v1/private';pub.mkdir(parents=True,exist_ok=True)
 (pub/'OFFICIAL_REFERENCE_TABLE.json').write_text(json.dumps({'source':BASE+'additional-information.txt','rows':provider},indent=2))
 (pub/'OFFICIAL_CORE_HASHES.json').write_text(json.dumps({'source':BASE+'SHA256SUMS.txt','hashes':{name:h for name,h in hashes.items() if re.fullmatch(r'[abcx]\d{2}\.(dat|hea|qrs|apn)',name)}},indent=2))
 (pub/'OFFICIAL_TEST_LABELS.json').write_text(json.dumps({'source':BASE+'test-dataset-annos.txt','labels':labels},indent=2))
 print('Raw data/reference prepared; public source attribution and exact software versions remain required.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--raw-dir',required=True);p.add_argument('--workspace',required=True);a=p.parse_args();prepare(a.raw_dir,a.workspace)
