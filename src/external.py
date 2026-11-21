"""Public file retrieval only; external outcome evaluation remains a separate locked gate."""
import urllib.request,urllib.error,time,hashlib,shutil,re,json
from pathlib import Path
from common import *
BASE='https://physionet.org/files/ucddb/1.0.0/'
NUMBERS=[2,3,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21,22,23,24,25,26,27,28]
def download(url,p,expected=None):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists() and expected and sha(p)==expected:return
 partial=p.with_suffix(p.suffix+'.part')
 for attempt in range(3):
  offset=partial.stat().st_size if partial.exists() else 0;req=urllib.request.Request(url,headers={'Range':f'bytes={offset}-'} if offset else {})
  try:
   with urllib.request.urlopen(req,timeout=30) as r:
    if offset and r.status!=206:raise RuntimeError('Server did not honor resume; retain partial for explicit investigation')
    with partial.open('ab' if offset else 'wb') as f:
     for chunk in iter(lambda:r.read(2**20),b''):f.write(chunk)
   if expected and sha(partial)!=expected:raise RuntimeError('Official SHA256 mismatch; retain partial and stop')
   partial.replace(p);return
  except urllib.error.URLError:
   if attempt==2:raise
   time.sleep(min(2**attempt,4))
def fetch():
 folder=ROOT/'private/ucddb';folder.mkdir(exist_ok=True)
 if shutil.disk_usage(folder).free<2_000_000_000:raise RuntimeError('Less than2GB free for selective PSG download')
 download(BASE+'SHA256SUMS.txt',folder/'SHA256SUMS.txt')
 mapping={}
 for line in (folder/'SHA256SUMS.txt').read_text().splitlines():
  a=line.split()
  if len(a)==2:mapping[a[1].lstrip('*')]=a[0]
 names=['RECORDS','SubjectDetails.xls']+[f'ucddb{i:03d}{suffix}' for i in NUMBERS for suffix in ['.rec','_stage.txt','_respevt.txt']]
 for name in names:
  if name not in mapping:raise RuntimeError('Required component missing from official checksum manifest')
  download(BASE+name,folder/name,mapping[name]);log('External PSG component verified (no model evaluation)')
 atomic(ROOT/'results/UCDDB_RETRIEVAL.json',dict(components=len(names),checksum_verified=True,Holter_downloaded=False,performance_evaluated=False))

def audit_edf(path):
 # EDF fixed header reader; supports actual per-channel sampling, calibration and start time.
 with Path(path).open('rb') as f:
  h=f.read(256)
  if len(h)!=256:raise ValueError('Short EDF header')
  ns=int(h[252:256]);header_bytes=int(h[184:192]);records=int(h[236:244]);duration=float(h[244:252]);body=f.read(ns*256)
 fields={};offset=0
 for name,width in [('labels',16),('transducer',80),('units',8),('physical_min',8),('physical_max',8),('digital_min',8),('digital_max',8),('prefilter',80),('samples_per_record',8),('reserved',32)]:
  fields[name]=[body[offset+i*width:offset+(i+1)*width].decode('ascii').strip() for i in range(ns)];offset+=width*ns
 expected=header_bytes+records*2*sum(map(int,fields['samples_per_record']))
 if Path(path).stat().st_size!=expected:raise ValueError('EDF size mismatch')
 ecg=[i for i,s in enumerate(fields['labels']) if 'ECG' in s.upper() or 'EKG' in s.upper()]
 if len(ecg)!=1:raise ValueError('Ambiguous PSG ECG channel')
 i=ecg[0]
 return dict(fields=fields,ecg_channel=i,sampling_hz=int(fields['samples_per_record'][i])/duration,start_time=h[176:184].decode(),duration_seconds=records*duration,header_bytes=header_bytes,size_verified=True)
if __name__=='__main__':fetch()
