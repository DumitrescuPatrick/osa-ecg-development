from pathlib import Path
import os,json,hashlib,time,platform,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];CFG=json.loads((ROOT/'configs/study.json').read_text())
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,str(CFG['threads']))
def atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)));tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def stopped(deadline):return (ROOT/'STOP').exists() or time.time()>=deadline
def seal():
 paths=[ROOT/'configs/study.json',ROOT/'reports/ANALYSIS_PROTOCOL.md',ROOT/'reports/CORRECTIONS.md',*sorted((ROOT/'src').glob('*.py'))]
 d={str(p.relative_to(ROOT)):sha(p) for p in paths};p=ROOT/'private/CODE_LOCK.json'
 if p.exists() and json.loads(p.read_text())!=d:raise RuntimeError('Source/config differs from executed lock: version correction required')
 if not p.exists():atomic(p,d)
 return sha(p)
def log(msg):
 line=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+' '+msg
 print(line,flush=True)
 with (ROOT/'logs/progress.log').open('a') as f:f.write(line+'\n')
