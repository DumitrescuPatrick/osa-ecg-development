"""Single-process finite runner. All statuses/checkpoints persist; no evaluation-derived re-tuning."""
import argparse,os,time,json,fcntl,traceback,signal
from pathlib import Path
from common import *

def queue():
 q=[]
 def add(stage,kind,args):q.append(dict(id=f'{kind}_'+ '_'.join(map(str,args)),stage=stage,kind=kind,args=args,status='pending'))
 for task in ['detect','forecast','forecast_transition']:
  for name in ['linear','boost']:add('classical','classical',[task,name,1])
 for name in ['mean','median','clinical','linear','combined']:add('classical','classical',['ahi',name,1])
 for task in ['detect','forecast','forecast_transition']:
  for name in ['forest','svm','tree','pca','waveform','rhythm']:add('classical','classical',[task,name,1])
  for name in ['linear','boost']:add('classical','classical',[task,name,2])
 for name in ['forest','boost','svm','waveform']:add('classical','classical',['ahi',name,1])
 for task in ['detect','forecast']:
  for family in ['cnn','resnet','tcn','gru']:add('neural','neural',[task,family,CFG['seed']])
 for k in ['pca_wave','ae','vae','masked']:add('representation','representation',[k])
 for L,g in CFG['residual_geometry']:
  for family in ['ridge','mlp']:add('residual','residual',[L,g,family])
 for s in CFG['neural']['finalist_seeds'][1:]:
  for family in ['cnn','resnet']:add('repeat','neural',['detect',family,s])
 return q

def execute(hours):
 from data import audit,prepare
 from neural import benchmark
 from report import report
 import torch
 torch.set_num_threads(CFG['threads']);start=time.time();deadline=start+hours*3600
 lockfile=(ROOT/'private/RUNNER.lock').open('a+')
 try:fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise SystemExit('Existing runner holds lock; do not launch a second job')
 code=seal();atomic(ROOT/'private/RUNNER.json',dict(pid=os.getpid(),start_epoch=start,deadline_epoch=deadline,code_lock=code,status='running'))
 def safe_signal(signum,frame):(ROOT/'STOP').touch()
 signal.signal(signal.SIGTERM,safe_signal);signal.signal(signal.SIGINT,safe_signal)
 registry_path=ROOT/'results/EXPERIMENT_REGISTRY.json';q=json.loads(registry_path.read_text()) if registry_path.exists() else queue()
 for item in q:
  if item['status']=='running':item['status']='paused'
 atomic(registry_path,q)
 try:
  audit()
  if not prepare(deadline):return
  data_files=[*sorted((ROOT/'private/cache').glob('*.npz')),ROOT/'private/reference.csv',ROOT/'private/reference_grouped.csv',ROOT/'private/record_audit.csv']
  data_hashes={str(p.relative_to(ROOT)):sha(p) for p in data_files};data_lock=ROOT/'private/DERIVED_DATA_LOCK.json'
  if data_lock.exists() and json.loads(data_lock.read_text())!=data_hashes:raise RuntimeError('Derived data or assignments changed; resume blocked')
  if not data_lock.exists():atomic(data_lock,data_hashes)
  if not (ROOT/'results/BENCHMARK.json').exists():benchmark()
  # Family caps reserve time for slower methods; unused time is not an extra search budget.
  caps={'classical':100,'neural':170,'representation':60,'residual':80,'repeat':40}
  for stage in caps:
   stage_deadline=min(deadline,time.time()+caps[stage]*60)
   for item in q:
    if item['stage']!=stage or item['status']=='complete':continue
    if stopped(stage_deadline):break
    item['status']='running';item['started']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());atomic(registry_path,q);t=time.time()
    try:
     if item['kind']=='classical':
      from classical import run
      task,name,duration=item['args'];run(task,name,duration,stage_deadline);expected=ROOT/'results'/f'{task}_{duration}m_{name}.json'
     elif item['kind']=='neural':
      from neural import run
      task,family,s=item['args'];run(task,family,stage_deadline,s);expected=ROOT/'results'/f'{task}_1m_neural_{family}_seed{s}.json'
     elif item['kind']=='representation':
      from representations import run
      kind=item['args'][0];run(kind,stage_deadline);expected=ROOT/'results'/f'detect_1m_representation_{kind}.json'
     else:
      from residual import prepare as rp,run
      if not rp(stage_deadline):item['status']='paused';atomic(registry_path,q);break
      L,g,fam=item['args'];run(L,g,fam,stage_deadline);expected=ROOT/'results'/f'residual_L{L}_g{g}_{fam}.json'
     item['status']='complete' if expected.exists() else 'paused'
    except Exception as e:
     item['status']='error';item['error_type']=type(e).__name__;item['error']=str(e)
     with (ROOT/'logs/errors.log').open('a') as f:f.write(item['id']+'\n'+traceback.format_exc()+'\n')
    item['seconds_last_attempt']=time.time()-t;atomic(registry_path,q)
    # Aggregate reporting is finite and cannot tune the following queue.
    if item['status']=='complete' and item['kind']=='classical':
     # Full report refresh at family boundary, not repeated bootstrap after each fit.
     pass
   report()
   if stopped(deadline):break
 finally:
  try:report()
  except Exception:
   with (ROOT/'logs/errors.log').open('a') as f:f.write('Reporting error\n'+traceback.format_exc())
  atomic(ROOT/'private/RUNNER.json',dict(pid=os.getpid(),start_epoch=start,deadline_epoch=deadline,stopped_epoch=time.time(),status='stopped',code_lock=code));log('Finite run stopped; saved work retained; no automatic extension')

def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['run','resume','status','stop','report','external-download']);p.add_argument('--hours',type=float,default=8);a=p.parse_args()
 if a.command=='status':
  p=ROOT/'private/RUNNER.json';print(p.read_text() if p.exists() else 'No runner started')
  q=json.loads((ROOT/'results/EXPERIMENT_REGISTRY.json').read_text()) if (ROOT/'results/EXPERIMENT_REGISTRY.json').exists() else queue()
  from collections import Counter
  print(dict(Counter(t['status'] for t in q)));print('STOP requested:',(ROOT/'STOP').exists())
  try:
   with (ROOT/'private/RUNNER.lock').open('a+') as f:
    try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);print('Live runner lock: not held')
    except BlockingIOError:print('Live runner lock: held')
  except OSError as e:print('Lock status unavailable',type(e).__name__)
 elif a.command=='stop':(ROOT/'STOP').touch();print('Safe stop requested; wait for checkpoint/bounded fit boundary')
 elif a.command=='report':
  from report import report
  report()
 elif a.command=='external-download':
  from external import fetch
  fetch()
 else:
  if a.command=='resume' and (ROOT/'STOP').exists():(ROOT/'STOP').unlink()
  execute(a.hours)
if __name__=='__main__':main()
