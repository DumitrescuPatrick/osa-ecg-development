"""Continuation controller; never edits or imports the locked experiment implementation."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import os,sys,time,json,fcntl,argparse,subprocess,traceback
ROOT=Path(__file__).resolve().parent

def read(path,default=None):
 try:return json.loads(Path(path).read_text())
 except (FileNotFoundError,json.JSONDecodeError):return default

def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def held(name):
 p=ROOT/'private'/name;p.parent.mkdir(exist_ok=True)
 with p.open('a+') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return False
  except BlockingIOError:return True

def status():
 q=read(ROOT/'results/EXPERIMENT_REGISTRY.json',[]);runner=read(ROOT/'private/RUNNER.json',{});controller=read(ROOT/'private/CONTROLLER.json',{});current=[]
 for p in sorted((ROOT/'results').glob('*_learning.json'),key=lambda p:p.stat().st_mtime,reverse=True)[:1]:
  d=read(p,{}) or {};history=d.get('history',[])
  if history:current.append({'experiment':p.stem,'epoch':history[-1]['epoch'],'selected_epoch_so_far':min(history,key=lambda a:a['validation_loss'])['epoch'],'done':d.get('done')})
 answer={'time_UTC':datetime.now(timezone.utc).isoformat(),'runner_lock_held':held('RUNNER.lock'),'runner':runner,'controller_lock_held':held('CONTROLLER.lock'),'controller':controller,'registry_counts':dict(Counter(t.get('status','unknown') for t in q)),'active_registry_entries':[t['id'] for t in q if t.get('status')=='running'],'last_learning_record':current,'stop_requested':(ROOT/'STOP').exists(),'errors_log_present':(ROOT/'logs/errors.log').exists()}
 print(json.dumps(answer,indent=2));return answer

def supervise():
 f=(ROOT/'private/CONTROLLER.lock').open('a+')
 try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:print('Existing continuation controller; no duplicate started',flush=True);return
 authorization=read(ROOT/'configs/CONTINUATION_AUTHORIZATION.json')
 if not authorization:raise RuntimeError('Explicit continuation authorization missing')
 end=authorization['maximum_deadline_epoch'];state={'pid':os.getpid(),'status':'watching_existing_runner','maximum_deadline_epoch':end,'launches':[]};atomic(ROOT/'private/CONTROLLER.json',state)
 def save(state_name):state['status']=state_name;state['heartbeat_epoch']=time.time();atomic(ROOT/'private/CONTROLLER.json',state)
 while time.time()<end:
  if (ROOT/'STOP').exists():save('manual_stop_requested_no_auto_resume');return
  if held('RUNNER.lock'):
   save('watching_existing_runner');time.sleep(15);continue
  q=read(ROOT/'results/EXPERIMENT_REGISTRY.json',[])
  pending=[x for x in q if x.get('status') in ['pending','paused','running']]
  errors=[x for x in q if x.get('status')=='error']
  if not pending:
   save('core_queue_terminal_preparing_reports');break
  original=authorization['original_deadline_epoch'];now=time.time()
  # Use remaining original budget first, then the authorized extension. At most two added launches.
  next_deadline=original if now<original-60 else end
  if next_deadline-now<60:time.sleep(min(15,max(end-now,0)));continue
  if len(state['launches'])>=2:save('bounded_resume_attempt_limit_reached');break
  hours=min(8,(next_deadline-now)/3600)
  if (ROOT/'STOP').exists():save('manual_stop_requested_no_auto_resume');return
  # The runner's own nonblocking lock is the final protection against races.
  env=os.environ.copy();env.update(MPLCONFIGDIR=str(ROOT/'private/matplotlib'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4');env.pop('PYTORCH_ENABLE_MPS_FALLBACK',None)
  cmd=['/usr/bin/caffeinate','-i',sys.executable,'-u',str(ROOT/'src/runner.py'),'resume','--hours',str(hours)]
  with (ROOT/'logs/continuation_runs.log').open('ab') as log:
   child=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
  state['launches'].append({'launcher_pid':child.pid,'start_epoch':time.time(),'hours':hours,'command':cmd,'pending_before':len(pending)});save('checkpoint_resume_running')
  while child.poll() is None:
   save('checkpoint_resume_running');time.sleep(15)
  state['launches'][-1]['exit_code']=child.returncode;save('resume_finished')
  if child.returncode!=0:save('resume_failed_requires_specific_error_review');break
  after=read(ROOT/'results/EXPERIMENT_REGISTRY.json',[])
  if sum(x.get('status')=='complete' for x in after)<=sum(x.get('status')=='complete' for x in q):save('no_new_completed_entries_requires_review');break
 else:save('authorized_time_budget_exhausted')
 while held('RUNNER.lock') and time.time()<end+900:
  save('waiting_for_checkpoint_safe_deadline_stop');time.sleep(15)
 if held('RUNNER.lock'):save('runner_still_active_after_grace_no_concurrent_postprocessing');return
 if (ROOT/'STOP').exists():save('manual_stop_requested_no_auto_postprocessing');return
 save('postprocessing')
 try:
  subprocess.run([sys.executable,str(ROOT/'completion.py'),'finalize'],cwd=ROOT,check=True)
  state['completion_assessment']=read(ROOT/'results/COMPLETION_ASSESSMENT.json',{})
  save('finished_with_documented_status')
 except Exception:
  (ROOT/'logs/completion_errors.log').write_text(traceback.format_exc());save('postprocessing_failed_requires_review')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['status','supervise','stop']);a=p.parse_args()
 if a.command=='status':status()
 elif a.command=='stop':(ROOT/'STOP').touch();print('Checkpoint-safe stop requested; controller will not automatically resume')
 else:supervise()
