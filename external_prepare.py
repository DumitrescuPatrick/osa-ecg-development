"""Outcome-blind EDF preparation and strict event parser. No prediction or outcome tuning."""
from pathlib import Path
import sys,re,json,math
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from external import audit_edf,fetch
from common import atomic,sha

def clock(value):
 a=list(map(int,value.replace('.',':').split(':')))
 if len(a)!=3 or not(0<=a[0]<24 and 0<=a[1]<60 and 0<=a[2]<60):raise ValueError('Invalid clock time')
 return a[0]*3600+a[1]*60+a[2]

def event_rows(text,start_clock,duration):
 start=clock(start_clock);events=[];unknown=[];last=-1
 for line in text.splitlines():
  if not re.match(r'^\s*\d{2}:\d{2}:\d{2}\s',line):continue
  t=line.split();code=t[1];offset=clock(t[0])-start
  if offset<0:offset+=86400
  if offset<last:raise ValueError('Event times are nonmonotonic after midnight mapping')
  last=offset
  if re.fullmatch(r'(APNEA|HYP)-(O|C|M)',code) is None:
   if code in ['PB','CS']:continue
   unknown.append(code);continue
  rest=t[2:]
  if rest and rest[0] in ['PB','CS']:rest=rest[1:]
  if not rest:raise ValueError('Missing event duration')
  length=float(rest[0])
  if not math.isfinite(length) or length<=0 or offset<0 or offset+length>duration:raise ValueError('Event outside PSG or invalid duration')
  events.append((offset,offset+length,code))
 if unknown:raise ValueError('Unrecognized respiratory types: '+','.join(sorted(set(unknown))))
 return events

def minute_labels(events,duration):
 n=int(duration//60);y=np.zeros(n,dtype=np.int8)
 for begin,end,code in events:
  for m in range(max(0,int(begin//60)),min(n,int(math.ceil(end/60)))):
   if min(end,(m+1)*60)>max(begin,m*60):y[m]=1
 return y

def ecg_signal(path,meta):
 fields=meta['fields'];i=meta['ecg_channel'];counts=list(map(int,fields['samples_per_record']));total=sum(counts);n=(Path(path).stat().st_size-meta['header_bytes'])//(2*total)
 raw=np.memmap(path,dtype='<i2',mode='r',offset=meta['header_bytes'],shape=(n,total));x=np.asarray(raw[:,sum(counts[:i]):sum(counts[:i+1])],float).reshape(-1)
 lo=float(fields['digital_min'][i]);hi=float(fields['digital_max'][i]);pl=float(fields['physical_min'][i]);ph=float(fields['physical_max'][i])
 if hi<=lo or ph<=pl:raise ValueError('Nonpositive EDF calibration')
 x=(x-lo)*(ph-pl)/(hi-lo)+pl;units=fields['units'][i].replace('µ','u').lower()
 factor={'mv':1.,'uv':.001,'v':1000.}.get(units)
 if factor is None:raise ValueError('Unsupported ECG unit')
 return x*factor

def prepare():
 folder=ROOT/'private/ucddb';records=sorted(folder.glob('*.rec')) if folder.exists() else []
 if not records:
  atomic(ROOT/'results/EXTERNAL_PREPARATION.json',{'status':'blocked','reason':'No local UCDDB PSG files; network DNS blocked in current execution environment','performance_evaluated':False,'model_manifest_frozen':False,'parser_synthetic_tests_only':True});return
 # Do not parse respiratory outcome files before models, thresholds and official checksum gate are frozen.
 receipt=ROOT/'private/EXTERNAL_FROZEN_MODEL_MANIFEST.json'
 headers=[]
 for p in records:
  meta=audit_edf(p);headers.append({'size_verified':True,'sampling_hz':meta['sampling_hz'],'duration_seconds':meta['duration_seconds'],'ECG_channel_identified':True})
 atomic(ROOT/'private/UCDDB_HEADER_AUDIT.json',headers)
 atomic(ROOT/'results/EXTERNAL_PREPARATION.json',{'status':'headers_prepared_outcomes_not_opened','records':len(headers),'sampling_frequencies_hz':sorted(set(v['sampling_hz'] for v in headers)),'duration_range_seconds':[min(v['duration_seconds'] for v in headers),max(v['duration_seconds'] for v in headers)],'all_header_sizes_verified':True,'model_manifest_frozen':receipt.exists(),'performance_evaluated':False,'stage_alignment_independently_verified':False,'AHI_denominator_verified':False})
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='download':fetch()
 else:prepare()
