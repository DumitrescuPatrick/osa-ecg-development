import json,sys,platform,shutil,textwrap,zipfile,html
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from common import *
from evaluate import paired

def report():
 registry=json.loads((ROOT/'results/EXPERIMENT_REGISTRY.json').read_text()) if (ROOT/'results/EXPERIMENT_REGISTRY.json').exists() else []
 rows=[]
 for p in sorted((ROOT/'results').glob('*.json')):
  d=json.loads(p.read_text())
  if isinstance(d,dict) and 'metrics' in d:
   a={'experiment':p.stem,'task':d['task'],'model':d['model']};a.update({k:v for k,v in d['metrics'].items() if not isinstance(v,(dict,list))});a['CI95']=d['metrics'].get('AUROC_CI95',d['metrics'].get('MAE_CI95'));rows.append(a)
 table=pd.DataFrame(rows);table.to_csv(ROOT/'results/COMPLETED_RESULTS.csv',index=False)
 audit=json.loads((ROOT/'results/DATA_AUDIT.json').read_text()) if (ROOT/'results/DATA_AUDIT.json').exists() else {}
 comparisons=[]
 for task in ['detect','forecast','forecast_transition']:
  a=ROOT/'private/predictions'/f'{task}_1m_linear_OOF.npz';b=ROOT/'private/predictions'/f'{task}_1m_boost_OOF.npz'
  if a.exists() and b.exists():
   x=np.load(a);z=np.load(b);assert np.array_equal(x['y'],z['y']) and np.array_equal(x['record'],z['record']) and np.array_equal(x['minute'],z['minute']);c=paired(x['y'],z['p'],x['p'],x['group'],task);c['task']=task;c['comparison']='boost minus logistic';comparisons.append(c)
 atomic(ROOT/'results/PAIRED_PRIMARY_COMPARISONS.json',comparisons)
 if rows:
  fig,ax=plt.subplots(figsize=(10,max(3,len(rows)*.23)))
  labs=[];values=[]
  for r in rows:
   if 'AUROC' in r:labs.append(r['experiment']);values.append(r['AUROC'])
  if labs:
   ax.barh(labs,values);ax.set_xlim(0,1);ax.set_xlabel('Outer proxy-group AUROC; historically explored database');fig.tight_layout()
   for ext in ['pdf','png']:fig.savefig(ROOT/'figures'/f'COMPLETED_DISCRIMINATION.{ext}',dpi=300)
  plt.close(fig)
 # Primary calibration plots from saved OOF only.
 from sklearn.calibration import calibration_curve
 fig,ax=plt.subplots(figsize=(6,5));found=False
 for task in ['detect','forecast','forecast_transition']:
  p=ROOT/'private/predictions'/f'{task}_1m_linear_OOF.npz'
  if p.exists():
   z=np.load(p);a,b=calibration_curve(z['y'],z['p'],n_bins=10,strategy='quantile');ax.plot(b,a,'o-',label=task);found=True
 if found:
  ax.plot([0,1],[0,1],'k--');ax.set(xlabel='Mean predicted probability',ylabel='Observed positive fraction');ax.legend();fig.tight_layout()
  for ext in ['pdf','png']:fig.savefig(ROOT/'figures'/f'CALIBRATION.{ext}',dpi=300)
 plt.close(fig)
 # Curves are model optimization evidence, not patient-level clinical results.
 paths=list((ROOT/'results').glob('*_learning.json'))
 if paths:
  fig,ax=plt.subplots(figsize=(8,5))
  for p in paths[:12]:
   d=json.loads(p.read_text());h=d['history'];ax.plot([v['epoch'] for v in h],[v['validation_loss'] for v in h],label=p.stem[:50])
  ax.set(xlabel='Epoch',ylabel='Inner-monitor loss');ax.legend(fontsize=5);fig.tight_layout()
  for ext in ['png','pdf']:fig.savefig(ROOT/'figures'/f'LEARNING_CURVES.{ext}',dpi=300)
  plt.close(fig)
 ncomplete=sum(t['status']=='complete' for t in registry);pending=sum(t['status'] in ['pending','paused','running'] for t in registry);failed=sum(t['status']=='error' for t in registry)
 status=f'{ncomplete} completed registry entries; {pending} pending/paused; {failed} errors. Only complete outer-fold results are reported.'
 (ROOT/'reports/RUN_STATUS.md').write_text('# Execution status\n\n'+status+'\n\nSee EXPERIMENT_REGISTRY.json for exact trials and errors. No partial outer-fold estimates are presented as complete. Resume: `./overnight.sh resume` from this study directory.\n')
 metrics=table.to_string(index=False) if len(table) else 'No complete outer-evaluation results yet. No numeric findings are inferred from incomplete trials.'
 text='''# ECG-based apnea detection, forecasting, overnight severity and physiological deviations: a grouped multimethod development study

Interim development report—experiments in progress.



## Abstract
Objective: Determine what ECG can support for current apnea-minute detection, future annotation forecasting, overnight AHI estimation and apnea-associated physiological deviations, without equating these tasks or forcing phenotype discovery.
Design: Historically exposed public-database development study with three outer and two inner proxy-group folds for structured models. Compact neural and representation models use separate inner monitoring, with explicit budget differences. Minute annotations include apnea or hypopnea. Forecasts use ECG ending 60 seconds before a later one-minute target; new transitions are evaluated separately. Whole-night AHI uses the provider's clinical reference rather than apnea-minute counts.
Results: '''+status+'''
'''+metrics+'''
Conclusion: These are development estimates conditional on grouping and historical exposure. Clinical usefulness, causal injury and novel physiological subtypes are not established. External confirmation is not claimed.

## Introduction
ECG rhythm and morphology may change with sleep-disordered breathing, but retrospective minute detection, future forecasting and estimating a night-level clinical index require different information and denominators. Minute labels do not identify exact event boundaries or distinguish all respiratory mechanisms. A broad comparative design is needed to separate model capacity, recording identity, normal-transition prediction error and physiological associations.

## Methods
### Data and provenance
The existing Apnea-ECG files were read without modification. Dataset auditing compared two local copies by SHA256, verified WFDB content/checksums and annotation spacing, and grouped documented overlapping records. The provider additional-information table supplies reference AHI and demographics, separately from minute annotations [1,2]. Reference values were manually transcribed with explicit provenance and require independent verification before final publication. The public page's old annotation-availability prose is not used to overrule actual available test-record annotations. Detailed current audit:
'''+json.dumps(audit,indent=2)+'''

### Prior exploration and subject grouping
Historical AHA scripts and audits are preserved. Reported 594 episodes/62 recordings and 570 controls cannot be treated as a recovered original analysis merely because a later parameter search matches aggregate counts. Separate RMSSD-included matrix/eigenmode and RMSSD-excluded residual approaches existed; surviving files do not establish one fully consistent June calculation. The new cohort is not selected to reproduce those numbers. All source records have historical exposure. Grouping combines documented c05/c06 overlap, exact common content and conservative demographic blocks; these blocks are not verified subject identities. Unknown repeated patients remain a limitation. Inner and outer groups never overlap, but true patient-independent generalization is not proven.

### Task definitions and processing
The dated ANALYSIS_PROTOCOL specifies one- and two-minute lookbacks. Detection is end-of-observation-window classification. State forecasting targets the minute starting 60 seconds after cutoff using ECG only; transition forecasting restricts eligibility to the observed current non-apnea minute and labels future N-to-A transitions without using intervening labels for input or eligibility. Feature extraction is bounded to the available ECG crop, with causal filtering and no future RR interval, normalization or interpolation. RR quality rules, amplitude calibration and excluded windows are documented in source and manifests. No one-minute frequency-HRV validity claim is made. Raw neural waveform information is downsampled to25Hz, limiting fast morphology interpretation.

### Models, representations and selection
Structured candidates are regularized linear, restrained random forest/boosting, RBF support vector and interpretable tree; mean/median AHI baselines and clinical/waveform/combined models are compared. Common proxy-group folds and evaluation rows are reused. Inner training fits preprocessing and hyperparameters. PCA is linear; AE/VAE/masked reconstruction learn from training ECGs only and use common logistic readouts. Clustering complexity uses training-only support and internal/stability criteria, not AHI associations. Compact CNN/ResNet/TCN/CNN–GRU models use AdamW/ReLU, best checkpoints, limited screening and finite epoch ceilings. A single inner monitoring holdout for neural early stopping is a computational compromise relative to full structured-model inner CV. Representation/prediction/reconstruction evidence is reported separately.

### Residual physiology
Ridge and small nonlinear normal-transition models fit training-only pseudo-events; held-out normal residual calibration precedes real-event interpretation. Four prespecified pre/post geometries compare paired real and N-only transitions. Residual scale is based on cross-fitted training normal prediction errors; normal-reference PCA is not fitted on apnea responses. RMSSD is excluded from the state panel; any same-window relation remains contemporaneous. Annotation-defined residual burden cannot be described as an ECG-only detector without an evaluated ECG-derived event schedule. Recovery and respiratory comparisons are reported only if their independent alignment and support checks finish.

### Statistical analysis
Primary current-detection comparison is boosting versus regularized logistic regression at one minute. Discrimination, AP/prevalence, calibration, inner-selected thresholds and whole-night regression metrics are reported from complete outer predictions. Proxy-group bootstrap500 intervals condition on fitted models and do not include full training variability. Repeated seeds assess optimizer variation only. Secondary model/window families require multiplicity-aware interpretation; a visually best model is not a confirmed winner. No partial completed folds are pooled as a final estimate. See paired comparison JSON and all-trial registry.

### External validation
UCDDB is designated external only. Its PSG contains modified-V2 ECG and aligned respiratory/stage annotations, whereas separate Holter timing requires additional evidence [3]. Direct network/DNS access in this execution environment failed, and EDF/event compatibility has not been established. No external performance was evaluated or invented. The selective resumable downloader uses official checksums and excludes unnecessary Holter data. SHHS/MESA require approved access [4,5]; no credential or restricted data access is assumed. Frozen weights, polarity, resampling, events, sleep/wake and AHI-denominator definitions must be finalized before any external comparative result.

## Results
'''+metrics+'''

The registry retains unsuccessful and interrupted trials, rather than interpreting budget exhaustion as scientific failure. Details of residual and clustering results appear in their completed JSON records; absent analyses are pending, not negative. No independent clinical review of prototypes has occurred. Runtime, training updates and checkpoint histories are retained privately, and aggregate curves are regenerated from completed logs.

## Discussion
Any useful detection performance must be interpreted in the context of recurrent apnea persistence, minute-level annotations, historically explored records and imperfect subject identity. Forecasting is not established by classifying post-apnea or preselected clean windows. Predicting provider AHI is distinct from counting apnea-positive minutes; a model trained on whole-night ECG cannot be marketed as early forecasting. Normal-transition deviations may combine physiology, signal quality and model misspecification. Clusters can reflect recording context or conventional rhythm, not new biological subtypes.

## Limitations
The independent sample is small, subject identity incompletely known and historical model-selection exposure extensive. Additional published annotation and AHI denominator details and manual reference transcription need verification. Signal QC can exclude informative difficult periods, which limits applicability; exclusions must be included with denominators. Model budgets differ by computational family and the chosen 25Hz signal loses high-frequency detail. Bootstrap intervals omit model-training/selection uncertainty. External label harmonization and clinical utility remain unverified. Complete broad-program conclusions require completed registered trials; elapsed overnight time alone is not completion.

## Conclusion
Only completed, task-specific development results support numerical claims. Neither novel ECG phenotypes, physiological injury, independent validation nor clinical usefulness follows from this software pipeline or from exploratory association alone.

## Study scope
No prospective registration or independent clinical review is claimed.

## References
1. Penzel T, Moody GB, Mark RG, Goldberger AL, Peter JH. The Apnea-ECG Database. Computers in Cardiology. 2000;27:255–258. https://physionet.org/content/apnea-ecg/1.0.0/
2. PhysioNet. Apnea-ECG additional information. Provider minute counts, reference AI/HI/AHI and demographics. https://physionet.org/content/apnea-ecg/1.0.0/additional-information.txt
3. St. Vincent's University Hospital / University College Dublin Sleep Apnea Database. Version1.0.0. DOI10.13026/C26C7D. https://physionet.org/content/ucddb/1.0.0/
4. National Sleep Research Resource. Sleep Heart Health Study. https://sleepdata.org/datasets/shhs
5. National Sleep Research Resource. MESA Sleep. https://sleepdata.org/datasets/mesa

## Tables and figures
COMPLETED_RESULTS.csv: every fully evaluated model/task, including negative findings. PAIRED_PRIMARY_COMPARISONS.json: conditional paired group intervals. EXPERIMENT_REGISTRY.json: completed, paused, failed and pending work. DATA_AUDIT.json: verified data availability. Figures distinguish calibration, learning and outer discrimination. Representative individual ECGs remain private and do not substitute for blinded clinical interpretation.
'''
 (ROOT/'manuscript/MANUSCRIPT.md').write_text(text);(ROOT/'manuscript/MANUSCRIPT.html').write_text('<!doctype html><meta charset="utf-8"><pre style="white-space:pre-wrap;max-width:1000px;margin:auto">'+html.escape(text)+'</pre>')
 with PdfPages(ROOT/'manuscript/MANUSCRIPT.pdf') as pdf:
  lines=[]
  for para in text.splitlines():lines.extend(textwrap.wrap(para,100) or [''])
  for i in range(0,len(lines),58):
   fig=plt.figure(figsize=(8.27,11.69));fig.text(.07,.95,'\n'.join(lines[i:i+58]),va='top',fontsize=8);pdf.savefig(fig);plt.close(fig)
 (ROOT/'manuscript/COVER_LETTER.md').write_text('Development-study outputs are generated separately from journal correspondence.\n')
 log('Aggregate report and complete-format manuscript draft refreshed; pending experiments explicit')
