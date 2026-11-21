import os,time,random,copy,json
import numpy as np,torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import GroupShuffleSplit
from common import *
from data import load,split_groups
from evaluate import summarize,threshold

def device():
 if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise RuntimeError('Silent MPS CPU fallback forbidden; unset PYTORCH_ENABLE_MPS_FALLBACK')
 return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s)
class Block(nn.Module):
 def __init__(self,a,b,stride=2,dilation=1):
  super().__init__();self.f=nn.Sequential(nn.Conv1d(a,b,7,stride=stride,padding=3*dilation,dilation=dilation),nn.GroupNorm(4,b),nn.ReLU(),nn.Conv1d(b,b,3,padding=1),nn.GroupNorm(4,b));self.skip=nn.Conv1d(a,b,1,stride=stride)
 def forward(self,x):return torch.relu(self.f(x)+self.skip(x))
class Encoder(nn.Module):
 def __init__(self,family='cnn',width=16):
  super().__init__();self.family=family
  if family=='cnn':self.net=nn.Sequential(nn.Conv1d(1,width,9,stride=3,padding=4),nn.ReLU(),nn.Conv1d(width,width*2,7,stride=3,padding=3),nn.ReLU(),nn.Conv1d(width*2,32,5,stride=2,padding=2),nn.ReLU())
  elif family in ['resnet','tcn']:self.net=nn.Sequential(Block(1,width,3),Block(width,width*2,3,2 if family=='tcn' else 1),Block(width*2,32,2,4 if family=='tcn' else 1))
  elif family=='gru':
   self.net=nn.Sequential(nn.Conv1d(1,16,9,stride=5,padding=4),nn.ReLU(),nn.Conv1d(16,32,5,stride=3,padding=2),nn.ReLU());self.gru=nn.GRU(32,32,batch_first=True)
  else:raise ValueError(family)
 def forward(self,x):
  z=self.net(x)
  if self.family=='gru':z,_=self.gru(z.transpose(1,2));return z[:,-1]
  return z.mean(-1)
class Net(nn.Module):
 def __init__(self,family='cnn',drop=.1):super().__init__();self.encoder=Encoder(family);self.head=nn.Sequential(nn.Dropout(drop),nn.Linear(32,1))
 def forward(self,x):return self.head(self.encoder(x)).squeeze(-1)
class Autoencoder(nn.Module):
 def __init__(self,kind='ae'):
  super().__init__();self.kind=kind;self.encoder=Encoder('cnn');self.mu=nn.Linear(32,16);self.logvar=nn.Linear(32,16);self.decoder=nn.Sequential(nn.Linear(16,32*25),nn.ReLU(),nn.Unflatten(1,(32,25)),nn.ConvTranspose1d(32,16,8,stride=5,padding=1),nn.ReLU(),nn.Conv1d(16,1,7,padding=3))
 def latent(self,x):return self.mu(self.encoder(x))
 def forward(self,x):
  inp=x
  if self.kind=='masked' and self.training:
   # Mask contiguous 0.4s blocks without warping amplitude, polarity or timing.
   mask=(torch.rand((len(x),1,150),device=x.device)>.15).repeat_interleave(10,-1);inp=x*mask
  z=self.encoder(inp);mu=self.mu(z);lv=self.logvar(z).clamp(-8,8);v=mu+torch.randn_like(mu)*torch.exp(.5*lv) if self.kind=='vae' and self.training else mu
  out=nn.functional.interpolate(self.decoder(v),size=x.shape[-1],mode='linear',align_corners=False)
  kl=.5*(mu.square()+lv.exp()-1-lv).mean() if self.kind=='vae' else mu.sum()*0
  return out,kl

def save_torch(path,value):
 tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)
def fit_net(net,x,y,train,monitor,tag,deadline,lr=.001,epochs=40,reconstruct=False,fixed=False,seed_value=None):
 seed(CFG['seed'] if seed_value is None else seed_value);dev=device();net.to(dev);params=sum(p.numel() for p in net.parameters());assert params<=CFG['neural']['max_parameters']
 ck=ROOT/'checkpoints_private'/f'{tag}.pt';opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=.001);sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=3,factor=.5)
 start=0;best=float('inf');beststate=copy.deepcopy(net.state_dict());bad=0;history=[];mean=float(x[train].mean());sd=max(float(x[train].std()),1e-6);updates=0;examples=0
 if ck.exists():
  c=torch.load(ck,map_location='cpu',weights_only=False);net.load_state_dict(c['model']);net.to(dev);opt.load_state_dict(c['optimizer']);sched.load_state_dict(c['scheduler']);start=c['epoch'];best=c['best'];beststate=c['best_model'];bad=c['bad'];history=c['history'];mean=c['mean'];sd=c['SD'];updates=c['updates'];examples=c['examples'];torch.set_rng_state(c['torch_rng']);np.random.set_state(c['numpy_rng']);random.setstate(c['python_rng'])
  if dev.type=='mps' and c.get('mps_rng') is not None:torch.mps.set_rng_state(c['mps_rng'])
  if c['done']:net.load_state_dict(beststate);return net,dict(mean=mean,SD=sd,epochs=start,selected_epoch=c['selected_epoch'],parameters=params,history=history,device=str(dev),updates=updates,examples=examples)
 # CPU tensors only become device tensors per batch; no per-batch .item/.cpu transfers.
 xx=torch.from_numpy(((x-mean)/sd).astype(np.float32)).unsqueeze(1);yy=torch.from_numpy(y.astype(np.float32))
 loader=DataLoader(TensorDataset(xx[train],yy[train]),batch_size=32,shuffle=True,num_workers=0,pin_memory=False)
 vl=DataLoader(TensorDataset(xx[monitor],yy[monitor]),batch_size=32,num_workers=0,pin_memory=False)
 selected=1 if not history else int(np.argmin([h['validation_loss'] for h in history])+1)
 for ep in range(start,epochs):
  if stopped(deadline):break
  begin=time.time();net.train();total=torch.zeros((),device=dev);count=0
  for xb,yb in loader:
   xb=xb.to(dev);yb=yb.to(dev);opt.zero_grad(set_to_none=True)
   if reconstruct:out,kl=net(xb);loss=(out-xb).square().mean()+(.001*kl if net.kind=='vae' else 0)
   else:loss=nn.functional.binary_cross_entropy_with_logits(net(xb),yb)
   loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);opt.step();total+=loss.detach()*len(xb);count+=len(xb);updates+=1;examples+=len(xb)
  net.eval();vt=torch.zeros((),device=dev);vc=0
  with torch.no_grad():
   for xb,yb in vl:
    xb=xb.to(dev);yb=yb.to(dev)
    if reconstruct:out,kl=net(xb);loss=(out-xb).square().mean()+(.001*kl if net.kind=='vae' else 0)
    else:loss=nn.functional.binary_cross_entropy_with_logits(net(xb),yb)
    vt+=loss*len(xb);vc+=len(xb)
  tl=float((total/max(count,1)).cpu());v=float((vt/max(vc,1)).cpu());sched.step(v)
  if v<best-CFG['neural']['minimum_improvement']:best=v;beststate=copy.deepcopy(net.state_dict());bad=0;selected=ep+1
  else:bad+=1
  if fixed:beststate=copy.deepcopy(net.state_dict());selected=ep+1
  history.append(dict(epoch=ep+1,training_loss=tl,validation_loss=v,seconds=time.time()-begin,learning_rate=opt.param_groups[0]['lr']))
  done=(ep+1==epochs) or (bad>=CFG['neural']['patience'] and not fixed)
  save_torch(ck,dict(model=net.state_dict(),best_model=beststate,optimizer=opt.state_dict(),scheduler=sched.state_dict(),epoch=ep+1,best=best,bad=bad,history=history,mean=mean,SD=sd,updates=updates,examples=examples,done=done,selected_epoch=selected,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),mps_rng=torch.mps.get_rng_state() if dev.type=='mps' else None))
  atomic(ROOT/'results'/f'{tag}_learning.json',dict(history=history,done=done,parameters=params,device=str(dev),updates=updates,examples=examples))
  log(f'{tag} epoch {ep+1}: train {tl:.4f}, monitor {v:.4f}; checkpoint saved')
  if done:break
 else:done=True
 if not ck.exists() or not torch.load(ck,map_location='cpu',weights_only=False)['done']:return None,None
 net.load_state_dict(beststate);return net,dict(mean=mean,SD=sd,epochs=len(history),selected_epoch=selected,parameters=params,history=history,device=str(dev),updates=updates,examples=examples)

def infer(net,x,info,latent=False):
 dev=device();net.eval();out=[]
 with torch.no_grad():
  for i in range(0,len(x),128):
   a=torch.from_numpy(((x[i:i+128]-info['mean'])/info['SD']).astype(np.float32)).unsqueeze(1).to(dev)
   p=net.latent(a) if latent else torch.sigmoid(net(a));out.append(p.cpu().numpy())
 return np.concatenate(out)
def run(task,family,deadline,seed_value=None):
 seed_value=CFG['seed'] if seed_value is None else seed_value;tag=f'{task}_1m_neural_{family}_seed{seed_value}'
 if (ROOT/'results'/f'{tag}.json').exists():return
 _,y,g,r,t,x=load(task,1);splits=split_groups(y,g);allp=np.full(len(y),np.nan);allth=np.full(len(y),np.nan)
 for fi,(tr,te) in enumerate(splits):
  if stopped(deadline):return
  pfile=ROOT/'private/predictions'/f'{tag}_fold{fi}.npz'
  if pfile.exists():z=np.load(pfile);allp[te]=z['p'];allth[te]=z['threshold'];continue
  a,b=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=CFG['seed']+fi).split(x[tr],y[tr],g[tr]));a=tr[a];b=tr[b]
  if np.unique(y[a]).size<2 or np.unique(y[b]).size<2:raise RuntimeError('Inner neural monitor lacks class support')
  rng=np.random.default_rng(CFG['seed']);a=np.sort(rng.choice(a,min(len(a),6000),replace=False));chosen=None;best=np.inf
  # Three-seed repeats fix configuration0; main seed screens two predefined rates.
  pars=range(2) if seed_value==CFG['seed'] else range(1)
  for par in pars:
   seed(seed_value);net=Net(family);net,info=fit_net(net,x,y,a,b,f'{tag}_f{fi}_screen{par}',deadline,lr=CFG['neural']['learning_rates'][par],seed_value=seed_value)
   if net is None:return
   v=min(h['validation_loss'] for h in info['history'])
   if v<best:best=v;chosen=(par,net,info)
  par,net,info=chosen;th=threshold(y[b],infer(net,x[b],info));budget=info['selected_epoch'];seed(seed_value);net=Net(family)
  fitix=np.sort(rng.choice(tr,min(len(tr),6000),replace=False));net,info=fit_net(net,x,y,fitix,b,f'{tag}_f{fi}_refit',deadline,lr=CFG['neural']['learning_rates'][par],epochs=budget,fixed=True,seed_value=seed_value)
  if net is None:return
  p=infer(net,x[te],info);allp[te]=p;allth[te]=th;np.savez_compressed(pfile,p=p,threshold=th)
  log(f'{tag} outer fold {fi+1}/3 complete')
 np.savez_compressed(ROOT/'private/predictions'/f'{tag}_OOF.npz',y=y,p=allp,group=g,record=r,minute=t,threshold=allth)
 atomic(ROOT/'results'/f'{tag}.json',dict(task=task,model=family,seed=seed_value,metrics=summarize(y,allp,g,task,allth),selection='one inner group holdout; 2 rate screen main seed; repeats fixed config0; conditional comparison, not equal full-inner-CV budget'))

def benchmark():
 seed(CFG['seed']);torch.set_num_threads(CFG['threads']);rows=[]
 for batch in [8,16,32]:
  net=Net('resnet');dev=device();net.to(dev);x=torch.randn(batch,1,1500,device=dev);y=torch.rand(batch,device=dev);opt=torch.optim.AdamW(net.parameters());start=time.time()
  for _ in range(5):opt.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(net(x),y);loss.backward();opt.step()
  if dev.type=='mps':torch.mps.synchronize()
  rows.append(dict(batch=batch,device=str(dev),examples_per_second=5*batch/(time.time()-start),parameters=sum(p.numel() for p in net.parameters())))
 atomic(ROOT/'results/BENCHMARK.json',dict(MPS_available=torch.backends.mps.is_available(),pin_memory=False,num_workers=0,rows=rows,unsupported_operations_fallback=False))
