"""Read-only checkpoint evaluation and isolated three-clip memorization experiment."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
from model import MotionModel
from data import load

torch.set_num_threads(2);torch.manual_seed(42)
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'diagnostics';ck=torch.load(ROOT/'model.pt',map_location='cpu',weights_only=True)
model=MotionModel(ck['bins'],ck['model_config']).eval();model.load_state_dict(ck['model'])
def prepare(path):
 x,y,ix,ts,p0=load(path,bins=[np.array(b) for b in ck['bins']]);s=np.load(Path(path)/'metadata'/ck['state_file']);t=np.load(Path(path)/'metadata/frame_times.npy');truth=np.stack([np.interp(ts,t,s[:,j]) for j in [1,3,0,2]],1);truth[:,:2]*=3.6
 return torch.from_numpy(x)[None],torch.from_numpy(p0)[None],torch.from_numpy(ix),torch.from_numpy(y),truth

def evaluate(m,items):
 truth=[];means=[];modes=[];probs=[];ce=[];accuracy=[]
 with torch.inference_mode():
  for x,p0,ix,y,tr in items:
   z,_=m(x,p0);z=[a[0,ix] for a in z];pp=[a.softmax(-1) for a in z];truth.append(tr);means.append(torch.stack([(a*b).sum(-1) for a,b in zip(pp,m.bins)],-1).numpy());modes.append(torch.stack([b[a.argmax(-1)] for a,b in zip(pp,m.bins)],-1).numpy());probs.append(pp[1].numpy());ce.append(np.array([F.cross_entropy(a,y[:,j]).item() for j,a in enumerate(z)])*len(y));accuracy.append(np.array([(a.argmax(-1)==y[:,j]).sum().item() for j,a in enumerate(z)]))
 tr=np.concatenate(truth);ex=np.concatenate(means);mo=np.concatenate(modes);pr=np.concatenate(probs);mask=abs(tr[:,1])>5;b=m.bins[1].numpy()
 result={'timestamps':len(tr),'ce':(np.sum(ce,axis=0)/len(tr)).tolist(),'bin_accuracy':(np.sum(accuracy,axis=0)/len(tr)).tolist(),'expectation_mae':abs(ex-tr).mean(0).tolist(),'argmax_mae':abs(mo-tr).mean(0).tolist(),'position_ade_m':np.linalg.norm(ex[:,2:]-tr[:,2:],axis=1).mean().item(),'vy_sign_accuracy_active':np.mean(np.sign(ex[mask,1])==np.sign(tr[mask,1])).item(),'vy_magnitude_ratio_active':(abs(ex[mask,1]).mean()/abs(tr[mask,1]).mean()).item(),'vy_both_sign_mass_gt20pct_active':np.mean(((pr[:,b>0].sum(1)>.2)&(pr[:,b<0].sum(1)>.2))[mask]).item()}
 return result
results={};train=[]
for split in ['train','val']:
 items=[]
 for i,p in enumerate(ck['split'][split]):
  items.append(prepare(p))
  if (i+1)%20==0: print('loaded',split,i+1,flush=True)
 print('evaluating',split,flush=True);results[split]=evaluate(model,items);print(split,json.dumps(results[split]),flush=True)
 if split=='train':train=items
(OUT/'train_validation.json').write_text(json.dumps(results,indent=2))
# Select three training clips with substantial repeated lateral sign reversals.
candidates=[]
for i,item in enumerate(train):
 vy=item[-1][:,1];v=vy[abs(vy)>5];cross=np.sum(np.sign(v[1:])!=np.sign(v[:-1]));candidates.append((int(cross),float(np.std(vy)),i))
# Spread speeds rather than selecting three near duplicates.
eligible=sorted([r for r in candidates if r[0]>=4],key=lambda r:r[1]);chosen=[eligible[int(q*(len(eligible)-1))][2] for q in [.25,.5,.75]];small=[train[i] for i in chosen]
settings=dict(ck['model_config']);settings['dropout']=0.0
fresh=MotionModel(ck['bins'],settings);opt=torch.optim.AdamW(fresh.parameters(),lr=3e-4,weight_decay=0)
report={'clips':[ck['split']['train'][i] for i in chosen],'settings':settings,'lr':3e-4,'initialization':'fresh random seed 42; dropout disabled; no physics loss','history':[]}
print('overfit clips',report['clips'],flush=True);started=time.time()
for epoch in range(1,201):
 fresh.train()
 for i in np.random.default_rng(epoch).permutation(3):
  x,p0,ix,y,tr=small[i];z,_=fresh(x,p0);loss=torch.stack([F.cross_entropy(a[0,ix],y[:,j]) for j,a in enumerate(z)]).mean();opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(fresh.parameters(),1);opt.step()
 if epoch==1 or epoch%10==0:
  fresh.eval();r=evaluate(fresh,small);r['epoch']=epoch;r['elapsed_seconds']=time.time()-started;report['history'].append(r);(OUT/'overfit.json').write_text(json.dumps(report,indent=2));print('overfit',epoch,json.dumps(r),flush=True)
  if min(r['bin_accuracy'])>.95:break
# Save separate experimental checkpoint; never overwrite model.pt.
torch.save({'model':fresh.state_dict(),'model_config':settings,'bins':ck['bins'],'clips':report['clips'],'epoch':epoch},OUT/'overfit_model.pt')
print('DONE',flush=True)
