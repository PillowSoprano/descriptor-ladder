"""Cross-size, cross-interaction prediction from existing GB1 data; no gain fits."""
import math,time,json
import numpy as np,pandas as pd
from numba import njit
from scipy.stats import binom
from scipy.optimize import brentq
import run_gain_by_block as run
ROOT=run.ROOT;OUT=ROOT/'reports/connectivity_window';OUT.mkdir(exist_ok=True)
MS=[7,10,13,16,19];ETAS=[.5,1.,1.5]

@njit
def exact_paths(U,ps,ym,yu,b,V,s,sp,edges,orders):
 L=len(edges);N=len(ym);p=U.shape[1];out=np.zeros((N-1,2,13))
 for t in range(L):
  i,j=edges[t];d=U[j]-U[i]
  for mod in range(2):
   pen=ps[mod];A=np.diag(1/pen);rm=np.zeros(p);ru=np.zeros(p)
   for n in range(N-1):
    a=A@d;pa=pen*a;mu=np.dot(pa,b)
    nu=max(0.,2+np.dot(d,a)-np.sum((pen+pen*pen)*a*a))
    em=ym[j]-ym[i]-np.dot(a,rm);eu=yu[j]-yu[i]-np.dot(a,ru)
    for k in range(3):
     eta=.5+.5*k;err=em+eta*eu
     out[n,mod,2*k]+=err*err;out[n,mod,2*k+1]+=abs(err)
     sd=np.sqrt(max(s*eta*eta*nu,1e-30))
     out[n,mod,10+k]+=sd*np.sqrt(2/np.pi)*np.exp(-mu*mu/(2*sd*sd))+mu*math.erf(mu/(np.sqrt(2)*sd))
    out[n,mod,6]+=mu*mu;out[n,mod,7]+=nu
    var=max(0.,np.dot(pa,V@pa)+sp*nu)
    out[n,mod,8]+=var;out[n,mod,9]+=np.sqrt(2/np.pi)*np.sqrt(var)
    if n<N-2:
     k=orders[t,n];u=U[k];au=A@u;A-=np.outer(au,au)/(1+np.dot(u,au));rm+=u*ym[k];ru+=u*yu[k]
 return out/L

def load_grids():
 hist=pd.read_csv(ROOT/'reports/protein_replication.csv');raw=pd.read_csv(ROOT/'data/raw/gb1/gb1.tsv',sep='\t');db=raw[raw.num_mutations==2].copy();parts=db.variant.str.split(',',expand=True)
 for i in [1,2]:db[f'p{i}']=parts[i-1].str.extract(r'(\d+)').astype(int);db[f'a{i}']=parts[i-1].str[-1]
 aa=sorted(set(db.a1)|set(db.a2));za=run.protein.descriptors(aa);lookup={v:i for i,v in enumerate(aa)}
 for pair in hist.pair.drop_duplicates():
  ap,bp=map(int,pair.split('-'));M=db[(db.p1==ap)&(db.p2==bp)].pivot_table(index='a1',columns='a2',values='score')
  Z={'A':za[[lookup[v] for v in M.index]],'B':za[[lookup[v] for v in M.columns]]}
  yield pair,M.to_numpy(),Z

def setup(m,Z):
 idx={'A':np.repeat(np.arange(m),m),'B':np.tile(np.arange(m),m)};L={s:run.protein.laplacian(z,3) for s,z in Z.items()}
 X,LB,*_=run.base.setup(idx,Z,L,[]);U=np.c_[np.ones(m*m)/m,X/np.sqrt(m)]
 ps=np.array([np.r_[.001/(m*m),np.full(2*m-2,.001/m)],np.r_[.001/(m*m),(.001+.3*np.diag(LB))/m]])
 return U,ps,LB

def cavity_t(m,f):
 d=np.arange(m);w=binom.pmf(d,m-1,f)
 if (m-1)*f<=1:return 0.
 return brentq(lambda t:np.sum(w*d/(1+d*t))-1,0.,1.)

def mode_risks(g,ps,b2,s,m):
 # g can be an array of effective observation precisions in whitened coordinates.
 p=ps[:,1:];b2=b2[1:];g=np.atleast_1d(g)
 return 2*s+np.sum(((b2-s)[None,None,:]*p[None,:,:]**2+s*g[:,None,None])/(g[:,None,None]+p[None,:,:])**2,axis=2)/(m*(m-1))

def closures(m,fs,ps,b2,s):
 d=np.arange(m);pm=binom.pmf(d[None,:],m-1,fs[:,None]);res={}
 res['mean']=mode_risks((m-1)*fs/m,ps,b2,s,m)
 res['degree']=pm@mode_risks(d/m,ps,b2,s,m)
 ts=np.array([cavity_t(m,f) for f in fs]);res['cavity']=np.array([ww@mode_risks(d*t/m,ps,b2,s,m) for ww,t in zip(pm,ts)])
 return res

def main(paths=128):
 start=time.time();pilots=pd.read_csv(ROOT/'reports/descriptor_joint_uncertainty/posterior_components_full_metric_kernel.csv');pilots=pilots[(pilots.coverage==.12)&(pilots.seed==0)].set_index('panel')
 fs=np.round(np.arange(.01,.901,.005),8);weights={m:binom.pmf(np.arange(m*m-1)[None,:],m*m-2,fs[:,None]) for m in MS}
 coefs={m:[] for m in MS};meta=[];curves=[];panels=[];snap={}
 for ix,(pair,yfull,Zfull) in enumerate(load_grids()):
  if pair not in pilots.index:continue
  Uf,pf,LBf=setup(19,Zfull);pp=pilots.loc[pair];vf=np.zeros(37);sp=pp.AB_variance_mean*(19/18)**2
  for k,site in enumerate(['A','B']):
   eig=np.diag(LBf)[18*k:18*(k+1)];w=1/(.001/.3+np.maximum(eig,0));w/=w.mean();vf[1+18*k:1+18*(k+1)]=361/18*pp[site+'_variance_mean']*(1-pp[site+'_smooth_fraction']+pp[site+'_smooth_fraction']*w)
  rng=np.random.default_rng(517001+ix);ia=rng.permutation(19);ib=rng.permutation(19);panels.append(pair)
  for m in MS:
   ra=np.sort(ia[:m]);rb=np.sort(ib[:m]);yy=yfull[np.ix_(ra,rb)];Z={'A':Zfull['A'][ra],'B':Zfull['B'][rb]};U,ps,LB=setup(m,Z)
   y=yy.ravel()-yy.mean();b=U.T@y;ym=U@b;yu=y-ym;s=np.dot(yu,yu)/(m-1)**2
   ids=(ra[:,None]*19+rb[None,:]).ravel();T=U.T@Uf[ids];V=sp*np.eye(2*m-1)+(T*(vf-sp))@T.T
   edges=[];orders=[]
   for t in range(paths):
    a,c=rng.choice(m,2,replace=False);bg=int(rng.integers(m))
    i,j=(a*m+bg,c*m+bg) if rng.random()<.5 else (bg*m+a,bg*m+c)
    edges.append((i,j));orders.append(rng.permutation(np.setdiff1d(np.arange(m*m),[i,j])))
   co=exact_paths(U,ps,ym,yu,b,V,s,sp,np.array(edges),np.array(orders));coefs[m].append(co)
   ex=np.einsum('fn,njk->fjk',weights[m],co)
   meta.append(dict(panel=pair,m=m,sigma=s,pilot_sigma=sp,main_variance=np.mean(ym*ym),interaction_variance=np.mean(yu*yu)))
   snap[(pair,m)]={'p':ps,'b2':b*b,'Vdiag':np.diag(V)}
   for k,eta in enumerate(ETAS):
    rr={'observed_mse':ex[:,:,2*k],'observed_mae':ex[:,:,2*k+1], 'oracle_mse':ex[:,:,6]+eta*eta*s*ex[:,:,7],'oracle_mae':ex[:,:,10+k]}
    rr.update({name+'_mse':r for name,r in closures(m,fs,ps,b*b,s*eta*eta).items()})
    if eta==1:
     rr['pilot_mse']=ex[:,:,8];rr['pilot_mae']=ex[:,:,9]
     rr.update({'pilot_'+name+'_mse':r for name,r in closures(m,fs,ps,np.diag(V),sp).items()})
    for name,r in rr.items():
     for fi,f in enumerate(fs):curves.append((pair,m,eta,f,name,r[fi,0],r[fi,1],100*(1-r[fi,1]/r[fi,0])))
  if ix%20==0:print(ix+1,round(time.time()-start,1),flush=True)
 for m in MS:
  np.savez_compressed(OUT/f'coefficients_m{m}.npz',coefficients=np.stack(coefs[m]),panels=np.array(panels),fs=fs,penalties=np.stack([snap[p,m]['p'] for p in panels]),b2=np.stack([snap[p,m]['b2'] for p in panels]),Vdiag=np.stack([snap[p,m]['Vdiag'] for p in panels]))
 d=pd.DataFrame(curves,columns=['panel','m','eta','coverage','model','ridge','metric','gain_pct']);d.to_csv(OUT/'curves.csv',index=False);pd.DataFrame(meta).to_csv(OUT/'grid_parameters.csv',index=False)
 med=d.groupby(['m','eta','coverage','model']).gain_pct.median().reset_index();med.to_csv(OUT/'median_curves.csv',index=False)
 pool=d.groupby(['m','eta','coverage','model'])[['ridge','metric']].mean().reset_index();pool['gain_pct']=100*(1-pool.metric/pool.ridge);pool.to_csv(OUT/'pooled_curves.csv',index=False)
 print('seconds',time.time()-start)
 (OUT/'run.json').write_text(json.dumps({'paths':paths,'grids':len(panels),'sizes':MS,'etas':ETAS,'seconds':time.time()-start,'pilot':'existing seed0 12% full-grid hyperparameters transported by covariance restriction'},indent=2))
if __name__=='__main__':main()
