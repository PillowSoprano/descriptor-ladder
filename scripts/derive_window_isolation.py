"""Add analytically computed chemical borrowing on isolated levels; no gain-fit coefficients."""
import numpy as np,pandas as pd,time
from scipy.optimize import brentq
from scipy.special import lambertw
from derive_connectivity_window import OUT,MS,ROOT,load_grids,setup
from summarize_connectivity_window import closing

def main():
 start=time.time();pilots=pd.read_csv(ROOT/'reports/descriptor_joint_uncertainty/posterior_components_full_metric_kernel.csv');pilots=pilots[(pilots.coverage==.12)&(pilots.seed==0)].set_index('panel');pars=[]
 for ix,(pair,yfull,Zfull) in enumerate(load_grids()):
  Uf,pf,LBf=setup(19,Zfull);pp=pilots.loc[pair];vf=np.zeros(37);sp=pp.AB_variance_mean*(19/18)**2
  for k,site in enumerate(['A','B']):
   eig=np.diag(LBf)[18*k:18*(k+1)];w=1/(.001/.3+np.maximum(eig,0));w/=w.mean();vf[1+18*k:1+18*(k+1)]=361/18*pp[site+'_variance_mean']*(1-pp[site+'_smooth_fraction']+pp[site+'_smooth_fraction']*w)
  rng=np.random.default_rng(517001+ix);ia=rng.permutation(19);ib=rng.permutation(19)
  for m in MS:
   ra=np.sort(ia[:m]);rb=np.sort(ib[:m]);yy=yfull[np.ix_(ra,rb)];Z={'A':Zfull['A'][ra],'B':Zfull['B'][rb]};U,ps,LB=setup(m,Z);y=yy.ravel()-yy.mean();b=U.T@y;s=np.sum((y-U@b)**2)/(m-1)**2
   ids=(ra[:,None]*19+rb[None,:]).ravel();T=U.T@Uf[ids];V=sp*np.eye(2*m-1)+(T*(vf-sp))@T.T
   for source,Vb,ss in [('oracle',np.outer(b,b),s),('pilot',V,sp)]:
    H=0.
    for axis in range(2):
     hs=[]
     for level in range(m):
      mask=(np.arange(m*m)//m==level) if axis==0 else (np.arange(m*m)%m==level)
      v=U[mask].sum(0)/np.sqrt(m);vals=[]
      for p in ps:
       pv=p*v;vals.append(pv@Vb@pv/(m*(v@pv)**2))
      hs.append(vals[0]-vals[1])
     H+=np.mean(hs)
    lam=ps[1,1:]*m;beta=np.diag(Vb)[1:]/m
    for eta in ([.5,1,1.5] if source=='oracle' else [1.]):
     noise=ss*eta*eta;A=np.sum((beta-noise/m)*lam**2-2*noise*lam)/(m-1);D=np.sum(2*(beta-noise/m)*lam**3-3*noise*lam**2)/(m-1)
     pars.append(dict(panel=pair,m=m,source=source,eta=eta,H=H,A=A,D=D,sigma=noise))
 p=pd.DataFrame(pars);p.to_csv(OUT/'isolation_parameters.csv',index=False)
 branch=pd.read_csv(OUT/'branch_curves.csv');branch['source']=branch.model.str.split('_').str[0];d=branch.merge(p,on=['panel','m','eta','source'],validate='many_to_one');q=(1-d.coverage)**(d.m-1);d['isolation_gain']=q*(1-q)*d.H
 d['metric']=d.metric-d.isolation_gain;d['gain_pct']=100*(1-d.metric/d.ridge);d['model']=d.source+'_branch_isolation_mse';d.to_csv(OUT/'isolation_curves.csv',index=False)
 med=d.groupby(['m','eta','coverage','model']).gain_pct.median().reset_index();med.to_csv(OUT/'isolation_median_curves.csv',index=False)
 vals=[]
 for (m,e,mod),g in med.groupby(['m','eta','model']): vals.append([m,e,mod,closing(g.coverage.to_numpy(),g.gain_pct.to_numpy()),g.gain_pct.max()])
 pd.DataFrame(vals,columns=['m','eta','model','closing','peak_gain']).to_csv(OUT/'isolation_boundaries.csv',index=False)
 print(pd.DataFrame(vals,columns=['m','eta','model','closing','peak_gain']).round(4).to_string(index=False));print(p.groupby(['source','m','eta'])[['H','A','D']].median().round(3).to_string());print('seconds',time.time()-start)
if __name__=='__main__':main()
