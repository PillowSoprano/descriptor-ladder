"""Parameter-free branch susceptibility closure via one-dimensional Laplace integrals."""
import numpy as np,pandas as pd,time
from scipy.stats import binom
from scipy.optimize import brentq
from numpy.polynomial.legendre import leggauss
from derive_connectivity_window import OUT,MS

def branch_quadrature(m,f,order=96):
 k=m-1
 if k*f<=1:return None
 q=brentq(lambda q:(1-(1-f*q)**k)/q-1,1e-10,1.)
 prob=f*q;ds=np.arange(1,k+1);ww=binom.pmf(ds,k,prob);ww/=ww.sum()
 t=brentq(lambda t:np.sum(ww*ds/(1+ds*t))-1,0.,1.)
 h=ds*t/(1+ds*t)
 xx,gg=leggauss(order);T=40/h[0];x=(xx+1)*np.log1p(T)/2;tt=np.expm1(x);weight=gg*np.log1p(T)/2*np.exp(x)
 avg=np.exp(-tt[:,None]*h[None,:])@ww
 atom=(1-prob)**k
 excess=atom*np.expm1(k*np.log1p(prob*avg/(1-prob)))
 return tt,weight*excess,atom,q,t

def risk(m,ps,b2,s,quad):
 beta=b2[1:]/m;lam=ps[:,1:]*m
 if quad is None:return np.full(2,2*s+np.sum(beta-s/m)/(m-1))
 tt,wt,atom,_,_=quad;ee=np.exp(-tt[:,None,None]*lam[None,:,:]);I1=np.sum(wt[:,None,None]*ee,axis=0);I2=np.sum((wt*tt)[:,None,None]*ee,axis=0)
 B=beta-s/m
 return 2*s+np.sum(B[None,:]*atom+s*I1+(B[None,:]*lam*lam-s*lam)*I2,axis=1)/(m-1)

def main():
 start=time.time();meta=pd.read_csv(OUT/'grid_parameters.csv').set_index(['panel','m']);rows=[]
 for m in MS:
  z=np.load(OUT/f'coefficients_m{m}.npz');fs=z['fs'];geo=[branch_quadrature(m,f) for f in fs]
  for j,pair in enumerate(z['panels']):
   for source in ['oracle','pilot']:
    b=z['b2'][j] if source=='oracle' else z['Vdiag'][j];s=meta.loc[(pair,m),'sigma' if source=='oracle' else 'pilot_sigma'];ps=z['penalties'][j]
    for eta in ([.5,1,1.5] if source=='oracle' else [1.]):
     for f,quad in zip(fs,geo):
      r=risk(m,ps,b,s*eta*eta,quad);rows.append((pair,m,eta,f,source+'_branch_mse',r[0],r[1],100*(1-r[1]/r[0])))
  print(m,round(time.time()-start,1),flush=True)
 d=pd.DataFrame(rows,columns=['panel','m','eta','coverage','model','ridge','metric','gain_pct']);d.to_csv(OUT/'branch_curves.csv',index=False)
 med=d.groupby(['m','eta','coverage','model']).gain_pct.median().reset_index();med.to_csv(OUT/'branch_median_curves.csv',index=False)
 from summarize_connectivity_window import closing
 vals=[]
 for (m,eta,model),g in med.groupby(['m','eta','model']):vals.append([m,eta,model,closing(g.coverage.to_numpy(),g.gain_pct.to_numpy()),g.gain_pct.max()])
 pd.DataFrame(vals,columns=['m','eta','model','closing','peak_gain']).to_csv(OUT/'branch_boundaries.csv',index=False)
 print(pd.DataFrame(vals,columns=['m','eta','model','closing','peak_gain']).round(4).to_string(index=False));print('seconds',time.time()-start)
if __name__=='__main__':main()
