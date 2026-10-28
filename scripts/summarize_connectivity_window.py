import json,math
import numpy as np,pandas as pd
from scipy.optimize import brentq
from derive_connectivity_window import OUT,MS,cavity_t

def closing(f,g):
 k=int(np.argmax(g))
 if g[k]<=1e-8:return np.nan
 for i in range(k,len(f)-1):
  if g[i]>0 and g[i+1]<=0:return float(f[i]-g[i]*(f[i+1]-f[i])/(g[i+1]-g[i]))
 return np.inf

def main():
 meta=pd.read_csv(OUT/'grid_parameters.csv').set_index(['panel','m']);rows=[]
 for m in MS:
  z=np.load(OUT/f'coefficients_m{m}.npz');panels=z['panels'];p=z['penalties'][:,1,1:]*m;bs=z['b2'][:,1:]/m;vp=z['Vdiag'][:,1:]/m
  leff=np.sum(p*p,axis=1)/np.sum(p,axis=1)
  for name,b in [('oracle',bs),('pilot',vp)]:
   rough=np.sum(b*p*p,axis=1)/np.sum(p*p,axis=1)
   for eta in ([.5,1.,1.5] if name=='oracle' else [1.]):
    for j,pair in enumerate(panels):
     ss=meta.loc[(pair,m),'sigma' if name=='oracle' else 'pilot_sigma']*eta*eta
     chi=rough[j]/ss-1/m-2/leff[j]
     conduct=1/chi if chi>0 else np.inf
     mean=conduct/(m-1);tree=(1+conduct)/(m-1)
     cav=np.inf
     if np.isfinite(conduct) and (m-1)*.999*cavity_t(m,.999)>conduct:
      cav=brentq(lambda f:(m-1)*f*cavity_t(m,f)-conduct,1/(m-1)+1e-9,.999)
     rows.append(dict(panel=pair,m=m,eta=eta,source=name,roughness=rough[j],lambda_eff=leff[j],sigma=ss,chi=chi,mean_boundary=mean,tree_boundary=tree,cavity_boundary=cav))
 pd.DataFrame(rows).to_csv(OUT/'closed_form_predictions.csv',index=False)
 tabs=[]
 for agg in ['median','pooled']:
  d=pd.read_csv(OUT/f'{agg}_curves.csv')
  for (m,e,model),g in d.groupby(['m','eta','model']):
   tabs.append(dict(aggregation=agg,m=m,eta=e,model=model,closing=closing(g.coverage.to_numpy(),g.gain_pct.to_numpy()),peak_coverage=g.loc[g.gain_pct.idxmax(),'coverage'],peak_gain=g.gain_pct.max()))
 d=pd.DataFrame(tabs);d.to_csv(OUT/'boundary_summary.csv',index=False)
 print('MEDIAN MSE CLOSINGS')
 print(d[(d.aggregation=='median')&d.model.isin(['observed_mse','oracle_mse','mean_mse','degree_mse','cavity_mse'])].pivot(index=['m','eta'],columns='model',values='closing').round(3).to_string())
 print('MOMENT CLOSING MEDIANS')
 print(pd.DataFrame(rows).groupby(['source','m','eta'])[['mean_boundary','tree_boundary','cavity_boundary','chi']].median().round(3).to_string())
if __name__=='__main__':main()
