import numpy as np,pandas as pd,json,math
from scipy.optimize import brentq
from scipy.stats import binom
from derive_connectivity_window import OUT,MS
from derive_bipartite_connectivity import connected_table
from summarize_connectivity_window import closing

def main():
 etas=np.round(np.arange(.3,1.801,.025),6);records=[];cur=pd.read_csv(OUT/'isolation_curves.csv');pars=pd.read_csv(OUT/'isolation_parameters.csv');result=[]
 for m in MS:
  z=np.load(OUT/f'coefficients_m{m}.npz');f=z['fs'];co=z['coefficients'];w=binom.pmf(np.arange(m*m-1)[None,:],m*m-2,f[:,None]);rr=np.einsum('fn,pnjk->pfjk',w,co)
  C=2*(rr[:,:,:,4]-2*rr[:,:,:,2]+rr[:,:,:,0]);B=rr[:,:,:,4]-rr[:,:,:,0]-2*C;A=rr[:,:,:,2]-B-C
  cc=cur[(cur.m==m)&(cur.source=='oracle')]
  # Each analytical risk is affine in eta^2; evaluate exactly from two stored values.
  lo=cc[cc.eta==.5].pivot_table(index='panel',columns='coverage',values=['ridge','metric']).reindex(z['panels']);hi=cc[cc.eta==1].pivot_table(index='panel',columns='coverage',values=['ridge','metric']).reindex(z['panels'])
  rlo=np.stack([lo['ridge'].to_numpy(),lo['metric'].to_numpy()],axis=-1);rhi=np.stack([hi['ridge'].to_numpy(),hi['metric'].to_numpy()],axis=-1)
  Cv=(rhi-rlo)/.75;Av=rhi-Cv
  for eta in etas:
   for name,r in [('observed',A+eta*B+eta*eta*C),('compact',Av+eta*eta*Cv)]:
    g=np.median(100*(1-r[:,:,1]/r[:,:,0]),axis=0);root=closing(f,g)
    records.append(dict(m=m,eta=eta,model=name,closing=root,peak_gain=g.max(),peak_f=f[g.argmax()],state='absent' if np.isnan(root) else ('beyond90' if np.isinf(root) else 'closed')))
  comb=np.zeros((m+1,m+1))
  for a in range(m+1):
   for b in range(a+1):comb[a,b]=math.comb(a,b)
  conn=brentq(lambda f:connected_table(m,f,comb)[m,m]-.5,.01,.9)
  o=[r for r in records if r['m']==m and r['eta']==1 and r['model']=='observed'][0]
  p=[r for r in records if r['m']==m and r['eta']==1 and r['model']=='compact'][0]
  result.append(dict(m=m,observed_mse_closing=o['closing'],compact_mse_closing=p['closing'],connection50=conn))
 d=pd.DataFrame(records);d.to_csv(OUT/'interaction_phase.csv',index=False)
 t=pd.DataFrame(result);t.to_csv(OUT/'size_comparison.csv',index=False)
 phase=[]
 for (m,model),sub in d.groupby(['m','model']):
  present=sub[sub.state!='absent'];late=sub[sub.state=='beyond90']
  phase.append(dict(m=m,model=model,first_positive_eta=float(present.eta.min()) if len(present) else None,first_beyond90_eta=float(late.eta.min()) if len(late) else None))
 summary={'size_comparison':result,'mean_absolute_boundary_error_pp':float(np.mean(abs(t.observed_mse_closing-t.compact_mse_closing))*100),'connection50_error_pp':float(np.mean(abs(t.observed_mse_closing-t.connection50))*100),'phase':phase,'interpretation':'Compact formula evaluated with complete-grid signal parameters; transport to training-only estimates is reported separately.'}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=lambda x:x.item()));print(json.dumps(summary,indent=2,default=lambda x:x.item()))
if __name__=='__main__':main()
