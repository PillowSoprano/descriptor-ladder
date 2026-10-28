"""Exact main-effect and interaction risk polynomials on the GB1 designs.

The complete-grid variance decomposition explains the measured curves. The designs, penalties and
response sampling of the protein sweep are kept.
"""
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.linalg import solve
import run_gain_by_block as run

ROOT = run.ROOT
OUT = ROOT / 'reports/shrinkage_window'

def mae_windows(a0, b0, a1, b1, upper=3.):
    """Exact piecewise-linear positive-gain intervals on [0, upper]."""
    ts=[]; jumps=[]
    for a,b,sgn in [(a0,b0,1.),(a1,b1,-1.)]:
        nz=np.abs(b)>1e-14
        t=-a[nz]/b[nz]; keep=(t>1e-12)&(t<upper)
        ts.extend(t[keep]); jumps.extend(sgn*2*np.abs(b[nz][keep])/len(a))
    order=np.argsort(ts); t=np.r_[0.,np.asarray(ts)[order],upper]
    jump=np.asarray(jumps)[order]
    slope=float(np.mean(np.where(a0==0,np.abs(b0),np.sign(a0)*b0)-np.where(a1==0,np.abs(b1),np.sign(a1)*b1)))
    slopes=np.r_[slope,slope+np.cumsum(jump)]
    vals=np.r_[np.mean(np.abs(a0)-np.abs(a1)),np.zeros(len(t)-1)]
    vals[1:]=vals[0]+np.cumsum(slopes*np.diff(t))
    cuts=[0.,upper]
    for i in np.flatnonzero(vals[:-1]*vals[1:]<0):
        cuts.append(float(t[i]-vals[i]/slopes[i]))
    cuts=np.unique(cuts); windows=[]
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        mid=(lo+hi)/2
        gain=np.mean(np.abs(a0+mid*b0)-np.abs(a1+mid*b1))
        if gain>1e-12: windows.append([float(lo),float(hi)])
    check=np.mean(np.abs(a0+upper*b0)-np.abs(a1+upper*b1))
    return windows,abs(vals[-1]-check)

def main():
    OUT.mkdir(exist_ok=True); start=time.time();rows=[]
    hist=pd.read_csv(ROOT/'reports/protein_replication.csv')
    raw=pd.read_csv(ROOT/'data/raw/gb1/gb1.tsv',sep='\t');db=raw[raw.num_mutations==2].copy()
    parts=db.variant.str.split(',',expand=True)
    for i in [1,2]:
        db[f'p{i}']=parts[i-1].str.extract(r'(\d+)').astype(int);db[f'a{i}']=parts[i-1].str[-1]
    aa=sorted(set(db.a1)|set(db.a2));za=run.protein.descriptors(aa);lookup={v:i for i,v in enumerate(aa)}
    for pi,(pair,hg) in enumerate(hist.groupby('pair',sort=False)):
        ap,bp=map(int,pair.split('-'))
        M=db[(db.p1==ap)&(db.p2==bp)].pivot_table(index='a1',columns='a2',values='score')
        Z={'A':za[[lookup[v] for v in M.index]],'B':za[[lookup[v] for v in M.columns]]}
        L={s:run.protein.laplacian(z,3) for s,z in Z.items()}
        idx={'A':np.repeat(np.arange(19),19),'B':np.tile(np.arange(19),19)}
        X,LB,*_=run.base.setup(idx,Z,L,[])
        m=M.to_numpy();main=(m.mean(0)[None,:]+m.mean(1)[:,None]-m.mean()).ravel();inter=m.ravel()-main
        penalties={'ridge':.001*np.eye(36),'metric':.001*np.eye(36)+.3*LB}
        for h in hg.itertuples():
            frac=float(h.frac);seed=int(h.seed);rng=np.random.default_rng(ap*100003+bp*1009+seed*31+int(frac*1000))
            mask=rng.random(361)<frac;edges=[]
            for i in range(19):
                for j in range(i+1,19):
                    for k in range(19):
                        for u,v in [(i*19+k,j*19+k),(k*19+i,k*19+j)]:
                            if not mask[u] and not mask[v]:edges.append((u,v))
            e=np.asarray(edges)
            if len(e)>4000:e=e[rng.choice(len(e),4000,replace=False)]
            errors={}
            for name,y in [('main',main),('interaction',inter)]:
                pred,_=run.model_predictions(X,y[mask],mask,penalties,.001)
                errors[name]={k:(y[e[:,1]]-y[e[:,0]])-(v[e[:,1]]-v[e[:,0]]) for k,v in pred.items()}
            a0=errors['main']['ridge'];a1=errors['main']['metric'];b0=errors['interaction']['ridge'];b1=errors['interaction']['metric']
            A=float(np.mean(a0*a0-a1*a1));B=float(np.mean(a0*b0-a1*b1));C=float(np.mean(b0*b0-b1*b1))
            r0=a0+b0;r1=a1+b1;delta=r0-r1;step=float(np.mean(delta**2));align=float(np.mean(delta*r0))
            windows,err=mae_windows(a0,b0,a1,b1)
            row={'panel':pair,'coverage':frac,'seed':seed,'n_train':int(mask.sum()),'A_main':A,'two_B_cross':2*B,'C_interaction':C,
                 'mse_gain':A+2*B+C,'ridge_mse':float(np.mean(r0*r0)),'alignment':align,'step_cost':step,
                 'alignment_ratio':2*align/max(step,1e-30),'gain_pct':100*(1-np.mean(abs(r1))/np.mean(abs(r0))),
                 'historical_gain_pct':100*(1-h.metric/h.free),'mse_polynomial_error':abs(A+2*B+C-np.mean(r0*r0-r1*r1)),
                 'mae_piecewise_error':err,'mae_windows_eta_0_3':json.dumps(windows),'n_mae_windows':len(windows),
                 'eta_mae_onset':windows[0][0] if windows else np.inf,
                 'mse_cross_changes_sign':bool((A+C)*(A+2*B+C)<0),'cross_positive':bool(B>0)}
            for t in [0.,.5,1.,2.]:
                x0=a0+t*b0;x1=a1+t*b1
                row[f'gain_eta_{t:g}']=100*(1-np.mean(abs(x1))/max(np.mean(abs(x0)),1e-30))
            for name,val in [('main',A),('cross',2*B),('interaction',C)]:row[name+'_normalized']=100*val/max(row['ridge_mse'],1e-30)
            # Geometry intervention: match the original metric smoother's total
            # effective degrees of freedom with isotropic ridge, using X only.
            xt=X[mask];n=len(xt);ss=xt.sum(0);tau=.001
            G=xt.T@xt-np.outer(ss,ss)/(n+tau)
            J=G-tau*np.outer(ss,ss)/(n+tau)**2
            target=float(np.trace(solve(G+penalties['metric']+1e-8*np.eye(36),J,assume_a='pos')))
            ge,gu=np.linalg.eigh(G);jd=np.diag(gu.T@J@gu)
            alpha=float(np.exp(brentq(lambda z:np.sum(jd/(ge+np.exp(z)+1e-8))-target,-25,15)))
            matched=[]
            for yy in [main,inter]:
                pp,_=run.model_predictions(X,yy[mask],mask,{'matched':alpha*np.eye(36)},.001)
                matched.append((yy[e[:,1]]-yy[e[:,0]])-(pp['matched'][e[:,1]]-pp['matched'][e[:,0]]))
            c0,d0=matched;rr=c0+d0;den=np.mean(rr*rr)
            row.update({'matched_ridge_alpha':alpha,'matched_effective_df':n/(n+tau)+target,
                'matched_gain_pct':100*(1-np.mean(abs(r1))/np.mean(abs(rr))),
                'matched_main_normalized':100*np.mean(c0*c0-a1*a1)/den,
                'matched_cross_normalized':200*np.mean(c0*d0-a1*b1)/den,
                'matched_interaction_normalized':100*np.mean(d0*d0-b1*b1)/den})
            rows.append(row)
        if pi%20==0:print(pi+1,len(rows),round(time.time()-start,1),flush=True)
    d=pd.DataFrame(rows);d.to_csv(OUT/'protein_exact_decomposition.csv',index=False)
    cols=['gain_pct','gain_eta_0','gain_eta_0.5','gain_eta_2','main_normalized','cross_normalized','interaction_normalized','alignment_ratio','mse_cross_changes_sign']
    curves=d.groupby(['coverage','panel'])[cols].median().groupby('coverage').median()
    curves.to_csv(OUT/'protein_window_curves.csv');print(curves.round(4).to_string(),flush=True)
    mcols=[c for c in d if c.startswith('matched_')]
    d.groupby(['coverage','panel'])[mcols].median().groupby('coverage').median().to_csv(OUT/'matched_geometry_curves.csv')
    d.groupby(['coverage','panel'])['eta_mae_onset'].median().groupby('coverage').median().to_csv(OUT/'mae_threshold_curve.csv')
    summary={'settings':len(d),'grids':d.panel.nunique(),'elapsed_seconds':time.time()-start,
             'max_historical_difference_pp':float(abs(d.gain_pct-d.historical_gain_pct).max()),
             'max_polynomial_error':float(d.mse_polynomial_error.max()),'max_mae_piecewise_error':float(d.mae_piecewise_error.max()),
             'fraction_cross_changes_sign':float(d.mse_cross_changes_sign.mean()),
             'fraction_multiple_mae_windows_0_3':float((d.n_mae_windows>1).mean()),
             'role':'Complete-grid explanatory decomposition.'}
    (OUT/'protein_summary.json').write_text(json.dumps(summary,indent=2));print(summary)

if __name__=='__main__':main()
