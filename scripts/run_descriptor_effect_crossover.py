"""Direct experiment on the scalar claim: partial-data effect estimation,
projection, and Gaussian shrinkage, evaluated against complete-grid main effects.
The full-grid effects enter only the last scoring block.
"""
from pathlib import Path
import sys,time
import numpy as np,pandas as pd
from scipy.linalg import helmert
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
import run_regularization_baselines as mh
OUT=ROOT/'reports/descriptor_crossover'

def run():
    frame=pd.read_parquet(ROOT/'data/interim/castelli_raw.parquet')
    if 'magnetic_moment' not in frame or frame.magnetic_moment.isna().all():frame['magnetic_moment']=frame['_magmom_signed' if '_magmom_signed' in frame else '_magmom'].abs()
    panels=mh.panel_definitions(frame)+[('full-52',sorted(frame.site_A.unique()),sorted(frame.site_B.unique()))]
    rows=[];skips=[];start=time.time()
    for panel,aa,bb in panels:
        sub=frame[frame.site_A.isin(aa)&frame.site_B.isin(bb)].reset_index(drop=True);levels={s:sorted(sub['site_'+s].unique()) for s in mh.SITES}
        H={s:helmert(len(levels[s]),full=False).T for s in levels};idx={s:sub['site_'+s].map({v:i for i,v in enumerate(levels[s])}).to_numpy() for s in levels}
        X=np.column_stack([np.ones(len(sub))]+[H[s][idx[s]] for s in levels]);slices={};off=1;P={}
        for s in levels:
            slices[s]=slice(off,off+len(levels[s])-1);off=slices[s].stop
            desc=mh.el.physical_descriptors(levels[s],'element' if s in ['A','B'] else 'auto');Z=np.nan_to_num(desc.matrix(levels[s],allow_missing=True));Z=Z-Z.mean(0)
            P[s]=Z@np.linalg.pinv(Z)
        fracs=(.01,.015,.02,.03,.05,.1,.2,.3,.5,.7,.9) if panel=='full-52' else (.03,.04,.05,.075,.1,.15,.2,.3,.5,.7,.9)
        for frac in fracs:
            for seed in range(5):
                split=mh.obs.random_nodes(sub,frac,seed=seed);mask=sub.sample_id.isin(split.observed_ids).to_numpy();xt=X[mask];G=xt.T@xt
                if mask.sum()<=X.shape[1]+2 or np.linalg.matrix_rank(G)<X.shape[1]:
                    skips.append({'panel':panel,'coverage':frac,'seed':seed,'n_train':int(mask.sum()),'reason':'additive free effects not estimable with residual degrees of freedom'});continue
                Gi=np.linalg.inv(G)
                for prop in mh.PROPERTIES:
                    yt=sub.loc[mask,prop].to_numpy();beta=Gi@xt.T@yt;sigma=float(np.sum((yt-xt@beta)**2)/(len(yt)-X.shape[1]))
                    for s in ['A','B']:
                        nlev=len(levels[s]);h=H[s];sl=slices[s];mu=h@beta[sl];V=sigma*h@Gi[sl,sl]@h.T;projector=P[s];perp=np.eye(nlev)-projector
                        vf=float(np.trace(V)/nlev);vp=float(np.trace(projector@V)/nlev);vu=max(vf-vp,0)
                        rawb=float(mu@perp@mu/nlev-vu);bias=max(rawb,0);rawtau=float(mu@mu/nlev-vf);tau=max(rawtau,bias,0)
                        rho=1-bias/tau if tau>0 else np.nan;weight=bias/(bias+vu) if bias+vu>0 else 0
                        pb=projector@mu;shrink=pb+weight*(perp@mu)
                        pred_free=vf;pred_basis=vp+bias;pred_shrink=vp+bias*vu/(bias+vu) if bias+vu>0 else vp
                        ne=mask.sum()/nlev;nstar_naive=sigma/bias if bias>0 else np.nan;nstar_design=ne*vu/bias if bias>0 else np.nan
                        # Evaluation labels: not used in any quantity above.
                        truth=sub.groupby('site_'+s)[prop].mean().reindex(levels[s]).to_numpy()-sub[prop].mean()
                        af=float(np.mean((mu-truth)**2));ab=float(np.mean((pb-truth)**2));ash=float(np.mean((shrink-truth)**2))
                        truebias=float(np.mean((perp@truth)**2));truetau=float(np.mean(truth**2))
                        rows.append({'panel':panel,'property':prop,'site':s,'coverage':frac,'seed':seed,'n_train':int(mask.sum()),'n_effective':ne,'n_levels':nlev,'descriptor_rank':int(round(np.trace(projector))),'sigma2':sigma,'tau2':tau,'rho':rho,'raw_bias2':rawb,'bias2':bias,'nstar_naive':nstar_naive,'nstar_design':nstar_design,'crossover_fraction':frac*vu/bias if bias>0 else np.nan,'free_pred_mse':pred_free,'basis_pred_mse':pred_basis,'shrink_pred_mse':pred_shrink,'free_actual_mse':af,'basis_actual_mse':ab,'shrink_actual_mse':ash,'predicted_basis_advantage':pred_free-pred_basis,'actual_basis_advantage':af-ab,'true_bias2_evaluation_only':truebias,'true_rho_evaluation_only':1-truebias/truetau if truetau>0 else np.nan})
        pd.DataFrame(rows).to_csv(OUT/'effect_crossovers.csv',index=False);pd.DataFrame(skips).to_csv(OUT/'effect_nonidentifiable_designs.csv',index=False)
        print(panel,len(rows),f'{time.time()-start:.1f}s',flush=True)
if __name__=='__main__':run()
