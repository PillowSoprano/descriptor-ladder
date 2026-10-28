"""Training-only variance components -> scalar crossover and finite-design risk.

No held-out response or complete-grid ANOVA enters estimate_components/risk.
Historical training-selected penalties are reused, not fitted to outer gains.
The covariance estimate is nonnegative Frobenius moment matching of centered yy'.
"""
from pathlib import Path
import sys, os, json, time, argparse
import numpy as np
import pandas as pd
from scipy.linalg import helmert, block_diag, solve
from scipy.optimize import nnls
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
import run_regularization_baselines as mh
import run_protein_replication as protein
OUT=ROOT/'reports/descriptor_crossover';OUT.mkdir(exist_ok=True)
GEOMETRY='span'
NOISE='moment'
TAG='span'

def site_coords(Z,L):
    h=helmert(len(Z),full=False).T
    u,s,_=np.linalg.svd(h.T@(Z-Z.mean(0)),full_matrices=True)
    if GEOMETRY=='graph': _,u=np.linalg.eigh(h.T@L@h)
    rank=int((s>max(s.max(),1)*1e-10).sum())
    q=h@u
    return q, q.T@L@q,rank

def setup(indices,Zs,Ls,pairs):
    sites=list(Zs);qs={};ls={};ranks={};cols=[];groups=[];labels=[];penalties=[];blocks={};offset=0
    for s in sites:qs[s],ls[s],ranks[s]=site_coords(Zs[s],Ls[s])
    for name in sites+pairs:
        if name in sites:
            x=qs[name][indices[name]];p=ls[name];capt=np.arange(x.shape[1])<ranks[name]
        else:
            a,b=name
            x=np.einsum('ni,nj->nij',qs[a][indices[a]],qs[b][indices[b]]).reshape(len(indices[a]),-1)
            p=np.kron(ls[a],np.eye(qs[b].shape[1]))+np.kron(np.eye(qs[a].shape[1]),ls[b])
            capt=(np.arange(qs[a].shape[1])[:,None]<ranks[a])&(np.arange(qs[b].shape[1])[None,:]<ranks[b]);capt=capt.ravel()
        cols.append(x);penalties.append(p);blocks[name]=np.arange(offset,offset+x.shape[1])
        if GEOMETRY=='span':
            for tag,mask in [('span',capt),('outside',~capt)]:
                if mask.any():groups.append((offset,mask.astype(float)));labels.append(name+'_'+tag)
        else:
            smooth=1/(1+np.maximum(np.diag(p),0));smooth/=smooth.mean()
            for tag,w in [('isotropic',np.ones(x.shape[1])),('smooth',smooth)]:
                groups.append((offset,w));labels.append(name+'_'+tag)
        offset+=x.shape[1]
    W=np.zeros((offset,len(groups)))
    for k,(off,w) in enumerate(groups):W[off:off+len(w),k]=w
    return np.column_stack(cols),block_diag(*penalties),W,labels,blocks,qs

def design_state(X,mask):
    xt=X[mask];mean=xt.mean(0);xc=xt-mean;G=xc.T@xc
    return xc,G,mean

def estimate_components(xc,G,y,groups):
    yc=y-y.mean();rhs=xc.T@yc;W=groups;k=W.shape[1];T=np.zeros((k+1,k+1));z=np.zeros(k+1)
    z[:k]=W.T@(rhs**2);T[:k,:k]=W.T@(G**2)@W
    T[:k,k]=T[k,:k]=W.T@np.diag(G)
    T[k,k]=len(y)-1;z[k]=yc@yc
    fixed_sigma=None
    if NOISE=='residual':
        ge,gu=np.linalg.eigh(G);keep=ge>max(ge.max(),1)*1e-10;df=len(y)-1-int(keep.sum())
        if df>=10:
            rss=max(float(yc@yc-np.sum((gu[:,keep].T@rhs)**2/ge[keep])),0)
            fixed_sigma=rss/df
    if fixed_sigma is not None:
        zfit=z[:k]-fixed_sigma*T[:k,k];Tfit=T[:k,:k]
    else:zfit=z;Tfit=T
    scale=np.sqrt(np.maximum(np.diag(Tfit),1e-20));Tn=Tfit/scale[:,None]/scale[None,:];zn=zfit/scale
    eig,U=np.linalg.eigh(Tn);good=eig>1e-11
    A=np.sqrt(eig[good])[:,None]*U[:,good].T
    b=(U[:,good].T@zn)/np.sqrt(eig[good])
    w,_=nnls(A,b,maxiter=1000);v=w/scale
    if fixed_sigma is not None:v=np.r_[v,fixed_sigma]
    cov=W@v[:-1]
    return cov,float(v[-1]),v,rhs

def edges_for(indices,mask):
    sites=list(indices);pairs=[];axis=[]
    for si,s in enumerate(sites):
        others=[indices[o] for o in sites if o!=s]
        codes=np.ravel_multi_index(tuple(others),tuple(int(a.max()+1) for a in others))
        for c in np.unique(codes):
            ix=np.flatnonzero((codes==c)&~mask)
            if len(ix)<2:continue
            i,j=np.triu_indices(len(ix),1);pairs.extend(zip(ix[i],ix[j]));axis.extend([s]*len(i))
    return np.asarray(pairs,dtype=int),np.asarray(axis)

def evaluate(X,G,xc,mask,edgepairs,axes,ytrain,yall,cov,sigma,penalties,mean,intercept_penalty=0.0):
    # Q only uses the design and identities of unseen compositions, not their labels.
    E=X[edgepairs[:,1]]-X[edgepairs[:,0]];Q=E.T@E/len(E)
    rhs=xc.T@(ytrain-ytrain.mean());d=G.shape[0];eye=np.eye(d);result={};riskmats={}
    for name,P in penalties.items():
        if intercept_penalty:
            xt=np.column_stack([np.ones(mask.sum()),X[mask]]);Gt=xt.T@xt;Pt=block_diag(intercept_penalty,P)
            At=solve(Gt+Pt+1e-8*np.eye(d+1),np.eye(d+1),assume_a='pos',check_finite=False)
            St=At@Gt-np.eye(d+1);ct=np.r_[0.,cov];mu=np.r_[ytrain.mean(),np.zeros(d)]
            rt=(St*ct)@St.T+sigma*(At@Gt@At.T)+np.outer(St@mu,St@mu)
            R=rt[1:,1:];b=(At@xt.T@ytrain)[1:]
        else:
            A=solve(G+P+1e-8*eye,eye,assume_a='pos',check_finite=False)
            S=A@G-eye;R=(S*cov)@S.T+sigma*(A@G@A.T)
            b=A@rhs
        predmse=max(float(np.sum(Q*R))+2*sigma,0)
        # Labels enter only here, after prediction is formed.
        err=(yall[edgepairs[:,1]]-yall[edgepairs[:,0]])-E@b
        result[name+'_pred_mse']=predmse;result[name+'_actual_mse']=float(np.mean(err**2));result[name+'_actual_mae']=float(np.mean(abs(err)))
        riskmats[name]=R
        for s in np.unique(axes):
            Es=E[axes==s];Qs=Es.T@Es/len(Es)
            result[name+'_'+s+'_pred_mse']=max(float(np.sum(Qs*R))+2*sigma,0)
            result[name+'_'+s+'_actual_mae']=float(np.mean(abs(err[axes==s])))
    result['predicted_gain_pct']=100*(1-np.sqrt(result['metric_pred_mse']/max(result['ridge_pred_mse'],1e-30)))
    result['actual_gain_pct']=100*(1-result['metric_actual_mae']/max(result['ridge_actual_mae'],1e-30))
    result['actual_rmse_gain_pct']=100*(1-np.sqrt(result['metric_actual_mse']/max(result['ridge_actual_mse'],1e-30)))
    for s in np.unique(axes):
        result[s+'_predicted_gain_pct']=100*(1-np.sqrt(result['metric_'+s+'_pred_mse']/max(result['ridge_'+s+'_pred_mse'],1e-30)))
        result[s+'_actual_gain_pct']=100*(1-result['metric_'+s+'_actual_mae']/max(result['ridge_'+s+'_actual_mae'],1e-30))
    return result

def scalar_stats(cov,sigma,groups,labels,blocks,indices,X,qs,Z):
    res={};vars_by_block={name:float(np.sum(np.mean(X[:,ii]**2,axis=0)*cov[ii])) for name,ii in blocks.items()}
    # Marginal-effect scalar approximation treats omitted pair interactions as residual.
    residual=sigma+sum(v for k,v in vars_by_block.items() if len(k)>1)
    res['scalar_residual_variance']=residual
    for site in indices:
        nlev=int(indices[site].max()+1);ii=blocks[site];tau=float(cov[ii].sum()/nlev)
        zz=Z[site]-Z[site].mean(0);proj=zz@np.linalg.pinv(zz)
        capture=np.sum(qs[site]*(proj@qs[site]),axis=0)
        bias=float(np.sum(cov[ii]*(1-capture))/nlev);bias=max(bias,0)
        rho=1-bias/tau if tau>0 else np.nan
        nstar=residual/bias if bias>0 else np.nan
        res.update({site+'_tau2':tau,site+'_rho':rho,site+'_bias2':bias,site+'_nstar':nstar})
    return res

def run_materials(limit=False):
    frame=pd.read_parquet(ROOT/'data/interim/castelli_raw.parquet')
    if 'magnetic_moment' not in frame or frame.magnetic_moment.isna().all():
        frame['magnetic_moment']=frame['_magmom_signed' if '_magmom_signed' in frame else '_magmom'].abs()
    old=pd.read_csv(ROOT/'reports/regularization_panel_curves.csv');rows=[];params=[];start=time.time()
    for panel,aa,bb in mh.panel_definitions(frame):
        sub=frame[frame.site_A.isin(aa)&frame.site_B.isin(bb)].reset_index(drop=True)
        levels={s:sorted(sub['site_'+s].unique()) for s in mh.SITES}
        desc={s:mh.el.physical_descriptors(levels[s],'element' if s in ['A','B'] else 'auto') for s in mh.SITES}
        Z={s:np.nan_to_num(desc[s].matrix(levels[s],allow_missing=True)) for s in levels}
        L={s:mh.hybrid.level_laplacian(desc[s],levels[s],k=3) for s in levels}
        idx={s:sub['site_'+s].map({l:i for i,l in enumerate(levels[s])}).to_numpy() for s in levels}
        X,Lbig,groups,labels,blocks,qs=setup(idx,Z,L,list(mh.hybrid.PAIRS));eye=np.eye(X.shape[1])
        for frac in ([.05,.2,.5] if limit else mh.FRACS):
            for seed in ([0] if limit else mh.SEEDS):
                sp=mh.obs.random_nodes(sub,frac,seed=seed);mask=sub.sample_id.isin(sp.observed_ids).to_numpy()
                xc,G,mean=design_state(X,mask);pairs,axes=edges_for(idx,mask)
                for prop in mh.PROPERTIES:
                    y=sub[prop].to_numpy(float);cov,sigma,v,rhs=estimate_components(xc,G,y[mask],groups)
                    hist=old[(old.panel==panel)&(old.property==prop)&np.isclose(old.coverage,frac)&(old.seed==seed)].set_index('model')
                    lr=float(hist.loc['ordinary_ridge','selected'].split('=')[1]);lm=float(hist.loc['descriptor_metric','selected'].split('=')[1])
                    scores=evaluate(X,G,xc,mask,pairs,axes,y[mask],y,cov,sigma,{'ridge':lr*eye,'metric':lm*Lbig},mean)
                    row={'domain':'materials','panel':panel,'property':prop,'coverage':frac,'seed':seed,'n_train':int(mask.sum()),'n_parameters':X.shape[1]+1,'sigma2':sigma,'lambda_ridge':lr,'lambda_metric':lm,**scores,**scalar_stats(cov,sigma,groups,labels,blocks,idx,X,qs,Z)}
                    row['historical_gain_pct']=100*(1-hist.loc['descriptor_metric','mae_both_hidden']/hist.loc['ordinary_ridge','mae_both_hidden'])
                    rows.append(row)
                    params.append({'panel':panel,'property':prop,'coverage':frac,'seed':seed,**dict(zip(labels+['residual'],v))})
        pd.DataFrame(rows).to_csv(OUT/('materials_pilot.csv' if limit else f'materials_{TAG}_predictions.csv'),index=False)
        pd.DataFrame(params).to_csv(OUT/('components_pilot.csv' if limit else f'materials_{TAG}_components.csv'),index=False)
        print(f'materials {panel}: {len(rows)} cases, {time.time()-start:.1f}s',flush=True)
        if limit:break

def run_proteins():
    old=pd.read_csv(ROOT/'reports/protein_replication.csv');raw=pd.read_csv(ROOT/'data/raw/gb1/gb1.tsv',sep='\t')
    db=raw[raw.num_mutations==2].copy();parts=db.variant.str.split(',',expand=True)
    db['p1']=parts[0].str.extract(r'(\d+)').astype(int);db['a1']=parts[0].str[-1]
    db['p2']=parts[1].str.extract(r'(\d+)').astype(int);db['a2']=parts[1].str[-1]
    allaa=sorted(set(db.a1)|set(db.a2));Zall=protein.descriptors(allaa);lookup={a:i for i,a in enumerate(allaa)}
    rows=[];start=time.time()
    for pi,(pair,hist) in enumerate(old.groupby('pair',sort=False)):
        a,b=map(int,pair.split('-'));sub=db[(db.p1==a)&(db.p2==b)];M=sub.pivot_table(index='a1',columns='a2',values='score')
        Z={'A':Zall[[lookup[k] for k in M.index]],'B':Zall[[lookup[k] for k in M.columns]]}
        L={s:protein.laplacian(z,3) for s,z in Z.items()};idx={'A':np.repeat(np.arange(19),19),'B':np.tile(np.arange(19),19)}
        X,Lbig,groups,labels,blocks,qs=setup(idx,Z,L,[]);eye=np.eye(X.shape[1]);y=M.to_numpy().ravel()
        for _,h in hist.iterrows():
            frac=float(h.frac);seed=int(h.seed);rng=np.random.default_rng(a*100003+b*1009+seed*31+int(frac*1000));mask=rng.random(361)<frac
            # Preserve the historical alternating A/B edge order before subsampling.
            pairs=[];axes=[]
            for i in range(19):
                for j in range(i+1,19):
                    for k in range(19):
                        for u,v,s in [(i*19+k,j*19+k,'A'),(k*19+i,k*19+j,'B')]:
                            if not mask[u] and not mask[v]:pairs.append((u,v));axes.append(s)
            pairs=np.asarray(pairs);axes=np.asarray(axes)
            if len(pairs)>4000:
                sample=rng.choice(len(pairs),4000,replace=False);pairs=pairs[sample];axes=axes[sample]
            xc,G,mean=design_state(X,mask);cov,sigma,v,rhs=estimate_components(xc,G,y[mask],groups)
            scores=evaluate(X,G,xc,mask,pairs,axes,y[mask],y,cov,sigma,{'ridge':protein.LAM*eye,'metric':protein.LAM*eye+protein.SMOOTH*Lbig},mean,intercept_penalty=protein.LAM)
            row={'domain':'protein','panel':pair,'property':'fitness','coverage':frac,'seed':seed,'n_train':int(mask.sum()),'n_parameters':37,'sigma2':sigma,**scores,**scalar_stats(cov,sigma,groups,labels,blocks,idx,X,qs,Z)}
            row['historical_gain_pct']=100*(1-h.metric/h.free);rows.append(row)
        if pi%20==0:print(f'protein {pi+1}/120: {len(rows)} cases, {time.time()-start:.1f}s',flush=True)
    pd.DataFrame(rows).to_csv(OUT/f'protein_{TAG}_predictions.csv',index=False)
    print(f'protein complete {len(rows)} cases, {time.time()-start:.1f}s',flush=True)

if __name__=='__main__':
    os.chdir(ROOT);p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--protein',action='store_true');p.add_argument('--geometry',choices=['span','graph'],default='span');p.add_argument('--noise',choices=['moment','residual'],default='moment');a=p.parse_args();GEOMETRY=a.geometry;NOISE=a.noise;TAG=GEOMETRY if NOISE=='moment' else GEOMETRY+'_'+NOISE
    if a.protein:run_proteins()
    else:run_materials(a.pilot)
