"""Expected substitution error under the posterior of each fit, for the block attribution.

Only training labels enter. The omitted interactions carry either an independent covariance or the
highest-order ANOVA covariance, and absolute errors are integrated pair by pair.
"""
from pathlib import Path
import sys,os,argparse,time,json
import numpy as np,pandas as pd
from scipy.linalg import solve,cho_factor,cho_solve
from scipy.optimize import nnls,minimize
from scipy.special import erf
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
import run_descriptor_crossover as base
import run_regularization_baselines as mh
import run_protein_replication as protein
base.GEOMETRY='graph'
OUT=ROOT/'reports/gain_by_block';OUT.mkdir(exist_ok=True)
ESTIMATOR='moment'
SMOOTH_INTERACTIONS=False
EXTRA_FEATURES=None
EXTRA_WEIGHTS=None

def set_extra_features(idx,qs,L):
    global EXTRA_FEATURES,EXTRA_WEIGHTS
    if not SMOOTH_INTERACTIONS:
        EXTRA_FEATURES=EXTRA_WEIGHTS=None;return
    features=np.ones((len(next(iter(idx.values()))),1));weights=np.ones(1)
    for s in idx:
        features=np.einsum('ni,nj->nij',features,qs[s][idx[s]]).reshape(len(features),-1)
        eig=np.maximum(np.diag(qs[s].T@L[s]@qs[s]),0)
        weights=np.kron(weights,1/(1+eig))
    EXTRA_FEATURES=features;EXTRA_WEIGHTS=weights/weights.mean()

def interaction_covariance(idx):
    n=len(next(iter(idx.values())));R=np.ones((n,n))
    for v in idx.values():
        levels=int(v.max()+1);R*=((v[:,None]==v[None,:]).astype(float)-1/levels)/(1-1/levels)
    return R

def moment_components(Xt,y,W,Rt):
    xc=Xt-Xt.mean(0);yc=y-y.mean();G=xc.T@xc;rhs=xc.T@yc
    rc=Rt-Rt.mean(0)[None,:]-Rt.mean(1)[:,None]+Rt.mean();k=W.shape[1]
    T=np.zeros((k+1,k+1));z=np.zeros(k+1)
    T[:k,:k]=W.T@(G*G)@W;z[:k]=W.T@(rhs*rhs)
    xr=xc.T@rc@xc;T[:k,k]=T[k,:k]=W.T@np.diag(xr)
    T[k,k]=np.sum(rc*rc);z[k]=yc@rc@yc
    e,U=np.linalg.eigh(G);good=e>max(e.max(),1)*1e-10;df=len(y)-1-int(good.sum());fixed=None
    if df>=10:
        rss=max(float(yc@yc-np.sum((U[:,good].T@rhs)**2/e[good])),0)
        residual_trace=float(np.trace(rc)-np.sum((U[:,good]*(xr@U[:,good])).sum(0)/e[good]))
        if residual_trace>1e-10:fixed=rss/residual_trace
    if fixed is None:t,zf=T,z
    else:t,zf=T[:k,:k],z[:k]-fixed*T[:k,k]
    sc=np.sqrt(np.maximum(np.diag(t),1e-20));tn=t/sc[:,None]/sc[None,:];zn=zf/sc
    e,U=np.linalg.eigh(tn);keep=e>1e-11
    w,_=nnls(np.sqrt(e[keep])[:,None]*U[:,keep].T,(U[:,keep].T@zn)/np.sqrt(e[keep]),maxiter=2000)
    v=w/sc
    if fixed is not None:v=np.r_[v,fixed]
    return W@v[:-1],float(v[-1]),v,df

def components(Xt,y,W,Rt):
    initial=moment_components(Xt,y,W,Rt)
    if ESTIMATOR=='moment':return initial
    # Restricted likelihood learns covariance using training labels, including
    # uncertainty in the unknown constant mean. Components are scaled to unit
    # mean diagonal for numerical conditioning, not fitted to prediction gains.
    n=len(y);ys=max(float(np.var(y)),1e-12);yc=(y-y.mean())/np.sqrt(ys)
    xc=Xt-Xt.mean(0);ge,gu=np.linalg.eigh(xc.T@xc);keep=ge>max(ge.max(),1)*1e-10
    Q=(xc@gu[:,keep])/np.sqrt(ge[keep]);yp=Q.T@yc;df=n-1-Q.shape[1]
    xq=Q.T@xc;rq=Q.T@Rt@Q
    kernels=np.stack([(xq*W[:,j])@xq.T for j in range(W.shape[1])]+[rq])
    # For both IID and the highest-order ANOVA field, the covariance is
    # isotropic on the complement of the fitted main/pair design. Integrate
    # that complement analytically; this is the same REML, with smaller solves.
    rc_trace=float(np.trace(Rt)-Rt.sum()/n)
    residual_scale=max((rc_trace-np.trace(rq))/df,1e-12) if df else 0.
    rss=max(float(yc@yc-yp@yp),0.) if df else 0.
    traces=np.trace(kernels,axis1=1,axis2=2);traces[-1]+=df*residual_scale
    scales=np.maximum(traces/(n-1),1e-12);kernels/=scales[:,None,None]
    residual_scale/=scales[-1]
    start=np.log(np.maximum(initial[2]*scales/ys,1e-4))
    eye=np.eye(Q.shape[1])
    def objective(logv):
        v=np.exp(logv);K=np.einsum('k,kij->ij',v,kernels)+1e-8*eye
        cf=cho_factor(K,lower=True,check_finite=False);inv=cho_solve(cf,eye,check_finite=False)
        alpha=inv@yp
        value=.5*(2*np.log(np.diag(cf[0])).sum()+yp@alpha)
        gradient=.5*v*np.einsum('ij,kij->k',inv-np.outer(alpha,alpha),kernels)
        if df:
            rv=v[-1]*residual_scale+1e-8
            value+=.5*(df*np.log(rv)+rss/rv)
            gradient[-1]+=.5*(df/rv-rss/rv**2)*v[-1]*residual_scale
        return value,gradient
    fit=minimize(objective,start,jac=True,method='L-BFGS-B',bounds=[(-16,8)]*len(start),options={'maxiter':80,'ftol':1e-8,'gtol':1e-5})
    v=np.exp(fit.x)*ys/scales
    return W@v[:-1],float(v[-1]),v,initial[3]

def posterior_hidden(X,ytrain,mask,cov,sigma,R):
    train=np.flatnonzero(mask);hidden=np.flatnonzero(~mask);xt=X[mask];xh=X[~mask]
    Ktt=(xt*cov)@xt.T+sigma*R[np.ix_(train,train)]
    Kht=(xh*cov)@xt.T+sigma*R[np.ix_(hidden,train)]
    Khh=(xh*cov)@xh.T+sigma*R[np.ix_(hidden,hidden)]
    # Numerical nugget only: scale-free and far below the fitted component scale.
    nugget=max(float(np.trace(Ktt)/len(train)),float(np.var(ytrain)),1e-12)*1e-9
    Ktt.flat[::len(train)+1]+=nugget
    f=cho_factor(Ktt,lower=True,check_finite=False)
    ai=cho_solve(f,np.ones(len(train)),check_finite=False);den=float(ai.sum())
    intercept=float(ai@ytrain/den);alpha=cho_solve(f,ytrain-intercept,check_finite=False)
    mean=intercept+Kht@alpha
    correction=1-Kht@ai
    V=Khh-Kht@cho_solve(f,Kht.T,check_finite=False)+np.outer(correction,correction)/den
    return mean,(V+V.T)*.5,hidden

def folded_mae(mean,var):
    sd=np.sqrt(np.maximum(var,1e-20));a=abs(mean);z=a/sd
    return sd*np.sqrt(2/np.pi)*np.exp(-.5*z*z)+a*erf(z/np.sqrt(2))

def model_predictions(X,y,mask,penalties,intercept_penalty):
    xt=X[mask];xc=xt-xt.mean(0);G=xc.T@xc;rhs=xc.T@(y-y.mean());pred={};betas={}
    for name,P in penalties.items():
        if intercept_penalty:
            aug=np.column_stack([np.ones(len(y)),xt]);pa=np.zeros((X.shape[1]+1,X.shape[1]+1));pa[0,0]=intercept_penalty;pa[1:,1:]=P
            beta=solve(aug.T@aug+pa+np.eye(pa.shape[0])*1e-8,aug.T@y,assume_a='pos',check_finite=False)[1:]
        else:beta=solve(G+P+np.eye(G.shape[0])*1e-8,rhs,assume_a='pos',check_finite=False)
        pred[name]=X@beta;betas[name]=beta
    return pred,betas

def forecast(X,ytrain,mask,edges,axes,W,labels,blocks,R,penalties,intercept_penalty):
    xp=X;wp=W;lp=labels
    if EXTRA_FEATURES is not None:
        xp=np.column_stack([X,EXTRA_FEATURES]);wp=np.zeros((xp.shape[1],W.shape[1]+1));wp[:len(W),:-1]=W;wp[len(W):,-1]=EXTRA_WEIGHTS
        lp=labels+['omitted_smooth']
    cov,sigma,v,df=components(xp[mask],ytrain,wp,R[np.ix_(mask,mask)])
    mu,V,hidden=posterior_hidden(xp,ytrain,mask,cov,sigma,R)
    lookup=np.full(len(X),-1,dtype=int);lookup[hidden]=np.arange(len(hidden));ee=lookup[edges]
    assert np.min(ee)>=0
    mean=mu[ee[:,1]]-mu[ee[:,0]]
    var=np.maximum(V[ee[:,0],ee[:,0]]+V[ee[:,1],ee[:,1]]-2*V[ee[:,0],ee[:,1]],0)
    pred,betas=model_predictions(X,ytrain,mask,penalties,intercept_penalty)
    edgepred={k:p[edges[:,1]]-p[edges[:,0]] for k,p in pred.items()}
    risks={k:folded_mae(mean-p,var) for k,p in edgepred.items()}
    rec={'sigma2':sigma,'residual_df':df,'ridge_forecast_mae':float(risks['ridge'].mean()),'metric_forecast_mae':float(risks['metric'].mean()),'posterior_contrast_variance':float(var.mean()),'posterior_contrast_mean_square':float(np.mean(mean**2))}
    rec['forecast_gain_pct']=100*(1-rec['metric_forecast_mae']/max(rec['ridge_forecast_mae'],1e-30))
    E=X[edges[:,1]]-X[edges[:,0]];delta=betas['metric']-betas['ridge'];mid=.5*(edgepred['metric']+edgepred['ridge']);attribution=[]
    for b,ii in blocks.items():
        d=E[:,ii]@delta[ii]
        attribution.append({'block':b,'predicted_mse_reduction':float(np.mean(2*d*(mean-mid)))})
    for s in np.unique(axes):
        m=axes==s
        rec[s+'_forecast_gain_pct']=100*(1-risks['metric'][m].mean()/max(risks['ridge'][m].mean(),1e-30))
    return rec,dict(zip(lp+['omitted_interaction'],v)),edgepred,attribution,(E,delta,mid)

def score(rec,pred,edges,axes,yall,attr,detail):
    truth=yall[edges[:,1]]-yall[edges[:,0]]
    for name,p in pred.items():
        rec[name+'_actual_mae']=float(np.mean(abs(truth-p)))
        rec[name+'_actual_mse']=float(np.mean((truth-p)**2))
    rec['actual_gain_pct']=100*(1-rec['metric_actual_mae']/max(rec['ridge_actual_mae'],1e-30))
    for s in np.unique(axes):
        m=axes==s
        rec[s+'_actual_gain_pct']=100*(1-np.mean(abs(truth[m]-pred['metric'][m]))/max(np.mean(abs(truth[m]-pred['ridge'][m])),1e-30))

def materials(pilot=False,noise='interaction'):
    frame=pd.read_parquet(ROOT/'data/interim/castelli_raw.parquet')
    if 'magnetic_moment' not in frame or frame.magnetic_moment.isna().all():frame['magnetic_moment']=frame['_magmom_signed' if '_magmom_signed' in frame else '_magmom'].abs()
    hist=pd.read_csv(ROOT/'reports/regularization_panel_curves.csv');rows=[];pars=[];attrs=[];start=time.time()
    for panel,aa,bb in mh.panel_definitions(frame):
        sub=frame[frame.site_A.isin(aa)&frame.site_B.isin(bb)].reset_index(drop=True);levels={s:sorted(sub['site_'+s].unique()) for s in mh.SITES}
        desc={s:mh.el.physical_descriptors(levels[s],'element' if s in ['A','B'] else 'auto') for s in levels}
        Z={s:np.nan_to_num(desc[s].matrix(levels[s],allow_missing=True)) for s in levels};L={s:mh.hybrid.level_laplacian(desc[s],levels[s],k=3) for s in levels}
        idx={s:sub['site_'+s].map({l:i for i,l in enumerate(levels[s])}).to_numpy() for s in levels}
        X,Lbig,W,labels,blocks,qs=base.setup(idx,Z,L,list(mh.hybrid.PAIRS));eye=np.eye(X.shape[1]);R=interaction_covariance(idx) if noise=='interaction' else np.eye(len(X))
        set_extra_features(idx,qs,L)
        for frac in ([.05,.1,.2,.5] if pilot else mh.FRACS):
            for seed in ([0] if pilot else mh.SEEDS):
                split=mh.obs.random_nodes(sub,frac,seed=seed);mask=sub.sample_id.isin(split.observed_ids).to_numpy();edges,axes=base.edges_for(idx,mask)
                for prop in mh.PROPERTIES:
                    h=hist[(hist.panel==panel)&(hist.property==prop)&np.isclose(hist.coverage,frac)&(hist.seed==seed)].set_index('model')
                    lr=float(h.loc['ordinary_ridge','selected'].split('=')[1]);lm=float(h.loc['descriptor_metric','selected'].split('=')[1]);y=sub[prop].to_numpy()
                    key={'panel':panel,'property':prop,'coverage':frac,'seed':seed,'n_train':int(mask.sum()),'noise_model':noise}
                    rec,pa,pred,at,de=forecast(X,y[mask],mask,edges,axes,W,labels,blocks,R,{'ridge':lr*eye,'metric':lm*Lbig},0)
                    score(rec,pred,edges,axes,y,at,de)
                    rec['historical_gain_pct']=100*(1-h.loc['descriptor_metric','mae_both_hidden']/h.loc['ordinary_ridge','mae_both_hidden'])
                    E,delta,mid=de;truth=y[edges[:,1]]-y[edges[:,0]]
                    for a in at:a['actual_mse_reduction']=float(np.mean(2*(E[:,blocks[a['block']]]@delta[blocks[a['block']]])*(truth-mid)))
                    rows.append({**key,**rec});pars.append({**key,**pa});attrs.extend([{**key,**a} for a in at])
        tag=('pilot_' if pilot else '')+noise+('_reml' if ESTIMATOR=='reml' else '')
        if SMOOTH_INTERACTIONS:tag+='_smooth'
        pd.DataFrame(rows).to_csv(OUT/f'materials_conditional_{tag}.csv',index=False);pd.DataFrame(pars).to_csv(OUT/f'materials_components_{tag}.csv',index=False);pd.DataFrame(attrs).to_csv(OUT/f'materials_attribution_{tag}.csv',index=False)
        print(panel,len(rows),f'{time.time()-start:.1f}s',flush=True)
        if pilot:break

def proteins(noise='interaction'):
    hist=pd.read_csv(ROOT/'reports/protein_replication.csv');raw=pd.read_csv(ROOT/'data/raw/gb1/gb1.tsv',sep='\t');db=raw[raw.num_mutations==2].copy();parts=db.variant.str.split(',',expand=True)
    db['p1']=parts[0].str.extract(r'(\d+)').astype(int);db['a1']=parts[0].str[-1];db['p2']=parts[1].str.extract(r'(\d+)').astype(int);db['a2']=parts[1].str[-1]
    allaa=sorted(set(db.a1)|set(db.a2));Zall=protein.descriptors(allaa);lookup={a:i for i,a in enumerate(allaa)};rows=[];pars=[];start=time.time()
    for pi,(pair,hgroup) in enumerate(hist.groupby('pair',sort=False)):
        a,b=map(int,pair.split('-'));M=db[(db.p1==a)&(db.p2==b)].pivot_table(index='a1',columns='a2',values='score')
        Z={'A':Zall[[lookup[k] for k in M.index]],'B':Zall[[lookup[k] for k in M.columns]]};L={s:protein.laplacian(z,3) for s,z in Z.items()};idx={'A':np.repeat(np.arange(19),19),'B':np.tile(np.arange(19),19)}
        X,Lbig,W,labels,blocks,qs=base.setup(idx,Z,L,[]);eye=np.eye(X.shape[1]);y=M.to_numpy().ravel();R=interaction_covariance(idx) if noise=='interaction' else np.eye(len(X))
        set_extra_features(idx,qs,L)
        for _,h in hgroup.iterrows():
            frac=float(h.frac);seed=int(h.seed);rng=np.random.default_rng(a*100003+b*1009+seed*31+int(frac*1000));mask=rng.random(361)<frac;edges=[];axes=[]
            for i in range(19):
                for j in range(i+1,19):
                    for k in range(19):
                        for u,v,s in [(i*19+k,j*19+k,'A'),(k*19+i,k*19+j,'B')]:
                            if not mask[u] and not mask[v]:edges.append((u,v));axes.append(s)
            edges=np.asarray(edges);axes=np.asarray(axes)
            if len(edges)>4000:
                ix=rng.choice(len(edges),4000,replace=False);edges=edges[ix];axes=axes[ix]
            key={'panel':pair,'property':'fitness','coverage':frac,'seed':seed,'n_train':int(mask.sum()),'noise_model':noise}
            rec,pa,pred,at,de=forecast(X,y[mask],mask,edges,axes,W,labels,blocks,R,{'ridge':protein.LAM*eye,'metric':protein.LAM*eye+protein.SMOOTH*Lbig},protein.LAM)
            score(rec,pred,edges,axes,y,at,de);rec['historical_gain_pct']=100*(1-h.metric/h.free)
            rows.append({**key,**rec});pars.append({**key,**pa})
        if pi%20==0:print(pi+1,len(rows),f'{time.time()-start:.1f}s',flush=True)
    tag=noise+('_reml' if ESTIMATOR=='reml' else '')
    if SMOOTH_INTERACTIONS:tag+='_smooth'
    pd.DataFrame(rows).to_csv(OUT/f'protein_conditional_{tag}.csv',index=False);pd.DataFrame(pars).to_csv(OUT/f'protein_components_{tag}.csv',index=False)
    print('protein complete',len(rows),f'{time.time()-start:.1f}s',flush=True)

if __name__=='__main__':
    os.chdir(ROOT);ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');ap.add_argument('--protein',action='store_true');ap.add_argument('--noise',choices=['iid','interaction'],default='interaction');ap.add_argument('--estimator',choices=['moment','reml'],default='moment');ap.add_argument('--smooth-interactions',action='store_true');args=ap.parse_args();ESTIMATOR=args.estimator;SMOOTH_INTERACTIONS=args.smooth_interactions
    if args.protein:proteins(args.noise)
    else:materials(args.pilot,args.noise)
