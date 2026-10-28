"""Summaries and plot sources of the block attribution."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'reports/gain_by_block'
FIG=ROOT/'reports/gain_by_block/figure_sources';FIG.mkdir(parents=True,exist_ok=True)
OLD=ROOT/'reports/descriptor_crossover'
variants={'conditional_iid':'iid','conditional_interaction':'interaction','conditional_reml':'interaction_reml'}
rows=[];metrics=[];frames={}
for domain,n in [('materials',567),('protein',1759)]:
    choices=dict(variants)
    if domain=='protein':
        choices['smooth_interaction']='interaction_reml_smooth'
        choices['noise_uncertainty']='interaction'
    for name,tag in choices.items():
        source=DATA/'noise_uncertainty' if name=='noise_uncertainty' else DATA
        d=pd.read_csv(source/f'{domain}_conditional_{tag}.csv');assert len(d)==n,(domain,tag,len(d))
        frames[domain,name]=d
        for prop,g in d.groupby('property'):
            panel=g.groupby(['coverage','panel'])[['forecast_gain_pct','historical_gain_pct']].median()
            curve=panel.groupby('coverage').median()
            for frac,r in curve.iterrows():rows.append({'domain':domain,'property':prop,'coverage':frac,'method':name,'prediction':r.forecast_gain_pct,'observation':r.historical_gain_pct})
            metrics.append({'domain':domain,'property':prop,'method':name,'n_cases':len(g),'curve_MAE_pp':float(abs(curve.forecast_gain_pct-curve.historical_gain_pct).mean()),'case_MAE_pp':float(abs(g.forecast_gain_pct-g.historical_gain_pct).mean()),'median_case_absolute_error_pp':float(abs(g.forecast_gain_pct-g.historical_gain_pct).median()),'case_spearman':float(g[['forecast_gain_pct','historical_gain_pct']].corr(method='spearman').iloc[0,1]),'case_sign_accuracy':float(np.mean((g.forecast_gain_pct>0)==(g.historical_gain_pct>0))),'source_reproduction_max_pp':float(abs(g.actual_gain_pct-g.historical_gain_pct).max())})
    old=pd.read_csv(OLD/f'{domain}_graph_residual_predictions.csv')
    for prop,g in old.groupby('property'):
        curve=g.groupby(['coverage','panel'])[['predicted_gain_pct','historical_gain_pct']].median().groupby('coverage').median()
        for frac,r in curve.iterrows():rows.append({'domain':domain,'property':prop,'coverage':frac,'method':'previous_integrated','prediction':r.predicted_gain_pct,'observation':r.historical_gain_pct})
        metrics.append({'domain':domain,'property':prop,'method':'previous_integrated','n_cases':len(g),'curve_MAE_pp':float(abs(curve.predicted_gain_pct-curve.historical_gain_pct).mean()),'case_MAE_pp':float(abs(g.predicted_gain_pct-g.historical_gain_pct).mean())})
curves=pd.DataFrame(rows);curves.to_csv(DATA/'curve_summary.csv',index=False)
pd.DataFrame(metrics).to_csv(DATA/'prediction_metrics.csv',index=False)
for prop,g in curves.groupby('property'):
    wide=g.pivot(index='coverage',columns='method',values='prediction');wide['observed']=g.groupby('coverage').observation.first();wide['pct']=100*wide.index
    wide.reset_index(drop=True).to_csv(FIG/f'gain_{prop}.dat',sep=' ',index=False,float_format='%.8g')
attr=pd.read_csv(DATA/'materials_attribution_interaction_reml.csv')
assert len(attr)==567*6
means=attr.groupby(['property','coverage','block'])[['predicted_mse_reduction','actual_mse_reduction']].mean().reset_index()
means.to_csv(DATA/'block_attribution_mean.csv',index=False)
summary={'phase':'block attribution','materials_cases_per_method':567,'protein_cases_per_method':1759,'metrics':metrics,'aggregation':'Median seeds within panel, then median panels, matching original figures. Curve error is the mean absolute deviation at the 9 materials or 8 protein coverage points.','training_information':'Only the outer training labels enter covariance fitting and the posterior expected errors. Original inner-training-selected penalties are reused. This predicts risk conditional on a computed training set, not before any response has been obtained.','method_selection':'Exploratory development using observed aggregate discrepancies; no gain-curve parameter is fitted.','smooth_materials_branch':'12-case alphabetical trial only; retained as a separate branch, not extrapolated to seven panels.'}
summary['protein_noise_uncertainty']='Hold training REML main-effect covariance fixed. Integrate AB variance under restricted training likelihood and an exponential prior with mean Var(y_train), using 16 posterior-quantile nodes. All 1,759 cases.'
summary['counterfactual']='Full-grid interaction variance enters no plot or metric of the training-only expected errors.'
(DATA/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
(FIG/'figure_contract.json').write_text(json.dumps({'backend':'LaTeX/PGFPlots','archetype':'quantitative grid','question':'Does conditional directional risk with ANOVA interactions recover the measured chemical-prior gain curves?','evidence':'Original observations against prior integrated risk, conditional moment risk, and conditional REML risk. Protein additionally integrates omitted-interaction variance uncertainty.','statistics':summary['aggregation'],'export':'Editable TeX, vector PDF and adjacent source-data DAT files; about 17 cm wide.','scope':'Three material properties on seven panels and 120 protein grids. Materials use tuned ridge baseline; proteins use historical fixed ridge baseline.','no_curve_fitting':True},indent=2))
titles={'heat_of_formation':r'\textbf{a} Formation energy: 7 grids','volume_per_fu':r'\textbf{b} Volume: 7 grids','magnetic_moment':r'\textbf{c} Magnetic moment: 7 grids','fitness':r'\textbf{d} Protein fitness: 120 grids'}
limits={'heat_of_formation':(-4,13),'volume_per_fu':(-5,32),'magnetic_moment':(-3,3),'fitness':(-10,15)}
tex=r'''\documentclass[border=6pt]{standalone}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots}
\pgfplotsset{compat=1.18}
\definecolor{moment}{RGB}{60,110,170}
\definecolor{reml}{RGB}{175,65,50}
\definecolor{smooth}{RGB}{65,130,110}
\begin{document}
\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 2,horizontal sep=1.5cm,vertical sep=1.6cm},width=7.5cm,height=5.2cm,
 xmode=log,xmin=2.5,xmax=100,xtick={3,5,10,20,50,90},xticklabels={3,5,10,20,50,90},
 xlabel={computed compositions (\%)},ylabel={MAE removed (\%)},
 tick label style={font=\scriptsize},label style={font=\footnotesize},title style={font=\footnotesize},
 legend style={draw=none,font=\scriptsize},legend columns=2,legend cell align=left]
'''
for i,prop in enumerate(titles):
    lo,hi=limits[prop];tex+=f'\\nextgroupplot[title={{{titles[prop]}}},ymin={lo},ymax={hi}'+(',legend to name=sharedlegend' if i==0 else '')+']\n'
    for style,col,label in [('only marks,black,mark=*,mark size=2','observed','Observed'),('black!45,dashed,thick','previous_integrated','Previous integrated risk'),('moment,dotted,very thick','conditional_interaction','Conditional / moments'),('reml,thick','conditional_reml','Conditional / REML')]:
        tex+=f'\\addplot[{style}] table[x=pct,y={col}]{{gain_{prop}.dat}};\n'
        if i==0:tex+=f'\\addlegendentry{{{label}}}\n'
    if prop=='fitness':tex+='\\addplot[color=smooth,dashdotted,thick] table[x=pct,y=noise_uncertainty]{gain_fitness.dat};\n'
    tex+='\\addplot[black!20,thin] coordinates{(2.5,0)(100,0)};\n'
tex+=r'''\end{groupplot}
\node[anchor=north] at ([yshift=-1.15cm]group c1r2.south east) {\ref{sharedlegend}};
\node[anchor=north,font=\scriptsize,align=center] at ([yshift=-2.02cm]group c1r2.south east)
 {Protein green dash-dot: integrate interaction-variance uncertainty.\\Expected errors use training labels and the selected penalties; points are historical measurements.};
\end{tikzpicture}
\end{document}
'''
(FIG/'interaction_gain_prediction.tex').write_text(tex)
print(pd.DataFrame(metrics)[['domain','property','method','curve_MAE_pp']].round(3).to_string(index=False))
