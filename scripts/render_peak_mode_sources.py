"""Generate editable PGFPlots source and source-data copies for the mode-coupling figure."""
from pathlib import Path
import pandas as pd
import json
root=Path(__file__).resolve().parents[1]
folder=root/'reports/peak_mode_coupling/figure_sources';folder.mkdir(parents=True,exist_ok=True)
p=pd.read_csv(root/'reports/peak_mode_coupling/peaks.csv')
c=pd.read_csv(root/'reports/peak_mode_coupling/curves.csv')
order=['additive','AX','BX','AB','AX+BX','AB+AX','all pairs']
contract={'conclusion':'The omitted part along barely determined directions carries the two large interpolation peaks; an isotropic omitted part keeps the pattern of large and small peaks.', 'panels':{'a':'Formation-energy peak intervention and isotropic expectation','b':'Volume peak intervention and isotropic expectation','c':'Design amplification at n/p=1','d':'Undetermined directions at n/p=1'},'archetype':'quantitative grid','backend':'LaTeX/PGFPlots','statistics':'Median across the same five designs at each of 13 coverages; peak is the largest rise over an earlier minimum; no confidence bands','outputs':['editable TeX','vector PDF','source CSV'],'interpretation':'The removal along weak directions uses the complete-grid omitted part. The isotropic line is an exact Gaussian expectation for an omitted part of matched covariance.'}
(folder/'figure_contract.json').write_text(json.dumps(contract,indent=2))
tex=r'''\documentclass[border=3mm]{standalone}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots}
\pgfplotsset{compat=1.18}
\definecolor{observed}{HTML}{533E47}
\definecolor{weak}{HTML}{5088E9}
\definecolor{isotropic}{HTML}{D3B001}
\begin{document}
\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 2,horizontal sep=1.8cm,vertical sep=1.7cm},
width=8.5cm,height=6.1cm,axis lines=left,grid=major,grid style={gray!12},
xmin=-0.3,xmax=6.3,xtick={0,1,2,3,4,5,6},xticklabels={Add.,AX,BX,AB,AX+BX,AB+AX,All},
xticklabel style={font=\scriptsize,rotate=30,anchor=north east},yticklabel style={font=\small},
label style={font=\small},title style={font=\small},
legend style={draw=none,fill=white,font=\scriptsize},every axis plot/.append style={thick}]
'''
def coord(values):
 return ' '.join(f'({i},{v:.8g})' for i,v in enumerate(values))
for label,prop in [('a','heat_of_formation'),('b','volume_per_fu')]:
 q=p[p.property==prop].set_index('model').loc[order]
 title='Formation energy' if label=='a' else 'Volume'
 tex+=rf'\nextgroupplot[title={{({label}) {title}}},ylabel={{Largest MAE rebound (\%)}},ymin=0,ymax=145,legend pos=north west]'+'\n'
 for col,color,mark,name,style in [('mae_rise_pct','observed','*','Original field',''),('mae_no_weak_rise_pct','weak','square*','Weak excitation removed',''),('isotropic_mae_rise_pct','isotropic','triangle*','Isotropic-field expectation','dashed,')]:
  tex+=rf'\addplot[{style}color={color},mark={mark}] coordinates {{{coord(q[col])}}};'+'\n'
  if label=='a': tex+=rf'\addlegendentry{{{name}}}'+'\n'
q=c[(c.property=='heat_of_formation')&(c.target==1)].set_index('model').loc[order]
tex+=r'\nextgroupplot[title={(c) Geometry amplifies omitted signal},ylabel={Amplification $\Gamma_S$ at $n/p=1$},ymin=0,ymax=12]'+'\n'
tex+=rf'\addplot[color=observed,mark=*] coordinates {{{coord(q.geometry)}}};'+'\n'
tex+=r'\nextgroupplot[title={(d) Zero modes differ from weak modes},ylabel={Unidentified directions (\%)},ymin=0,ymax=40]'+'\n'
tex+=rf'\addplot[ybar,bar width=12pt,fill=weak,draw=weak!70!black] coordinates {{{coord(100*q.nullity/q.p)}}};'+'\n'
tex+=r'''\end{groupplot}
\end{tikzpicture}
\end{document}
'''
(folder/'peak_modes.tex').write_text(tex)
# Data files are included with the figure, retaining every model and mask.
p.to_csv(folder/'peak_summary.csv',index=False)
c.to_csv(folder/'source_curves.csv',index=False)
print(folder/'peak_modes.tex')
