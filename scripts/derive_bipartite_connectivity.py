"""Exact finite bipartite connectivity and rank recurrence, no signal labels."""
import math,json
from pathlib import Path
import numpy as np,pandas as pd
from numba import njit
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/random_sampling_window'
@njit
def connected_table(m,f,comb):
 c=np.zeros((m+1,m+1));c[1,0]=c[0,1]=1.
 for a in range(1,m+1):
  for b in range(1,m+1):
   v=1.
   for r in range(1,a+1):
    for s in range(b+1):
     if r==a and s==b:continue
     v-=comb[a-1,r-1]*comb[b,s]*c[r,s]*(1-f)**(r*(b-s)+s*(a-r))
   c[a,b]=max(0.,min(1.,v))
 return c

def main():
 m=19;comb=np.zeros((m+1,m+1))
 for a in range(m+1):
  for b in range(a+1):comb[a,b]=math.comb(a,b)
 rows=[]
 for f in [.03,.05,.08,.089,.10,.12,.162,.196,.20,.35]:
  c=connected_table(m,f,comb);ec=0.
  for a in range(m+1):
   for b in range(m+1):
    if a+b:ec+=comb[m,a]*comb[m,b]*c[a,b]*(1-f)**(a*(m-b)+b*(m-a))
  rows.append(dict(coverage=f,connected_probability=c[m,m],expected_rank=2*m-ec,expected_corank=ec-1,expected_isolated_vertices=2*m*(1-f)**m))
 conn50=brentq(lambda f:connected_table(m,f,comb)[m,m]-.5,.15,.25)
 approx=1-(math.log(2)/(2*m))**(1/m)
 pd.DataFrame(rows).to_csv(OUT/'connectivity.csv',index=False)
 result={'unconditional_connectivity_50pct':conn50,'isolated_vertex_poisson_approximation':approx,'role':'Design-only geometric scale; not a universal benefit boundary or the endpoint-conditioned target.'}
 (OUT/'connectivity_summary.json').write_text(json.dumps(result,indent=2));print(pd.DataFrame(rows).round(5).to_string(index=False));print(result)
if __name__=='__main__':main()
