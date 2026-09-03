#!/usr/bin/env python3
from pathlib import Path
import argparse, numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, average_precision_score

def midrank(x):
    order=np.argsort(x); z=x[order]; out=np.empty(len(x),float); i=0
    while i<len(x):
        j=i
        while j<len(x) and z[j]==z[i]: j+=1
        out[order[i:j]]=(i+j-1)/2+1; i=j
    return out
def delong(y, a, b):
    y=np.asarray(y); order=np.argsort(-y); a=np.asarray(a)[order]; b=np.asarray(b)[order]; m=int(y.sum()); n=len(y)-m; k=2
    preds=np.vstack([a,b]); tx=np.zeros((k,m)); ty=np.zeros((k,n)); tz=np.zeros((k,m+n))
    for r in range(k):
        tx[r]=midrank(preds[r,:m]); ty[r]=midrank(preds[r,m:]); tz[r]=midrank(preds[r])
    auc=tz[:,:m].sum(1)/(m*n)-(m+1)/(2*n)
    v01=(tz[:,:m]-tx)/n; v10=1-(tz[:,m:]-ty)/m
    cov=np.cov(v01)+np.cov(v10)/1
    cov=cov/m+cov*0 # placeholder overwritten below
    sx=np.cov(v01); sy=np.cov(v10); cov=sx/m+sy/n
    d=auc[0]-auc[1]; se=np.sqrt(max(cov[0,0]+cov[1,1]-2*cov[0,1],1e-15)); return float(d),float(2*norm.sf(abs(d/se)))
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--bootstrap',type=int,default=2000); a=ap.parse_args(); r=a.root
 c3=pd.read_csv(r/'chapter3_ensemble_predictions.csv'); c4=pd.read_csv(r/'chapter4_ensemble_predictions.csv'); rng=np.random.default_rng(8888); records=[]; tests=[]
 analyses=[('chapter3',c3,'method',{'Random':'Random','HeMiCoRe':'HeMiCoRe','MS-RLSurv_revision':'MS-RLSurv_revision'}, [('MS-RLSurv_revision','Random'),('MS-RLSurv_revision','HeMiCoRe')]),('chapter4',c4,'setting',{'Global-only':'Global-only','Global + Random':'Global + Random','Global + HeMiCoRe':'Global + HeMiCoRe','Global + MS-RLSurv revision':'Global + MS-RLSurv revision'}, [('Global + MS-RLSurv revision','Global-only'),('Global + MS-RLSurv revision','Global + Random'),('Global + MS-RLSurv revision','Global + HeMiCoRe')])]
 for an,df,col,names,comparisons in analyses:
  frames={k:df[df[col]==k].sort_values('case_id') for k in names}; y=next(iter(frames.values())).label.to_numpy(); strata=[np.flatnonzero(y==q) for q in (0,1)]; boot={k:{'auroc':[],'auprc':[]} for k in names}; dif={}
  for _ in range(a.bootstrap):
   ix=np.concatenate([rng.choice(s,len(s),replace=True) for s in strata])
   for k,f in frames.items(): boot[k]['auroc'].append(roc_auc_score(y[ix],f.probability.to_numpy()[ix])); boot[k]['auprc'].append(average_precision_score(y[ix],f.probability.to_numpy()[ix]))
   for x,z in comparisons:
    dif.setdefault((x,z),{'auroc':[],'auprc':[]})
    for q in ('auroc','auprc'): dif[(x,z)][q].append(boot[x][q][-1]-boot[z][q][-1])
  for k in names:
   for q in ('auroc','auprc'):
    v=np.asarray(boot[k][q]); records.append({'analysis':an,'comparison':k,'metric':q,'estimate':float(v.mean()),'ci_low':float(np.quantile(v,.025)),'ci_high':float(np.quantile(v,.975))})
  for x,z in comparisons:
   d,p=delong(y,frames[x].probability,frames[z].probability)
   for q in ('auroc','auprc'):
    v=np.asarray(dif[(x,z)][q]); pboot=min(1.,2*min(np.mean(v<=0),np.mean(v>=0))); tests.append({'analysis':an,'comparison':x+' vs '+z,'metric':q,'estimate':float(v.mean()),'p_raw':float(p if q=='auroc' else pboot)})
   v=np.asarray(dif[(x,z)]['auroc']); records.append({'analysis':an,'comparison':x+' - '+z,'metric':'auroc','estimate':float(v.mean()),'ci_low':float(np.quantile(v,.025)),'ci_high':float(np.quantile(v,.975))})
   v=np.asarray(dif[(x,z)]['auprc']); records.append({'analysis':an,'comparison':x+' - '+z,'metric':'auprc','estimate':float(v.mean()),'ci_low':float(np.quantile(v,.025)),'ci_high':float(np.quantile(v,.975))})
 pd.DataFrame(records).to_csv(r/'bootstrap_confidence_intervals.csv',index=False); t=pd.DataFrame(tests); order=np.argsort(t.p_raw.to_numpy()); adj=np.empty(len(t));
 for rank,i in enumerate(order): adj[i]=min(1.,(len(t)-rank)*t.iloc[i].p_raw)
 t['p_holm']=adj; t['test_method']=np.where(t.metric=='auroc','paired DeLong','paired stratified bootstrap'); t.to_csv(r/'paired_significance_tests.csv',index=False); print(t.to_string(index=False))
if __name__=='__main__': main()
