#!/usr/bin/env python3
from pathlib import Path
import argparse, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

SETTINGS = {'Global-only':'global_only','Local-only revision':'local_revision','Global + Random':'fusion_random','Global + HeMiCoRe':'fusion_hemi','Global + MS-RLSurv revision':'fusion_revision'}
SEEDS=(1,17,42)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--bootstrap',type=int,default=2000); a=ap.parse_args(); root=a.root; rows=[]; preds={}; gates=[]; shuffles=[]
 for setting,dname in SETTINGS.items():
  for seed in SEEDS:
   d=root/f'chapter4/seed{seed}/{dname}'; m=json.loads((d/'metrics_final.json').read_text())
   test=m['test']; rows.append({'setting':setting,'seed':seed,**{k:test[k] for k in ('auc','auprc','accuracy','f1','sensitivity','specificity')}})
   x=pd.read_csv(d/'test_predictions.csv'); preds[(setting,seed)]=x[['case_id','label','probability']]
   if (d/'gate_weights_test.csv').exists():
    g=pd.read_csv(d/'gate_weights_test.csv'); gates.append({'setting':setting,'seed':seed,'mean_gate':g.gate_weight.mean(),'std_gate':g.gate_weight.std(),'min_gate':g.gate_weight.min(),'max_gate':g.gate_weight.max()})
   if (d/'local_shuffle_audit.json').exists():
    z=json.loads((d/'local_shuffle_audit.json').read_text()); shuffles.append({'setting':setting,'seed':seed,'original_auroc':z['original']['auc'],'shuffled_auroc':z['shuffled']['auc'],'delta_auroc':z['shuffled']['auc']-z['original']['auc'],'original_auprc':z['original']['auprc'],'shuffled_auprc':z['shuffled']['auprc'],'delta_auprc':z['shuffled']['auprc']-z['original']['auprc']})
 pd.DataFrame(rows).to_csv(root/'chapter4_multiseed_results.csv',index=False); pd.DataFrame(gates).to_csv(root/'gate_statistics_multiseed.csv',index=False); pd.DataFrame(shuffles).to_csv(root/'local_shuffle_multiseed.csv',index=False)
 ens=[]; ens_frames={}
 for setting in SETTINGS:
  base=preds[(setting,1)][['case_id','label']].copy(); ps=[preds[(setting,s)].set_index('case_id').loc[base.case_id,'probability'].to_numpy() for s in SEEDS]; base['probability']=np.mean(ps,axis=0); base['setting']=setting; ens_frames[setting]=base; ens.append({'setting':setting,'seed':'ensemble','auroc':roc_auc_score(base.label,base.probability),'auprc':average_precision_score(base.label,base.probability)})
 pd.concat(ens_frames.values(),ignore_index=True).to_csv(root/'chapter4_ensemble_predictions.csv',index=False); pd.DataFrame(ens).to_csv(root/'chapter4_ensemble_results.csv',index=False)
 rng=np.random.default_rng(5252); y=ens_frames['Global-only'].label.to_numpy(); strata=[np.flatnonzero(y==c) for c in (0,1)]; methods=list(SETTINGS); boot={m:{'auc':[],'pr':[]} for m in methods}; dif={}
 for _ in range(a.bootstrap):
  idx=np.concatenate([rng.choice(ix,len(ix),replace=True) for ix in strata]);
  for m in methods:
   yy=ens_frames[m].label.to_numpy()[idx]; pp=ens_frames[m].probability.to_numpy()[idx]; boot[m]['auc'].append(roc_auc_score(yy,pp)); boot[m]['pr'].append(average_precision_score(yy,pp))
  for m in methods[1:]:
   dif.setdefault((m,'Global-only'),{'auc':[],'pr':[]})
   for q in ('auc','pr'): dif[(m,'Global-only')][q].append(boot[m][q][-1]-boot['Global-only'][q][-1])
 ci=[]
 for m in methods:
  for q in ('auc','pr'):
   v=np.asarray(boot[m][q]); ci.append({'analysis':'chapter4','comparison':m,'metric':'auroc' if q=='auc' else 'auprc','estimate':float(v.mean()),'ci_low':float(np.quantile(v,.025)),'ci_high':float(np.quantile(v,.975))})
 for (m,b),dd in dif.items():
  for q,vv in dd.items():
   v=np.asarray(vv); ci.append({'analysis':'chapter4','comparison':m+' - '+b,'metric':'auroc' if q=='auc' else 'auprc','estimate':float(v.mean()),'ci_low':float(np.quantile(v,.025)),'ci_high':float(np.quantile(v,.975))})
 pd.DataFrame(ci).to_csv(root/'chapter4_bootstrap_confidence_intervals.csv',index=False)
 print(pd.DataFrame(rows).to_string(index=False)); print(pd.DataFrame(ens).to_string(index=False))
if __name__=='__main__': main()
