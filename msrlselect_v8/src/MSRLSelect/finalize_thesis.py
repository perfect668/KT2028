#!/usr/bin/env python3
import argparse,json,shutil
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_curve

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args();r=a.root
 rows4=[]
 for d in sorted((r/'chapter4').iterdir()):
  if not d.is_dir() or not (d/'metrics_final.json').exists(): continue
  m=json.loads((d/'metrics_final.json').read_text()); x=m['test']; y=m['valid']; rows4.append({'model':d.name,'split':'test','auc':x['youden']['auc'],'auprc':x['youden']['auprc'],'accuracy_fixed_0_5':x['fixed_0_5']['accuracy'],'f1_fixed_0_5':x['fixed_0_5']['f1'],'sensitivity_fixed_0_5':x['fixed_0_5']['sensitivity'],'specificity_fixed_0_5':x['fixed_0_5']['specificity'],'accuracy_youden':x['youden']['accuracy'],'f1_youden':x['youden']['f1'],'sensitivity_youden':x['youden']['sensitivity'],'specificity_youden':x['youden']['specificity'],'validation_auc':y['youden']['auc'],'threshold_source':'validation_youden'})
 # Existing fusion result is converted to the common death-risk schema using its patient predictions.
 old=a.base/'msrlselect_basic'/'sg_survnet_v2'/'fusion'; pred=old/'test_predictions.csv'; val=old/'valid_predictions.csv'
 if pred.exists() and val.exists():
  td=pd.read_csv(pred); vd=pd.read_csv(val); pcol='probability' if 'probability' in td else 'death_risk'; th=float(roc_curve(vd.label,vd[pcol])[2][np.argmax(roc_curve(vd.label,vd[pcol])[1]-roc_curve(vd.label,vd[pcol])[0])]);
  from sklearn.metrics import roc_auc_score,average_precision_score,accuracy_score,f1_score,confusion_matrix
  def one(df,t):
   y=df.label.to_numpy();p=df[pcol].to_numpy();pr=(p>=t).astype(int);tn,fp,fn,tp=confusion_matrix(y,pr,labels=[0,1]).ravel();return {'auc':roc_auc_score(y,p),'auprc':average_precision_score(y,p),'accuracy_fixed_0_5':accuracy_score(y,p>=.5),'f1_fixed_0_5':f1_score(y,p>=.5,zero_division=0),'sensitivity_fixed_0_5':tp/(tp+fn) if tp+fn else 0,'specificity_fixed_0_5':tn/(tn+fp) if tn+fp else 0,'accuracy_youden':accuracy_score(y,pr),'f1_youden':f1_score(y,pr,zero_division=0),'sensitivity_youden':tp/(tp+fn) if tp+fn else 0,'specificity_youden':tn/(tn+fp) if tn+fp else 0}
  z=one(td,th); rows4.append({'model':'fusion_existing','split':'test',**z,'validation_auc':roc_auc_score(vd.label,vd[pcol]),'threshold_source':'validation_youden','youden_threshold':th})
 pd.DataFrame(rows4).to_csv(r/'chapter4_results.csv',index=False)
 s=pd.read_csv(r/'chapter3/chapter3_selection_statistics.csv'); agg=s.groupby('method').agg(selected_regions_mean=('selected_regions','mean'),selected_patches_mean=('selected_patches','mean'),stop_step_mean=('stop_step','mean'),repeat_rate_mean=('repeat_rate','mean')).reset_index(); agg.to_csv(r/'chapter3_selection_statistics.csv',index=False)
 # selection methods are evaluated as deterministic transfer candidates; complete the table with auditable status.
 rows3=[]
 for method in ['random','HeMiCoRe','MS-RLSelect_fixed_budget','MS-RLSelect_adaptive_budget','MS-RLSelect_no_spatial','MS-RLSelect_no_evidence_reward']:
  rows3.append({'method':method,'validation_auc':np.nan,'validation_auprc':np.nan,'test_auc':np.nan,'test_auprc':np.nan,'evidence_status':'selection statistics/cache generated' if method not in ('HeMiCoRe','MS-RLSelect_no_spatial','MS-RLSelect_no_evidence_reward') else 'not independently retrained in this pass'})
 pd.DataFrame(rows3).to_csv(r/'chapter3_results.csv',index=False)
 # compact representative scatter plots
 try:
  import matplotlib.pyplot as plt
  test=sorted((r/'chapter3/selection_cache/test').glob('*_adaptive.npz'))[0]; case=test.name.replace('_adaptive.npz',''); z=np.load(test); xy=z['selected_coords']; plt.figure(figsize=(6,5));plt.scatter(xy[:,0],xy[:,1],s=2,c=np.arange(len(xy)),cmap='viridis');plt.title(f'{case} adaptive selected patches');plt.xlabel('x');plt.ylabel('y');plt.colorbar(label='patch order');plt.tight_layout();plt.savefig(r/'chapter3'/f'{case}_adaptive_coordinates.png',dpi=180);plt.close()
 except Exception as e: (r/'chapter3'/'visualization_error.txt').write_text(str(e))
 print(json.dumps({'chapter3_results':str(r/'chapter3_results.csv'),'chapter4_results':str(r/'chapter4_results.csv'),'selection_stats':str(r/'chapter3_selection_statistics.csv')},indent=2))
if __name__=='__main__': main()
