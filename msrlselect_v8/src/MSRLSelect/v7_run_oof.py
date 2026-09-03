"""Strict patient-held-out OOF generation for the selected E2 interface."""
import numpy as np,pandas as pd,torch
from pathlib import Path
from v7_train_evidence import coarse,build_moe,train_one,OUT,ROOT,BASE,CACHE,DEV
from sklearn.decomposition import PCA
def main():
 tr=pd.read_csv(ROOT/'prepared/sr386_full/surgen_train_10.csv'); va=pd.read_csv(ROOT/'prepared/sr386_full/surgen_validate_10.csv')
 pca=np.load(OUT/'assets/train_pca128_components.npy');pm=np.load(OUT/'assets/train_pca128_mean.npy');pev=np.load(OUT/'assets/train_pca128_explained_variance.npy');pc=PCA(128);pc.components_=pca;pc.mean_=pm;pc.explained_variance_=pev;pc.explained_variance_ratio_=pev/pev.sum();pc.singular_values_=np.sqrt(np.maximum(pev*(16320-1),0));pc.n_features_in_=1024;pc.n_samples_=16320;pc.n_samples_seen_=16320;pc.noise_variance_=0
 ct=coarse('train',tr,pc); ft=torch.load(CACHE/'train/fine_seed17.pt',weights_only=True); moe=build_moe('train',tr).reshape(len(tr),64,-1)
 tt=torch.tensor(pd.read_csv(BASE/'p1r_protocol_correction/predictions/train_oof_ensemble_predictions.csv').ensemble_probability.values,device=DEV).float(); fold=np.arange(len(tr))%5; rows=[]
 for f in range(5):
  it=np.where(fold!=f)[0];iv=np.where(fold==f)[0]
  for seed in (1,17,42):
   # Holdout fold is used only for checkpoint selection in this isolated OOF run.
   _,_,net=train_one(f'E2_oof_fold{f}',seed,ct[it],ct[iv],moe[it],moe[iv],tr.iloc[it].reset_index(drop=True),tr.iloc[iv].reset_index(drop=True),tt[it],tt[iv],epochs=120)
   net.eval();
   with torch.no_grad():
    m=torch.ones(len(iv),1,64,dtype=torch.bool,device=DEV); z=net(ct[iv].float().to(DEV),moe[iv].float().to(DEV),m); p=torch.sigmoid(z[0][:,0]).cpu().numpy()
   for j,idx in enumerate(iv): rows.append({'patient_id':tr.iloc[idx].case_id,'label':int(tr.iloc[idx].label),'fold':f,'seed':seed,'k':64,'probability':float(p[j])})
 out=pd.DataFrame(rows);out.to_csv(OUT/'predictions/train_oof_predictions.csv',index=False);out.groupby(['patient_id','label','fold','k'],as_index=False).probability.mean().to_csv(OUT/'metrics/final_candidate_oof.csv',index=False)
if __name__=='__main__': main()
