#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from v5_repaired_model import RepairedDenseModel

def load_case(root, split, case, k):
    data=root/'prepared/sr386_full'; man=pd.read_csv(data/('surgen_train_10.csv' if split=='train' else 'surgen_validate_10.csv')); r=man[man.case_id==case].iloc[0]
    with np.load(r.features_filepath) as z: f=z['img_features'].astype(np.float32)
    z=np.load(root/'outputs/thesis_algorithm_upgrade_v5/stage1a_infrastructure/cache'/f'regions_k{k}'/split/f'{case}.npz'); q=z['region_patch_indices']; ptr=z['region_ptr']; parts=[f[q[ptr[i]:ptr[i+1]]] for i in range(k)]; p=max(map(len,parts)); x=np.zeros((1,k,p,1024),np.float32); pm=np.zeros((1,k,p),bool)
    for i,a in enumerate(parts): x[0,i,:len(a)]=a; pm[0,i,:len(a)]=True
    return torch.from_numpy(x),torch.from_numpy(pm),torch.from_numpy(z['region_centroid'][None].astype(np.float32)),torch.from_numpy((z['region_size']/len(f))[None].astype(np.float32)),case

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True); dev=torch.device('cuda'); rows=[]; case=pd.read_csv(a.root/'prepared/sr386_full/surgen_validate_10.csv').case_id.iloc[0]
    for name,pool,spatial,rel in [('R0_K64_mean_no_spatial','mean_pool',False,True),('R1_K64_gated_no_spatial','gated_attention_pool',False,True),('R2_K64_gated_spatial','gated_attention_pool',True,True),('R5_K64_local_transformer_spatial','local_transformer_pool',True,True)]:
        x,pm,c,s,case=load_case(a.root,'valid',case,64); m=RepairedDenseModel(pool=pool,spatial=spatial,no_relation=not rel).to(dev).to(torch.bfloat16); x=x.to(dev).bfloat16(); pm=pm.to(dev); c=c.to(dev).bfloat16(); s=s.to(dev).bfloat16(); m.train()
        for ep in range(2):
            m.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16): out,aux=m(x,pm,None,c,s,torch.ones(1,64,dtype=torch.bool,device=dev),True); loss=out.float().pow(2).mean()
            loss.backward(); rows.append({'configuration':name,'epoch':ep+1,'case_id':case,'loss':float(loss),'nan_inf':not bool(torch.isfinite(loss)),'gate_mean':float(np.mean(aux['spatial_gates']))})
    pd.DataFrame(rows).to_csv(a.out/'smoke_results.csv',index=False); print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__': main()
