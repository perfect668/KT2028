from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class V5RegionDataset(Dataset):
    def __init__(self, manifest, cache_root, num_regions):
        self.frame=pd.read_csv(manifest); self.root=Path(cache_root); self.k=int(num_regions)
    def __len__(self): return len(self.frame)
    def __getitem__(self,i):
        r=self.frame.iloc[i]; split='valid' if 'validate' in Path(r.features_filepath).name else ('train' if 'train' in Path(r.features_filepath).name else 'test')
        p=self.root/f'regions_k{self.k}'/split/f'{r.case_id}.npz'
        with np.load(p,allow_pickle=False) as z:
            return {'case_id':str(r.case_id),'patient_id':str(r.patient_id),'label':int(r.label),'region_feature_mean':z['region_feature_mean'].astype(np.float32),'region_feature_variance':z['region_feature_variance'].astype(np.float32),'region_centroid':z['region_centroid'].astype(np.float32),'region_size':z['region_size'].astype(np.int64),'region_morph_cluster':z['region_morph_cluster'].astype(np.int64),'region_patch_indices':z['region_patch_indices'].astype(np.int64),'region_ptr':z['region_ptr'].astype(np.int64),'raw_patch_count':int(z['raw_patch_count'][0])}

def collate_v5(items):
    b=len(items); rmax=max(x['region_feature_mean'].shape[0] for x in items); d=items[0]['region_feature_mean'].shape[1]; pmax=max(len(x['region_patch_indices']) for x in items)
    out={'region_features':torch.zeros(b,rmax,d),'region_variance':torch.zeros(b,rmax,d),'region_centroid':torch.zeros(b,rmax,2),'region_size':torch.zeros(b,rmax,dtype=torch.long),'region_morph_cluster':torch.full((b,rmax),-1,dtype=torch.long),'region_mask':torch.zeros(b,rmax,dtype=torch.bool),'region_patch_indices':torch.full((b,pmax),-1,dtype=torch.long),'region_patch_mask':torch.zeros(b,pmax,dtype=torch.bool),'region_ptr':torch.full((b,rmax+1),-1,dtype=torch.long),'labels':torch.tensor([x['label'] for x in items]),'case_id':[x['case_id'] for x in items],'patient_id':[x['patient_id'] for x in items],'raw_patch_count':torch.tensor([x['raw_patch_count'] for x in items])}
    for i,x in enumerate(items):
        rr=len(x['region_feature_mean']); pp=len(x['region_patch_indices']); out['region_features'][i,:rr]=torch.from_numpy(x['region_feature_mean']); out['region_variance'][i,:rr]=torch.from_numpy(x['region_feature_variance']); out['region_centroid'][i,:rr]=torch.from_numpy(x['region_centroid']); out['region_size'][i,:rr]=torch.from_numpy(x['region_size']); out['region_morph_cluster'][i,:rr]=torch.from_numpy(x['region_morph_cluster']); out['region_mask'][i,:rr]=True; out['region_patch_indices'][i,:pp]=torch.from_numpy(x['region_patch_indices']); out['region_patch_mask'][i,:pp]=True; out['region_ptr'][i,:rr+1]=torch.from_numpy(x['region_ptr'])
    return out
