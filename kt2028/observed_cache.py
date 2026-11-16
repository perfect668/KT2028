"""CPU cache of actually requested regions; zero mode never reads fine values."""
from collections import OrderedDict
import numpy as np
import torch

from .mixed_survival_data import PatientCase
from .run_multicancer import Dataset


class CachedCase:
    def __init__(self, original, cache, case_key):
        self.original,self.cache,self.case_key=original,cache,case_key

    def __getattr__(self,name):return getattr(self.original,name)

    def inputs(self, selected):
        selected=[int(i) for i in selected]
        if len(selected)!=len(set(selected)) or any(i<0 or i>=len(self.counts) or self.counts[i]<=0 for i in selected):
            raise ValueError('Invalid requested observed region')
        missing=[i for i in selected if (self.case_key,i) not in self.cache.values]
        if missing:
            fresh=self.original.inputs(missing)
            for i in missing:
                self.cache.put((self.case_key,i),fresh['observed_fine'][i],fresh['relative_xy'][i])
            self.cache.regions_read+=len(missing)
        fine={};relative={}
        for i in selected:
            f,r=self.cache.values.pop((self.case_key,i));self.cache.values[self.case_key,i]=(f,r)
            fine[i],relative[i]=f,r
        result=dict(coarse=torch.from_numpy(self.coarse.copy()),boxes=torch.from_numpy(self.boxes.copy()),
            observed_fine=fine,relative_xy=relative,candidate_patch_count=int(self.counts.sum()),
            slide_indices=torch.from_numpy(self.slide_indices.copy()))
        self.cache.trim()
        return result


class RegionCache:
    def __init__(self,gib):
        self.values=OrderedDict();self.bytes=0;self.limit=int(gib*1024**3);self.regions_read=0

    def put(self,key,fine,relative):
        assert key not in self.values
        self.values[key]=(fine,relative)
        self.bytes+=fine.numel()*fine.element_size()+relative.numel()*relative.element_size()

    def trim(self):
        while self.bytes>self.limit and self.values:
            _,(f,r)=self.values.popitem(last=False)
            self.bytes-=f.numel()*f.element_size()+r.numel()*r.element_size()


class CachedDataset(Dataset):
    def __init__(self,rows,root,cfg,gib):
        super().__init__(rows,root,cfg,'independent')
        self.region_cache=RegionCache(gib)

    def case(self,i):
        if i not in self._cache:
            raw=PatientCase(self.rows[i],self.roots,self.contract['feature_protocol'])
            self._cache[i]=CachedCase(raw,self.region_cache,self.rows[i]['canonical_case_key'])
        return self._cache[i]


class DatasetSubset:
    def __init__(self,data,indices):
        self.data,self.indices=data,list(indices);self.rows=[data.rows[i] for i in self.indices]

    def __len__(self):return len(self.indices)

    def sample(self,i,mode,epoch=0):return self.data.sample(self.indices[i],mode,epoch)

    def case(self,i):return self.data.case(self.indices[i])
