"""Partial-observation case interface. Labels never enter this acquisition object."""
import json
from pathlib import Path

import numpy as np
import zarr


class RegionCase:
    def __init__(self, folder):
        folder=Path(folder);self.receipt=json.loads((folder/'receipt.json').read_text())
        if self.receipt['status'] != 'completed':raise ValueError('Incomplete regional asset')
        with np.load(folder/'regions.npz',allow_pickle=False) as src:
            self.coarse=src['coarse'].copy();self.boxes=src['bbox_um'].copy()
            self.counts=src['fine_counts'].copy();self.ptr=src['fine_ptr'].copy()
            self.indices=src['fine_indices'].copy();self.valid=src['fine_image_valid_fraction'].copy()
        r=len(self.coarse)
        if self.coarse.shape!=(r,512) or len(self.ptr)!=r+1 or self.ptr[-1]!=len(self.indices):
            raise ValueError('Invalid regional contract')
        if len(np.unique(self.indices))!=len(self.indices):raise ValueError('Repeated fine ownership')
        self.observed=np.zeros(r,dtype=bool);self._fine={};self.patch_cost=0

    def visible_state(self):
        return {'coarse':self.coarse.copy(),'bbox_um':self.boxes.copy(),
                'observed':self.observed.copy(),'observed_fine':{i:v.copy() for i,v in self._fine.items()},
                'acquirable':((self.counts>0)&~self.observed),
                'cached_patch_counts':self.counts.copy(),'spent_patch_count':self.patch_cost,
                'cost_scope':'cached candidate access; full preprocessing cost accounted separately'}

    def acquire(self, region_id):
        if not isinstance(region_id,(int,np.integer)) or not 0<=region_id<len(self.coarse):
            raise ValueError('Invalid region index')
        if self.observed[region_id] or self.counts[region_id]<=0:raise ValueError('Illegal acquisition')
        indices=self.indices[self.ptr[region_id]:self.ptr[region_id+1]]
        group=zarr.open_group(self.receipt['fine_path'],mode='r')
        value=np.asarray(group['features'].oindex[indices,:])
        if value.shape!=(len(indices),1024) or not np.isfinite(value).all():raise ValueError('Invalid acquired fine')
        self._fine[int(region_id)]=value;self.observed[region_id]=True;self.patch_cost+=len(indices)
        return {'fine':value.copy(),'indices':indices.copy(),'new_patch_cost':len(indices),
                'image_valid_fraction':self.valid[indices].copy()}
