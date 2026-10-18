"""One patient/target, multiple independently positioned slides, no hidden fine reads."""
import hashlib
import json
from pathlib import Path
import numpy as np

from .uni_encoder import sha256_file
from .prepare_mixed_survival import target


def inspect_asset(folder, slide, protocol):
    folder = Path(folder)
    r = json.loads((folder/'receipt.json').read_text(encoding='utf-8'))
    if r.get('status') != 'completed' or not r.get('original_uni_frozen'):
        raise ValueError('Require completed frozen-UNI extraction')
    if r.get('weight_sha256') != protocol['weight_sha256'] or r.get('feature_dimension') != 1024:
        raise ValueError('Different feature encoder')
    if protocol['input_pixels'] != 224 or protocol['feature_dimension'] != 1024:
        raise ValueError('Unsupported image encoder input dimensions')
    if slide.get('receipt_sha256') and sha256_file(folder/'receipt.json') != slide['receipt_sha256']:
        raise ValueError('Previously audited receipt changed')
    if sha256_file(folder/'geometry.npz') != r['geometry_sha256']:
        raise ValueError('Geometry checksum differs')
    import zarr
    for name, mpp in [('coarse',protocol['coarse_mpp_um']),('fine',protocol['fine_mpp_um'])]:
        group = zarr.open_group(str(folder/(name+'.zarr')),mode='r')
        if group.attrs['mpp_um'] != mpp or group.attrs['status'] != 'completed':
            raise ValueError('Missing extraction or wrong physical scale; old 4/1 MPP is forbidden')
        boxes=np.asarray(group['bbox_um'][:])
        # Boundary boxes may be shorter, but never exceed the registered field.
        if boxes.shape!=(r[name]['rows'],4) or np.any(boxes[:,2:]-boxes[:,:2]>224*mpp+1e-6):
            raise ValueError('Physical patch footprint differs from 224-pixel protocol')
        if (group['features'].shape != (r[name]['rows'],1024) or str(group['features'].dtype) != 'float32'
            or group.attrs['completed_rows'] != r[name]['rows']):
            raise ValueError('Uncommitted feature store')
        if slide.get('feature_hashes') and r[name]['features_sha256'] != slide['feature_hashes'][name]:
            raise ValueError('Audited feature fingerprint changed')
    return r


class PatientCase:
    def __init__(self, row, roots, protocol):
        import zarr
        self.parts=[]
        coarse, boxes, counts, slide_indices = [],[],[],[]
        offset=0
        for sid,s in enumerate(row['slides']):
            folder=Path(roots[row['cohort']])/s['asset_id']
            receipt=inspect_asset(folder,s,protocol)
            with np.load(folder/'geometry.npz',allow_pickle=False) as g:
                b=g['coarse_bbox_um'].astype(np.float32); n=g['fine_counts'].copy(); ptr=g['fine_ptr'].copy()
                rel=g['fine_relative_xy'].astype(np.float32); parents=g['fine_parent_indices'].copy()
            cg=zarr.open_group(str(folder/'coarse.zarr'),mode='r')
            c=np.asarray(cg['features'][:],dtype=np.float32)
            if c.shape!=(len(n),1024) or not np.isfinite(c).all() or np.any(n>16):
                raise ValueError('Invalid nested coarse/fine asset')
            if not np.array_equal(np.diff(ptr),n) or ptr[-1]!=len(rel) or len(parents)!=len(rel):
                raise ValueError('Fine CSR differs')
            if not np.array_equal(parents,np.repeat(np.arange(len(n)),n)):
                raise ValueError('Fine parents are not in registered nested order')
            if b.shape!=(len(n),4) or not np.isfinite(b).all() or np.any((rel<0)|(rel>1)):
                raise ValueError('Invalid physical/relative geometry')
            coarse.append(c); boxes.append(b); counts.append(n)
            slide_indices.extend([sid]*len(n))
            self.parts.append(dict(folder=folder,offset=offset,counts=n,ptr=ptr,relative=rel,receipt=receipt))
            offset+=len(n)
        self.coarse=np.concatenate(coarse); self.boxes=np.concatenate(boxes)
        self.counts=np.concatenate(counts); self.slide_indices=np.asarray(slide_indices,dtype=np.int64)

    def inputs(self, selected):
        import torch
        import zarr
        selected=[int(i) for i in selected]
        if len(set(selected))!=len(selected) or any(i<0 or i>=len(self.coarse) or self.counts[i]<=0 for i in selected):
            raise ValueError('Invalid observed region IDs')
        fine,relative={},{}
        for p in self.parts:
            ids=[i for i in selected if p['offset']<=i<p['offset']+len(p['counts'])]
            if not ids:
                continue  # ZERO mode never reads any fine feature values.
            spans=[tuple(p['ptr'][i-p['offset']:i-p['offset']+2]) for i in ids]
            indices=np.concatenate([np.arange(a,b) for a,b in spans])
            group=zarr.open_group(str(p['folder']/'fine.zarr'),mode='r')
            values=np.asarray(group['features'].oindex[indices,:],dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite observed fine')
            off=0
            for i,(a,b) in zip(ids,spans):
                fine[i]=torch.from_numpy(values[off:off+b-a].copy())
                relative[i]=torch.from_numpy(p['relative'][a:b].copy()); off+=b-a
        return dict(coarse=torch.from_numpy(self.coarse.copy()), boxes=torch.from_numpy(self.boxes.copy()),
            observed_fine=fine,relative_xy=relative,candidate_patch_count=int(self.counts.sum()),
            slide_indices=torch.from_numpy(self.slide_indices.copy()))


class MixedSurvivalDataset:
    def __init__(self, manifest, roots, *, contract, purpose='train', seed=41):
        doc=json.loads(Path(manifest).read_text(encoding='utf-8'))
        expected={'train':'train','model_selection':'validate'}.get(purpose)
        if expected is None or doc['schema']!='KT2028-mixed-survival-manifest-1' or doc['partition']!=expected:
            raise ValueError('Only explicit mixed Train/Validation entry; no Test')
        if doc['data_contract']!=contract:
            raise ValueError('Clinical/feature protocol mismatch')
        self.rows,self.roots,self.contract,self.seed,self._cache=doc['patients'],roots,contract,seed,{}
        keys=[p['canonical_case_key'] for p in self.rows]
        if not keys or len(set(keys))!=len(keys):
            raise ValueError('Duplicate/empty patient cohort')
        for r in self.rows:
            t=r['survival_target']
            if t!=target(t['time_days'],t['event'],contract['interval_edges_days']) or not r['slides']:
                raise ValueError('Survival target or patient slides differ')

    def __len__(self): return len(self.rows)

    def case(self,i):
        if i not in self._cache:
            if len(self._cache)>=8: self._cache.pop(next(iter(self._cache)))
            self._cache[i]=PatientCase(self.rows[i],self.roots,self.contract['feature_protocol'])
        return self._cache[i]

    def sample(self,i,mode,epoch=0):
        case=self.case(i); eligible=np.flatnonzero(case.counts>0)
        key=f"{self.seed}:{epoch}:{self.rows[i]['canonical_case_key']}:{mode}".encode()
        rng=np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8],'little'))
        if mode=='zero': k=0
        elif mode=='full': k=len(eligible)
        elif mode=='random':
            u=rng.random()
            k=0 if u<.25 else len(eligible) if u<.5 else int(rng.integers(1,len(eligible))) if len(eligible)>1 else len(eligible)
        else:
            fraction=float(mode)
            if not 0<fraction<1: raise ValueError('Invalid observation fraction')
            k=min(len(eligible),max(1,int(np.ceil(fraction*len(eligible)))))
        return case.inputs(rng.permutation(eligible)[:k]),dict(self.rows[i]['survival_target'])
