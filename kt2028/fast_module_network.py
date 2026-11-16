"""W154 single-split screening of remaining W150 ideas; original R3 is intact.

DSCA/PANTHER variants are explicitly inspired adaptations for partial evidence.
OT uses the pinned OTSurv differentiable solver with fixed full transport mass.
"""
import hashlib
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .network import GatedPool
from .survival_network import SurvivalPredictor
from .survival import curve_from_logits


def check_prototype_source(path, metadata, training_rows):
    doc=json.loads(Path(metadata).read_text())
    identity=hashlib.sha256(json.dumps([r['canonical_case_key'] for r in training_rows]).encode()).hexdigest()
    assert doc['fit_partition']=='this_inner_training_only' and doc['patients']==len(training_rows)
    assert doc['training_ids_sha256']==identity,'Morphology prototypes fitted on a different patient pool'
    assert doc['sha256']==hashlib.sha256(Path(path).read_bytes()).hexdigest()
    value=np.load(path,allow_pickle=False)
    assert value.shape==(16,1024) and np.isfinite(value).all()
    return torch.from_numpy(value.astype(np.float32))


class JointRegionalPool(nn.Module):
    """Score paired regions jointly, then normalize each stream on its support."""
    def __init__(self, dim):
        super().__init__()
        self.attention=GatedPool(2*dim)
        self.last_weights=None

    def forward(self, coarse, fine, ids):
        observed=torch.zeros_like(coarse).index_copy(0,ids,fine)
        paired=torch.cat([coarse,observed],-1)
        scores=self.attention.score(self.attention.a(paired).tanh()*self.attention.b(paired).sigmoid())[:,0]
        wc=scores.softmax(0)
        # Softmax over observed support directly avoids underflow in wc[ids]/sum.
        wf=scores[ids].softmax(0)
        self.last_weights=(wc.detach(),wf.detach())
        return wc@coarse,wf@fine


class MorphologyGroupPool(nn.Module):
    def __init__(self, dim, prototypes, dropout=.25):
        super().__init__()
        self.register_buffer('prototypes',prototypes.detach().clone())
        self.mapping=nn.Sequential(nn.Linear(dim,dim//2),nn.GELU(),nn.Dropout(dropout),nn.Linear(dim//2,dim))
        self.last_group_ids=None

    def forward(self, visible_coarse, observed_fine, pool):
        labels=torch.cdist(visible_coarse.float(),self.prototypes).argmin(1)
        counts=torch.bincount(labels,minlength=len(self.prototypes))
        means=observed_fine.new_zeros(len(self.prototypes),observed_fine.shape[-1]).index_add(0,labels,observed_fine)
        occupied=counts>0
        means=means[occupied]/counts[occupied,None]
        self.last_group_ids=labels.detach()
        return pool(self.mapping(means))[0][0]


def solver_class(source):
    """Load only official solver, preserving relative source identity."""
    directory=Path(source)/'src/mil_models/otsurv_component'
    name='_kt2028_w154_ot_source'
    if name in sys.modules:
        assert sys.modules[name].__path__==[str(directory)]
    else:
        package=types.ModuleType(name);package.__path__=[str(directory)];sys.modules[name]=package
    return importlib.import_module(name+'.sinkhornknopp').SemiCurrSinkhornKnopp_stable


class TransportPool(nn.Module):
    def __init__(self, dim, source, slots=16):
        super().__init__()
        self.tokens=nn.Embedding(slots,dim)
        self.slot_readout=nn.Linear(slots,1)
        self.solver=solver_class(source)
        self.last_flow=None

    def transport(self, values, tokens):
        x=values-values.amin(-1,keepdim=True)
        y=tokens-tokens.amin(-1,keepdim=True)
        cost=torch.cdist(x,y)
        cost=cost/cost.max().clamp_min(torch.finfo(cost.dtype).eps)
        solver=self.solver(num_iters=3,epsilon=.1,gamma=1,stoperr=1e-10,numItermax=1000,
                           rho=1.,semi_use=True,prior=None)
        return solver.cost_forward(cost,final=False,count=True)[0]

    def forward(self, x):
        flow=checkpoint(self.transport,x,self.tokens.weight,use_reentrant=False) if self.training and torch.is_grad_enabled() else self.transport(x,self.tokens.weight)
        pooled=self.slot_readout((flow.T@x).T).squeeze(-1)
        mass=flow.sum(1);weights=mass/mass.sum()
        self.last_flow=flow.detach()
        return pooled[None],weights[None]


class FastModulePredictor(SurvivalPredictor):
    def __init__(self, arm, *, prototypes=None, ot_source=None, **architecture):
        super().__init__(**architecture)
        self.arm=arm
        if arm=='joint':self.joint_pool=JointRegionalPool(self.dim)
        elif arm=='morphology':self.group_pool=MorphologyGroupPool(self.dim,prototypes,architecture.get('dropout',.25))
        elif arm=='transport':self.fine_pool=TransportPool(self.dim,ot_source)
        else:raise ValueError('Unregistered module')

    def forward(self, coarse, boxes, observed_fine, relative_xy, candidate_patch_count,
                *, return_aux=False, slide_indices=None):
        out=super().forward(coarse,boxes,observed_fine,relative_xy,candidate_patch_count,
                            return_aux=return_aux,slide_indices=slide_indices)
        if not observed_fine or self.arm=='transport':return out
        ids=torch.as_tensor(out['observed_ids'],device=coarse.device,dtype=torch.long)
        zc=out['coarse_patient']
        if self.arm=='joint':
            fused=out['coarse_tokens'].clone()
            fused[ids]=fused[ids]*out['coarse_gates']
            zc,zf=self.joint_pool(fused,out['observed_tokens'],ids)
        else:
            zf=self.group_pool(coarse[ids],out['observed_tokens'],self.fine_pool)
        out['coarse_patient'],out['fine_patient']=zc,zf
        out['logits']=self.readout(zc,zf,out['observation_state'])
        out['hazards'],out['survival']=curve_from_logits(out['logits'])
        return out
