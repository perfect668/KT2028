"""Load pristine DSCA a749074 source and adapt QC-missing children, not its weights.

Official files stay byte-identical in a private experiment source snapshot.
Only package loading, input/mask hooks and the existing five-year target interface
are adapted. All actual embedding, attention, Transformer and pooling modules
are instantiated by official WSIHierNet.
"""
import importlib
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn


def load_official(source):
    source=Path(source).resolve()
    # Avoid model/__init__ importing the unused legacy training CLI and tensorboard.
    name='_kt2028_official_dsca'
    if name in sys.modules:
        if Path(sys.modules[name].__path__[0])!=source/'model':
            raise RuntimeError('A different DSCA snapshot was already loaded')
    else:
        package=types.ModuleType(name);package.__path__=[str(source/'model')]
        sys.modules[name]=package
    previous=sys.modules.get('utils')
    # Its only external model import is the official coordinate utility.
    spec=importlib.util.spec_from_file_location('_dsca_official_utils',source/'utils'/'func.py')
    utils=importlib.util.module_from_spec(spec);spec.loader.exec_module(utils)
    sys.modules['utils']=utils
    try:
        hier=importlib.import_module(name+'.HierNet')
        helpers=importlib.import_module(name+'.model_utils')
    finally:
        if previous is None:sys.modules.pop('utils',None)
        else:sys.modules['utils']=previous
    spec=importlib.util.spec_from_file_location('_dsca_official_loss',source/'loss'/'loss_interface.py')
    loss=importlib.util.module_from_spec(spec);spec.loader.exec_module(loss)
    spec=importlib.util.spec_from_file_location('_dsca_official_lookahead',source/'optim'/'lookahead.py')
    optim=importlib.util.module_from_spec(spec);spec.loader.exec_module(optim)
    return SimpleNamespace(network=hier.WSIHierNet,helpers=helpers,utils=utils,SurvMLE=loss.SurvMLE,Lookahead=optim.Lookahead)


def source_network(source,dropout=.6,dim=384):
    official=load_official(source)
    fine=SimpleNamespace(backbone='capool',in_dim=1024,out_dim=dim,scale=4,dropout=dropout,dw_conv=False,ksize=3)
    coarse=SimpleNamespace(backbone='conv1d',in_dim=1024,out_dim=dim,scale=1,dropout=dropout,dw_conv=False,ksize=5)
    encoder=SimpleNamespace(backbone='Transformer',d_model=dim,d_out=dim,nhead=8,dropout=dropout,num_layers=1,ksize=3,dw_conv=False,epsilon=.8)
    return official.network([1024,dim,dim,5],fine,coarse,encoder,dropout=dropout,pool='gap',join='post',fusion='fusion')


def pack_inputs(inputs,official):
    """Map nominal 448um parents/112um children to fixed row-major 4x4 grids."""
    coarse=inputs['coarse'];n=len(coarse)
    fine=coarse.new_zeros((n,16,1024));valid=torch.zeros((n,16),device=coarse.device,dtype=torch.bool)
    for parent,values in inputs['observed_fine'].items():
        rel=inputs['relative_xy'][parent]
        slots=torch.floor(rel*4).to(torch.int64)
        if torch.any((slots<0)|(slots>3)):raise ValueError('Fine child outside nominal 4x4 grid')
        slot=slots[:,1]*4+slots[:,0]
        if len(torch.unique(slot))!=len(slot):raise ValueError('Duplicate fine grid slot')
        fine[parent,slot]=values;valid[parent,slot]=True
    if int(valid.sum())!=inputs['candidate_patch_count']:
        raise ValueError('DSCA full-input adapter must read every eligible child')
    slides=inputs.get('slide_indices',torch.zeros(n,dtype=torch.int64,device=coarse.device))
    coords=[]
    for sid in torch.unique_consecutive(slides):
        xy=inputs['boxes'][slides==sid,:2]
        coords.append(torch.round(xy/448).to(torch.int64))
    coords=torch.cat(official.utils.rearrange_coord(coords,discretization=True)).unsqueeze(0)
    return dict(coarse=coarse.unsqueeze(0),fine=fine.reshape(1,n*16,1024),valid=valid,coords=coords)


class SourceDSCA(nn.Module):
    def __init__(self,source,dropout=.6,dim=384):
        super().__init__();self.official=load_official(source)
        self.network=source_network(source,dropout,dim)
        self._valid=None;self._logits=None
        self.network.patchx20_embedding_layer.cross_att_pool.register_forward_pre_hook(self._mask_children,with_kwargs=True)
        self.network.patchx20_embedding_layer.register_forward_hook(self._clear_empty_parents)
        self.network.patch_encoder_layer_parallel.register_forward_pre_hook(self._mask_parents,with_kwargs=True)
        self.network.patch_encoder_layer_parallel.register_forward_hook(self._clear_empty_tokens)
        self.network.out_layer[0].register_forward_hook(self._capture_logits)

    def _mask_children(self,module,args,kwargs):
        if self._valid is not None and not bool(self._valid.all()):
            valid=self._valid.clone();valid[~valid.any(1),0]=True
            kwargs['key_padding_mask']=~valid
        return args,kwargs

    def _clear_empty_parents(self,module,args,out):
        if self._valid is None:return out
        values,attn,query=out
        return values.masked_fill(~self._valid.any(1)[None,:,None],0),attn,query

    def _mask_parents(self,module,args,kwargs):
        if self._valid is not None:
            valid=self._valid.any(1)
            if not bool(valid.all()):
                safe=valid.clone()
                if not bool(safe.any()):safe[0]=True
                kwargs['src_key_padding_mask']=~safe[None]
        return args,kwargs

    def _clear_empty_tokens(self,module,args,out):
        if self._valid is None:return out
        return out.masked_fill(~self._valid.any(1)[None,:,None],0)

    def _capture_logits(self,module,args,out):self._logits=out

    def forward_packed(self,packed):
        self._valid=packed['valid'];self._logits=None
        fine=packed['fine'].reshape(len(self._valid),16,1024)
        fine=fine.masked_fill(~self._valid[:,:,None],0).reshape(1,-1,1024)
        hazards=self.network(fine,packed['coarse'],packed['coords'])[0]
        logits=self._logits[0];self._valid=None;self._logits=None
        return dict(logits=logits,hazards=hazards,survival=torch.cumprod(1-hazards,dim=0))

    def forward(self,**inputs):return self.forward_packed(pack_inputs(inputs,self.official))


def source_likelihood(official,outputs,targets,alpha):
    """Official SurvMLE with current completed-interval censoring preserved.

For censoring, source t+1 must equal our completed_intervals. Cases with no
completed interval contribute exactly zero, including their normalization.
"""
    keep=[];labels=[]
    for i,t in enumerate(targets):
        if t['event']:
            keep.append(i);labels.append([t['event_bin'],0])
        elif t['completed_intervals']:
            keep.append(i);labels.append([t['completed_intervals']-1,1])
    if not keep:return outputs.sum()*0
    y=outputs.new_tensor(labels)
    return official(y,outputs[keep],cur_alpha=alpha)*len(keep)/len(targets)


def source_optimizer(model,official,lr=8e-5,weight_decay=5e-4):
    # Exactly the official create_optimizer bias/normalization decay grouping.
    decay=[];no_decay=[]
    for name,p in model.named_parameters():
        if p.requires_grad:(no_decay if len(p.shape)==1 or name.endswith('.bias') else decay).append(p)
    base=torch.optim.Adam([{'params':no_decay,'weight_decay':0.},{'params':decay,'weight_decay':weight_decay}],lr=lr)
    return official.Lookahead(base)


def save_optimizer(opt,model):
    if not hasattr(opt,'base_optimizer'):return dict(kind='adamw',state=opt.state_dict())
    # Legacy Lookahead serializes slow-state keys as object IDs. Save by name
    # instead so a new Python process can restore them to the correct parameters.
    slow={n:{k:v.detach().cpu() if torch.is_tensor(v) else v for k,v in opt.state[p].items()}
          for n,p in model.named_parameters() if p in opt.state}
    return dict(kind='official_lookahead',fast=opt.base_optimizer.state_dict(),slow=slow)


def load_optimizer(opt,model,state,device):
    if state['kind']=='adamw':opt.load_state_dict(state['state']);return
    opt.base_optimizer.load_state_dict(state['fast']);opt.param_groups=opt.base_optimizer.param_groups
    for n,p in model.named_parameters():
        if n in state['slow']:opt.state[p]={k:v.to(device) if torch.is_tensor(v) else v for k,v in state['slow'][n].items()}
