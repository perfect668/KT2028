import math, torch
from torch import nn
import torch.nn.functional as F

class ResidualExpert(nn.Module):
    def __init__(self,d=384,hidden=768,dropout=.2):
        super().__init__();self.net=nn.Sequential(nn.Linear(2*d,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,d))
    def forward(self,z,c): return z+self.net(torch.cat([z,c],-1))

class E2MModel(nn.Module):
    """Fixed E2 backbone with trainable patient/region-conditioned Top-2 experts."""
    def __init__(self,base,coarse_dim,slot_dim=256,d=384):
        super().__init__();self.base=base;self.d=d
        self.slot_projection=nn.Linear(slot_dim,d)
        self.experts=nn.ModuleList([ResidualExpert(d) for _ in range(4)])
        self.router=nn.Sequential(nn.Linear(3*d,d),nn.GELU(),nn.Linear(d,1))
        self.slot_identity=nn.Parameter(torch.zeros(4,d));nn.init.normal_(self.slot_identity,std=.02)
        self.beta_gate=nn.Sequential(nn.Linear(3*d,128),nn.GELU(),nn.Linear(128,1))
        nn.init.zeros_(self.beta_gate[-1].weight);nn.init.constant_(self.beta_gate[-1].bias,math.log(.1/.9))
        self.region_norm=nn.LayerNorm(d);self.output_head=nn.Linear(d,1);self.output_head.load_state_dict(base.fh.state_dict())
    def set_stage(self,stage):
        for p in self.parameters():p.requires_grad=False
        for mod in [self.slot_projection,self.experts,self.router,self.beta_gate,self.region_norm,self.output_head,self.slot_identity]:
            for p in (mod.parameters() if hasattr(mod,'parameters') else [mod]):p.requires_grad=True
        if stage=='B':
            for mod in [self.base.lproj,self.base.res,self.base.alpha,self.base.gpool,self.base.gh,self.base.lh,self.base.uh]:
                for p in mod.parameters():p.requires_grad=True
    def encode_regions(self,coarse,slots,disable_moe=False,shuffle_global=False,shuffle_region=False,disabled_expert=None):
        gtok=self.base.cg(coarse);gw=F.softmax(self.base.gpool(gtok).squeeze(-1).float(),-1).to(gtok.dtype);g=(gtok*gw[...,None]).sum(1)
        b=self.base.lproj(slots.flatten(-2));z=self.slot_projection(slots)+self.slot_identity[None,None]
        present=slots.norm(dim=-1)>0;c=gtok
        gg=g[:,None,None,:].expand(-1,64,4,-1);cc=c[:,:,None,:].expand(-1,-1,4,-1)
        if shuffle_global:gg=gg[torch.randperm(len(gg),device=gg.device)]
        if shuffle_region:cc=cc[:,torch.randperm(64,device=cc.device)]
        logits=self.router(torch.cat([gg,cc,z],-1)).squeeze(-1).masked_fill(~present,-1e9)
        topv,topi=logits.topk(2,-1);weights=torch.zeros_like(logits).scatter(-1,topi,F.softmax(topv.float(),-1).to(logits.dtype));weights=weights*present
        if disabled_expert is not None:weights[...,disabled_expert]=0
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
        eo=torch.stack([self.experts[i](z[:,:,i],c) for i in range(4)],2);m=(weights[...,None]*eo).sum(2)
        beta=torch.sigmoid(self.beta_gate(torch.cat([g[:,None].expand(-1,64,-1),c,b],-1))).squeeze(-1)
        h=self.region_norm(b+(0 if disable_moe else beta[...,None]*m));return g,h,dict(router_logits=logits,weights=weights,present=present,beta=beta,expert_outputs=eo,global_repr=g,coarse_repr=c)
    def forward(self,coarse,slots,selected_mask,disable_moe=False,shuffle_global=False,shuffle_region=False,disabled_expert=None):
        g,h,a=self.encode_regions(coarse,slots,disable_moe,shuffle_global,shuffle_region,disabled_expert);B,M,_=selected_mask.shape;n=selected_mask.sum(-1,keepdim=True).float();local=(h[:,None]*selected_mask[...,None]).sum(2)/n.clamp_min(1);rho=n/64.;inp=torch.cat([g[:,None].expand(-1,M,-1),local,rho],-1);alpha=torch.sigmoid(self.base.alpha(inp)).squeeze(-1);alpha=torch.where(n.squeeze(-1)==0,torch.zeros_like(alpha),alpha);state=F.layer_norm(g[:,None]+alpha[...,None]*self.base.res(inp),(self.d,));return dict(fused_logit=self.output_head(state).squeeze(-1),global_logit=self.base.gh(g),local_logit=self.base.lh(local).squeeze(-1),uncertainty=torch.sigmoid(self.base.uh(g)).squeeze(-1),alpha=alpha,state=state,**a)
