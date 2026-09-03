import torch
from torch import nn
import torch.nn.functional as F

class Top2MoERegionEncoder(nn.Module):
    """Four independent experts with differentiable Top-2 routing."""
    def __init__(self, in_dim=1024, hidden=256, centers=None):
        super().__init__(); self.experts=nn.ModuleList([nn.Sequential(nn.Linear(in_dim,hidden),nn.GELU(),nn.Linear(hidden,hidden)) for _ in range(4)])
        self.centers=nn.Parameter(torch.as_tensor(centers,dtype=torch.float32) if centers is not None else torch.randn(4,in_dim),requires_grad=False)
        self.router=nn.Linear(in_dim,4,bias=False); self.router_scale=nn.Parameter(torch.tensor(.1))
    def forward(self,x):
        x=F.normalize(x.float(),dim=-1); logits=self.router(x)+self.router_scale*(x@F.normalize(self.centers,dim=-1).t()); vals,idx=logits.topk(2,-1); rw=F.softmax(vals.float(),-1)
        out=[]; occ=[]
        for e,expert in enumerate(self.experts):
            y=expert(x); mask=idx.eq(e); weight=(mask*rw.new_zeros(mask.shape,dtype=rw.dtype)).float()
            weight=torch.where(mask, rw.gather(-1,idx.eq(e).long().argmax(-1,keepdim=True)).squeeze(-1), torch.zeros_like(rw[...,0]))
            out.append(y*weight[...,None]); occ.append(mask.float().mean())
        return torch.stack(out,1), torch.stack(occ), logits.softmax(-1).entropy(-1).mean()
