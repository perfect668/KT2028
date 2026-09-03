from __future__ import annotations
import math
import torch
from torch import nn

def finite(x, name):
    if not torch.isfinite(x).all():
        idx=(~torch.isfinite(x)).nonzero()[0].tolist()
        raise FloatingPointError(f'non-finite at {name}, first_index={idx}')

class SafeGatedPool(nn.Module):
    def __init__(self,d):
        super().__init__(); self.v=nn.Linear(d,d); self.u=nn.Linear(d,d); self.a=nn.Linear(d,1)
    def forward(self,x,mask):
        score=self.a(torch.tanh(self.v(x))*torch.sigmoid(self.u(x))).squeeze(-1).float()
        score=score.masked_fill(~mask,-1e4)
        w=torch.softmax(score,-1)*mask.float(); w=w/w.sum(-1,keepdim=True).clamp_min(1e-8)
        out=(x*w.to(x.dtype).unsqueeze(-1)).sum(1); finite(out,'gated_pool'); return out,w

class MeanPool(nn.Module):
    def forward(self,x,mask):
        m=mask.float(); out=(x*m.unsqueeze(-1)).sum(1)/m.sum(1,keepdim=True).clamp_min(1)
        finite(out,'mean_pool'); return out,None

class SpatialBlock(nn.Module):
    def __init__(self,d=384,h=8,drop=.2,spatial=False):
        super().__init__(); self.d=d; self.h=h; self.spatial=spatial
        self.n=nn.LayerNorm(d); self.qkv=nn.Linear(d,3*d); self.o=nn.Linear(d,d)
        self.ff=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d*4),nn.GELU(),nn.Dropout(drop),nn.Linear(d*4,d))
        self.bias_mlp=nn.Sequential(nn.Linear(5,64),nn.GELU(),nn.Linear(64,h)) if spatial else None
        self.gate=nn.Parameter(torch.full((h,),-8.0)) if spatial else None
    def forward(self,x,mask,cent,size,collect=False):
        b,n,d=x.shape; q,k,v=self.qkv(self.n(x)).chunk(3,-1)
        q=q.view(b,n,self.h,d//self.h).transpose(1,2); k=k.view(b,n,self.h,d//self.h).transpose(1,2); v=v.view(b,n,self.h,d//self.h).transpose(1,2)
        logits=(q@k.transpose(-2,-1))/math.sqrt(d//self.h); raw_bias=None; gated_bias=None
        if self.spatial:
            delta=cent[:,:,None,:]-cent[:,None,:,:]; dist=torch.sqrt((delta**2).sum(-1,keepdim=True)+1e-6)
            ratio=torch.log((size[:,:,None]+1e-3)/(size[:,None,:]+1e-3)).unsqueeze(-1)
            feat=torch.cat([delta,dist,torch.log(dist+1e-4),ratio],-1)
            raw_bias=self.bias_mlp(feat).permute(0,3,1,2)
            raw_bias=raw_bias.clone(); raw_bias[:,:,0,:]=0; raw_bias[:,:,:,0]=0
            gated_bias=torch.sigmoid(self.gate)[None,:,None,None]*raw_bias
            logits=logits+gated_bias.to(logits.dtype)
        valid=mask[:,None,:,None]&mask[:,None,None,:]
        logits=logits.float().masked_fill(~valid,-1e4); att=torch.softmax(logits,-1)*valid.float()
        att=att/att.sum(-1,keepdim=True).clamp_min(1e-8)
        y=(att@v.float()).to(x.dtype).transpose(1,2).reshape(b,n,d)
        z=x+self.o(y); z=z+self.ff(z); z=z.masked_fill(~mask.unsqueeze(-1),0); finite(z,'spatial_block')
        aux={'attention':att}
        if collect and raw_bias is not None:
            aux.update(raw_bias=raw_bias,gated_bias=gated_bias)
        return z,aux

class Stage1CDenseModel(nn.Module):
    def __init__(self,input_dim=1024,d=384,layers=4,heads=8,pool='gated_attention_pool',spatial=False,relation=True,dropout=.2):
        super().__init__(); self.pool_name=pool; self.spatial=spatial; self.relation=relation
        self.proj=nn.Sequential(nn.Linear(input_dim,d),nn.GELU(),nn.Dropout(dropout))
        self.pool=MeanPool() if pool=='mean_pool' else SafeGatedPool(d)
        self.blocks=nn.ModuleList([SpatialBlock(d,heads,dropout,spatial) for _ in range(layers)]) if relation else nn.ModuleList()
        self.cls=nn.Parameter(torch.randn(1,1,d)*.02); self.head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,2))
    def forward(self,patches,patch_mask,cent,size,region_mask,collect=False,bypass_spatial=False):
        b,r,p,di=patches.shape; z=self.proj(patches.reshape(b*r,p,di)); finite(z,'projection')
        pooled,w=self.pool(z,patch_mask.reshape(b*r,p)); pooled=pooled.reshape(b,r,-1)
        if not self.relation:
            m=region_mask.float(); rep=(pooled*m.unsqueeze(-1)).sum(1)/m.sum(1,keepdim=True).clamp_min(1); block_aux=[]
        else:
            tokens=torch.cat([self.cls.expand(b,1,-1),pooled],1)
            m=torch.cat([torch.ones(b,1,dtype=torch.bool,device=patches.device),region_mask],1)
            c=torch.cat([torch.zeros(b,1,2,device=cent.device,dtype=cent.dtype),cent],1)
            s=torch.cat([torch.ones(b,1,device=size.device,dtype=size.dtype),size],1); block_aux=[]
            for block in self.blocks:
                old=block.spatial
                if bypass_spatial: block.spatial=False
                tokens,a=block(tokens,m,c,s,collect)
                block.spatial=old; block_aux.append(a)
            region_mean=(tokens[:,1:]*region_mask.unsqueeze(-1).to(tokens.dtype)).sum(1)/region_mask.sum(1,keepdim=True).clamp_min(1).to(tokens.dtype)
            rep=tokens[:,0]+region_mean
        out=self.head(rep); finite(out,'logits')
        aux={'pool_weights':None if w is None else w.reshape(b,r,p),'blocks':block_aux,
             'gates':[torch.sigmoid(x.gate).detach().cpu().tolist() for x in self.blocks if x.gate is not None]}
        return (out,aux) if collect else out
