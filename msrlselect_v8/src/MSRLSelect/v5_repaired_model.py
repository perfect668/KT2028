from __future__ import annotations
import math
import torch
from torch import nn

def finite(x,name):
    if not torch.isfinite(x).all(): raise FloatingPointError(f'non-finite tensor at {name}')

class SafeGatedPool(nn.Module):
    def __init__(self,d): super().__init__(); self.v=nn.Linear(d,d); self.u=nn.Linear(d,d); self.a=nn.Linear(d,1)
    def forward(self,x,mask):
        score=self.a(torch.tanh(self.v(x))*torch.sigmoid(self.u(x))).squeeze(-1)
        score=score.float().masked_fill(~mask,-1e4); w=torch.softmax(score,-1); w=w*mask.float(); den=w.sum(-1,keepdim=True).clamp_min(1e-8); w=w/den
        return (x*w.to(x.dtype).unsqueeze(-1)).sum(1),w
class MeanPool(nn.Module):
    def forward(self,x,mask):
        m=mask.float(); return (x*m.unsqueeze(-1)).sum(1)/m.sum(1,keepdim=True).clamp_min(1),None

class LocalPool(nn.Module):
    def __init__(self,d,h=8):
        super().__init__(); self.cls=nn.Parameter(torch.randn(1,1,d)*.02); l=nn.TransformerEncoderLayer(d,h,d*4,.2,'gelu',batch_first=True,norm_first=True); self.enc=nn.TransformerEncoder(l,2,enable_nested_tensor=False)
    def forward(self,x,m):
        b=x.size(0); z=torch.cat([self.cls.expand(b,1,-1),x],1); pm=torch.cat([torch.zeros(b,1,dtype=torch.bool,device=x.device),~m],1); return self.enc(z,src_key_padding_mask=pm)[:,0],None

class SpatialBlock(nn.Module):
    def __init__(self,d=384,h=8,drop=.2,spatial=True):
        super().__init__(); self.d=d; self.h=h; self.spatial=spatial; self.n=nn.LayerNorm(d); self.qkv=nn.Linear(d,3*d); self.o=nn.Linear(d,d); self.ff=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d*4),nn.GELU(),nn.Dropout(drop),nn.Linear(d*4,d)); self.bias=nn.Sequential(nn.Linear(5,64),nn.GELU(),nn.Linear(64,h)) if spatial else None; self.gate=nn.Parameter(torch.tensor(-8.0)) if spatial else None
    def forward(self,x,mask,cent,size):
        b,n,d=x.shape; q,k,v=self.qkv(self.n(x)).chunk(3,-1); q=q.view(b,n,self.h,d//self.h).transpose(1,2); k=k.view(b,n,self.h,d//self.h).transpose(1,2); v=v.view(b,n,self.h,d//self.h).transpose(1,2); logits=q@k.transpose(-2,-1)/math.sqrt(d//self.h)
        if self.spatial:
            delta=cent[:,:,None,:]-cent[:,None,:,:]; dist=torch.sqrt((delta**2).sum(-1,keepdim=True)+1e-6); lr=torch.log(dist+1e-4); ratio=torch.log((size[:,:,None]+1e-3)/(size[:,None,:]+1e-3)).unsqueeze(-1); feat=torch.cat([delta,dist,lr,ratio],-1); bias=self.bias(feat).permute(0,3,1,2); logits=logits+torch.sigmoid(self.gate)*bias
        logits=logits.float().masked_fill(~mask[:,None,None,:],-1e4).masked_fill(~mask[:,None,:,None],-1e4); att=torch.softmax(logits,-1); y=(att@v.float()).to(x.dtype).transpose(1,2).reshape(b,n,d); y=self.o(y); y=y.masked_fill(~mask.unsqueeze(-1),0); return x+y+self.ff(x+y),att

class RepairedDenseModel(nn.Module):
    def __init__(self,input_dim=1024,d=384,layers=4,heads=8,pool='gated_attention_pool',spatial=True,no_relation=False):
        super().__init__(); self.pool_name=pool; self.spatial=spatial; self.no_relation=no_relation; self.proj=nn.Sequential(nn.Linear(input_dim,d),nn.GELU(),nn.Dropout(.2)); self.pool=MeanPool() if pool=='mean_pool' else (SafeGatedPool(d) if pool=='gated_attention_pool' else LocalPool(d,heads)); self.blocks=nn.ModuleList([SpatialBlock(d,heads,.2,spatial and not no_relation) for _ in range(layers)]); self.cls=nn.Parameter(torch.randn(1,1,d)*.02); self.head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,2))
    def forward(self,patches,patch_mask,region_ptr,cent,size,region_mask,return_aux=False):
        b,r,p,di=patches.shape; z=self.proj(patches.reshape(b*r,p,di)); rm=patch_mask.reshape(b*r,p); pooled,w=self.pool(z,rm); pooled=pooled.reshape(b,r,-1); tokens=torch.cat([self.cls.expand(b,1,-1),pooled],1); m=torch.cat([torch.ones(b,1,dtype=torch.bool,device=patches.device),region_mask],1); c=torch.cat([torch.zeros(b,1,2,device=patches.device,dtype=cent.dtype),cent],1); s=torch.cat([torch.ones(b,1,device=patches.device,dtype=size.dtype),size],1)
        gates=[]; atts=[]
        for block in self.blocks: tokens,a=block(tokens,m,c,s); atts.append(a); gates.append(torch.sigmoid(block.gate).detach().float().item() if block.gate is not None else 0.)
        out=self.head(tokens[:,0]); finite(out,'logits'); aux={'pool_weights':w.reshape(b,r,p) if w is not None else None,'spatial_gates':gates,'attention':atts,'tokens':tokens}; return (out,aux) if return_aux else out
