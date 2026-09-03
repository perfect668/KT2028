import torch
from torch import nn
import torch.nn.functional as F

class EvidenceNet(nn.Module):
    """Global coarse encoder plus selected-fine content-conditioned residual."""
    def __init__(self, coarse_dim, fine_dim=384, d_model=384, dropout=.30):
        super().__init__(); self.d_model=d_model
        self.coarse=nn.Sequential(nn.Linear(coarse_dim,d_model),nn.GELU(),nn.Dropout(dropout))
        self.pool_score=nn.Linear(d_model,1); self.fine=nn.Linear(fine_dim,d_model)
        self.residual=nn.Sequential(nn.Linear(2*d_model+1,256),nn.GELU(),nn.Dropout(dropout),nn.Linear(256,d_model))
        self.alpha=nn.Sequential(nn.Linear(2*d_model+1,128),nn.GELU(),nn.Linear(128,1))
        self.global_head=nn.Linear(d_model,1); self.local_head=nn.Linear(d_model,1); self.fused_head=nn.Linear(d_model,1); self.uncertainty=nn.Linear(d_model,1)
    def forward(self, coarse, fine, selected_mask):
        gtok=self.coarse(coarse); w=F.softmax(self.pool_score(gtok).squeeze(-1).float(),-1).to(gtok.dtype); g=(gtok*w[...,None]).sum(1)
        x=self.fine(fine); x=x[:,None].expand(coarse.shape[0],selected_mask.shape[1],64,-1)
        n=selected_mask.sum(-1,keepdim=True).float(); local=(x*selected_mask[...,None]).sum(2)/n.clamp_min(1)
        empty=n.eq(0); rho=n/64.; inp=torch.cat([g[:,None].expand_as(local),local,rho],-1)
        a=torch.sigmoid(self.alpha(inp)).squeeze(-1); a=torch.where(empty.squeeze(-1),torch.zeros_like(a),a)
        z=F.layer_norm(g[:,None,:]+a[...,None]*self.residual(inp),(self.d_model,))
        return self.fused_head(z).squeeze(-1),self.global_head(g).squeeze(-1),self.local_head(local).squeeze(-1),torch.sigmoid(self.uncertainty(g)).squeeze(-1),a
