"""Small fixed-budget selector and an SDPA-output gate adaptation (W136).

Output-gate placement follows Qiu et al., NeurIPS 2025, arXiv:2505.06708.
This local cross-attention implementation is an adaptation, not Qwen reproduction.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .fusion_network import normalized_boxes
from .survival_network import SurvivalPredictor


class GatedLocalAttention(nn.MultiheadAttention):
    @classmethod
    def from_attention(cls, attention):
        obj=cls(attention.embed_dim,attention.num_heads,dropout=attention.dropout,batch_first=True)
        obj.load_state_dict(attention.state_dict())
        obj.output_gate=nn.Linear(attention.embed_dim,attention.num_heads)
        nn.init.zeros_(obj.output_gate.weight);nn.init.zeros_(obj.output_gate.bias)
        return obj

    def forward(self,query,key,value,key_padding_mask=None,need_weights=False,**kwargs):
        if need_weights or kwargs:raise ValueError('Only registered local cross attention is supported')
        b,t,d=query.shape;h=self.num_heads;head=d//h
        weights=self.in_proj_weight.chunk(3);biases=self.in_proj_bias.chunk(3)
        q,k,v=[F.linear(x,w,bias).view(b,-1,h,head).transpose(1,2) for x,w,bias in zip((query,key,value),weights,biases)]
        mask=None
        if key_padding_mask is not None:
            if torch.any(key_padding_mask.all(-1)):raise ValueError('Empty fine bag')
            mask=(~key_padding_mask)[:,None,None,:]
        y=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=self.dropout if self.training else 0.)
        gate=self.output_gate(query).sigmoid().transpose(1,2).unsqueeze(-1)
        y=(y*gate).transpose(1,2).reshape(b,t,d)
        return self.out_proj(y),None


def predictor(gated=False,**architecture):
    model=SurvivalPredictor(**architecture)
    if gated:model.local_attention=GatedLocalAttention.from_attention(model.local_attention)
    return model


class RegionSelector(nn.Module):
    """Scores coarse regions without reading unselected fine features or outcomes."""
    def __init__(self,dim=128):
        super().__init__()
        self.score=nn.Sequential(nn.Linear(2*dim+5,32),nn.Tanh(),nn.Linear(32,1))

    def forward(self,tokens,patient,boxes,counts,slides):
        xy=torch.zeros_like(boxes)
        for sid in torch.unique(slides):
            ix=torch.where(slides==sid)[0];xy[ix]=normalized_boxes(boxes[ix])
        counts=counts.to(tokens.device)
        x=torch.cat([tokens.detach(),patient.detach().expand(len(tokens),-1),xy,
                     counts.float()[:,None]/16.],-1)
        return self.score(x).squeeze(-1).masked_fill(counts<=0,float('-inf'))


def choose(scores,counts,fraction=.2,stochastic=False):
    ids=torch.where(counts.to(scores.device)>0)[0]
    if not len(ids):raise ValueError('No eligible fine region')
    k=max(1,math.ceil(len(ids)*fraction));k=min(k,len(ids))
    logits=scores[ids]
    if stochastic:
        # Gumbel top-k samples the exact Plackett-Luce ordered slate.
        noisy=logits-torch.empty_like(logits).exponential_().log()
        order=noisy.topk(k).indices
    else:order=logits.topk(k).indices
    probs=logits.softmax(0)
    selected=probs[order]
    remaining=1-torch.cat([selected.new_zeros(1),selected.cumsum(0)[:-1]])
    log_probability=(logits.log_softmax(0)[order]-remaining.clamp_min(1e-12).log()).sum()
    entropy=-(probs*logits.log_softmax(0)).sum()
    return ids[order].tolist(),log_probability,entropy
