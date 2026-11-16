"""W159 continuous interval exposure likelihood with unchanged annual outputs.

eta_j=softplus(logit_j) is integrated hazard over the registered interval.
Thus exp(-eta_j)=1-sigmoid(logit_j), preserving the exact existing annual
survival curve. Time density is expressed per year (365 days).
"""
import torch
from torch.nn import functional as F

from .survival import encode_target


def exposure_nll(logits, target, edges):
    if logits.ndim!=1 or not torch.isfinite(logits).all():
        raise ValueError('Expected finite annual interval logits')
    expected=encode_target(target['time_days'],target['event'],edges)
    if any(expected[k]!=target[k] for k in expected) or len(logits)!=len(edges)-1:
        raise ValueError('Target/edges differ from the registered annual contract')
    t=float(target['time_days']);event=int(target['event'])
    if event:
        j=int(target['event_bin'])
    else:
        j=min(int(target['completed_intervals']),len(logits)-1)
    duration=float(edges[j+1]-edges[j])
    fraction=(t-float(edges[j]))/duration
    if not 0<=fraction<=1:
        raise ValueError('Invalid within-interval exposure')
    rates=F.softplus(logits)
    cumulative=rates[:j].sum()+fraction*rates[j]
    if not event:
        return cumulative
    # log(softplus(x)) ~ x for very negative x; clamp only the inactive branch
    # to keep backward finite when float32 softplus would underflow to zero.
    log_rate=torch.where(logits[j]<-20,logits[j],rates[j].clamp_min(torch.finfo(logits.dtype).tiny).log())
    log_width_years=logits.new_tensor(duration/365.).log()
    return cumulative-log_rate+log_width_years
