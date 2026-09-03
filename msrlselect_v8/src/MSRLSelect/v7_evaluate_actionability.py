"""Candidate-by-candidate action evaluation; no P3 action-value reuse."""
import torch
def enumerate_acquire(model,coarse,fine,active):
    n=coarse.shape[0]; out=[]
    for i in range(64):
        if active[:,i].any(): continue
        m=active.clone();m[:,i]=True
        with torch.no_grad(): out.append(torch.sigmoid(model(coarse,fine,m[:,None])[0][:,0]))
    return torch.stack(out,1) if out else coarse.new_empty((n,0))
