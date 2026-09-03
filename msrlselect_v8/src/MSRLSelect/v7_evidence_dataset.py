"""Deterministic mask and compact cache helpers for Phase E."""
import torch
KS=(0,1,2,4,8,16,32,48,64)
def evidence_masks(n,k,seed,device='cpu'):
    g=torch.Generator(device=device).manual_seed(seed); m=torch.zeros(n,64,dtype=torch.bool,device=device)
    if k: m[torch.arange(n,device=device)[:,None],torch.stack([torch.randperm(64,generator=g,device=device)[:k] for _ in range(n)])]=True
    return m
