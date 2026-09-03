#!/usr/bin/env python3
import json
import torch
from models.ms_policy import MSRegionPolicy

def main():
    torch.manual_seed(1)
    b, k, d = 4, 10, 256
    device = torch.device("cuda")
    policy = MSRegionPolicy(d, 256, max_budget=4).to(device)
    g, r, s = torch.randn(b, d, device=device), torch.randn(b, k, d, device=device), torch.rand(b, k, 5, device=device)
    mask = torch.ones(b, k, dtype=torch.bool, device=device)
    a, lp, ent, v = policy.select(g, r, s, mask)
    lp2, ent2, v2 = policy.evaluate(g, r, s, mask, a)
    assert a.shape == (b, k) and int(a.sum(1).max()) <= 4 and bool(a.any(1).all())
    assert torch.isfinite(torch.stack((lp, ent, v, lp2, ent2, v2))).all()
    print(json.dumps({"status":"PASS", "actions":list(a.shape), "selected":a.sum(1).tolist(),
                      "log_prob_delta":float((lp-lp2).abs().max())}, indent=2))

if __name__ == "__main__":
    main()
