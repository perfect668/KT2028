#!/usr/bin/env python3
from __future__ import annotations
import json
import torch
from models.ms_reward import compute_reward
from models.ms_budget import AdaptiveBudgetController

def main():
    torch.manual_seed(1)
    tokens = torch.randn(2, 10, 16, device="cuda")
    coords = torch.rand(2, 10, 2, device="cuda")
    mask = torch.zeros(2, 10, dtype=torch.bool, device="cuda")
    mask[:, :4] = True
    out = compute_reward(torch.tensor([.7, .6], device="cuda"),
                         torch.tensor([.65, .61], device="cuda"), tokens, mask, coords)
    controller = AdaptiveBudgetController()
    gains = torch.stack([out["confidence_gain"], out["confidence_gain"] * .5])
    rewards = torch.stack([out["reward"], out["reward"] * .5])
    stop = controller.should_stop(2, gains, rewards)
    assert torch.isfinite(out["reward"]).all()
    print(json.dumps({"status": "PASS", "reward": out["reward"].tolist(),
                      "components": {k: list(v.shape) for k, v in out.items()},
                      "stop": stop.tolist()} , indent=2))

if __name__ == "__main__":
    main()
