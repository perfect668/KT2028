#!/usr/bin/env python3
"""Export frozen original HeMiCoRe stage-3 region-selection traces on SR386."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
import dhg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--args-yaml", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    # Original-package imports are intentionally isolated from MS-RLSelect.
    sys.path.insert(0, str(args.baseline_root))
    from HGNN import H_GNN
    from models import rlmil
    from utils.datasets import get_feats
    from pipeline_common import load_case, read_config, seed_all

    config = read_config(args.config)
    seed_all(int(config["seed"]))
    # The recorded local run arguments contain a serialized torch.device.
    # This file is an on-server artifact created by our own prior run.
    raw = yaml.unsafe_load(args.args_yaml.read_text())
    checkpoint = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    model = H_GNN.HGNN(
        in_dim=int(raw["feature_num"]),
        n_hid=int(raw["GCN_hidden_dim"]),
        out_dim=int(raw["feature_num"]),
        drop_rate=float(raw["GCN_dropout"]),
    ).cuda().eval()
    fc = rlmil.Full_layer(
        int(raw["feature_num"]), int(raw["fc_hidden_dim"]), bool(raw["fc_rnn"]), int(raw["num_classes"])
    ).cuda().eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    fc.load_state_dict(checkpoint["fc"])
    ppo = rlmil.PPO(
        int(raw["feature_num"]),
        int(raw["model_dim"]),
        int(raw["policy_hidden_dim"]),
        bool(raw["policy_conv"]),
        action_std=float(raw["action_std"]),
        lr=float(raw["ppo_lr"]),
        gamma=float(raw["ppo_gamma"]),
        K_epochs=int(raw["K_epochs"]),
        action_size=int(raw["num_clusters"]),
    )
    ppo.policy.load_state_dict(checkpoint["policy"])
    ppo.policy_old.load_state_dict(checkpoint["policy"])
    ppo.policy_old.eval()
    args.output_root.mkdir(parents=True, exist_ok=True)
    statistics = []
    for split in ("train", "valid", "test"):
        frame = pd.read_csv(config[f"{split}_manifest"])
        for row in frame.itertuples(index=False):
            features, coords, clusters = load_case(row)
            feature_tensor = torch.from_numpy(features).cuda().unsqueeze(0)
            coord_tensor = torch.from_numpy(coords.astype(np.int64)).cuda()
            memory = rlmil.Memory()
            action = torch.rand((1, int(raw["num_clusters"])), device="cuda")
            actions, step_indices, step_risks = [], [], []
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                for step in range(int(raw["T"])):
                    x_view, construction, coord_distribution, key_patches = get_feats(
                        [feature_tensor], [clusters], [coord_tensor], action,
                        feat_size=int(raw["feat_size"]),
                        feat_size_ratio=float(raw["feat_size_ratio"]),
                        coord_clusters=int(raw["coord_clusters"]),
                    )
                    hypergraphs = [
                        dhg.Hypergraph(int(raw["feat_size"]), construction[0] + coord_distribution[0])
                    ]
                    states, logits = model(x_view, hypergraphs)
                    logits = fc(logits, restart=step == 0)
                    step_risks.append(float(logits.softmax(1)[0, 1]))
                    selected = np.unique(
                        np.concatenate([np.asarray(edge, dtype=np.int64) for edge in key_patches[0]])
                    )
                    step_indices.append(selected)
                    actions.append(action[0].detach().cpu().numpy().astype(np.float32))
                    if step + 1 < int(raw["T"]):
                        action = ppo.select_action(states, memory, restart_batch=step == 0, training=False)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            final_indices = step_indices[-1]
            output = args.output_root / "selection_cache" / "HeMiCoRe" / split / f"{row.case_id}.npz"
            output.parent.mkdir(parents=True, exist_ok=True)
            step_means = np.stack([features[idx].mean(0) for idx in step_indices]).astype(np.float32)
            np.savez_compressed(
                output,
                selected_indices=final_indices,
                selected_coords=coords[final_indices].astype(np.float32),
                selected_features=features[final_indices].astype(np.float16),
                selected_feature_mean=step_means[-1],
                step_feature_means=step_means,
                step_selected_patch_counts=np.asarray([len(x) for x in step_indices]),
                action_sequence=np.stack(actions),
                selector_death_risk=np.asarray(step_risks, dtype=np.float32),
                inference_time_ms=np.asarray(elapsed_ms, dtype=np.float64),
            )
            repeated = int(np.stack(actions).sum()) - int((np.stack(actions) > 0).any(0).sum())
            statistics.append({
                "method": "HeMiCoRe", "case_id": str(row.case_id), "split": split,
                "selected_regions": int(len(actions)), "selected_patches": int(len(final_indices)),
                "stop_step": int(len(actions)),
                "repeat_rate": float(repeated / max(1.0, np.stack(actions).sum())),
                "inference_time_ms": elapsed_ms,
            })
        print(json.dumps({"status": "split_complete", "split": split, "n": len(frame)}), flush=True)
    pd.DataFrame(statistics).to_csv(args.output_root / "hemicore_selection_statistics.csv", index=False)
    (args.output_root / "hemicore_export_status.json").write_text(json.dumps({"status":"COMPLETE", "checkpoint": str(args.checkpoint), "test_evaluated": False}, indent=2))


if __name__ == "__main__":
    main()
