#!/usr/bin/env python3
"""Fixed-seed patch-level Attention MIL for the chapter-4 matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from pipeline_common import read_config, seed_all
from thesis_metrics import save_prediction_artifacts


class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, manifest: Path, max_patches: int, seed: int):
        frame = pd.read_csv(manifest)
        self.items = []
        for row in frame.itertuples(index=False):
            with np.load(row.features_filepath) as z:
                patches = z["img_features"]
                if len(patches) > max_patches:
                    digest = hashlib.sha256(str(row.case_id).encode()).digest()
                    local_seed = (seed + int.from_bytes(digest[:8], "little")) % (2**32)
                    selected = np.random.default_rng(local_seed).choice(
                        len(patches), size=max_patches, replace=False
                    )
                    patches = patches[np.sort(selected)]
                else:
                    patches = patches[:]
            self.items.append((patches.astype(np.float16), int(row.label), str(row.case_id)))
        self.max_patches = max_patches

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        patches, label, case_id = self.items[index]
        padded = np.zeros((self.max_patches, 1024), dtype=np.float16)
        padded[: len(patches)] = patches
        mask = np.zeros(self.max_patches, dtype=bool)
        mask[: len(patches)] = True
        return torch.from_numpy(padded), torch.from_numpy(mask), torch.tensor(label), case_id


class AttentionMIL(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(1024, 256), nn.GELU(), nn.Dropout(0.2))
        self.attention = nn.Sequential(nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1))
        self.classifier = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 2))

    def forward(self, patches, mask):
        tokens = self.projection(patches)
        scores = self.attention(tokens).squeeze(-1).masked_fill(~mask, -torch.inf)
        weights = torch.softmax(scores, dim=1)
        pooled = (tokens * weights.unsqueeze(-1)).sum(dim=1)
        return self.classifier(pooled)


def predict(model, loader, device):
    model.eval(); risks=[]; labels=[]; ids=[]
    with torch.no_grad():
        for patches, mask, y, case_ids in loader:
            logits = model(patches.to(device, dtype=torch.float32), mask.to(device))
            risks.extend(logits.softmax(1)[:,1].cpu().tolist()); labels.extend(y.tolist()); ids.extend(case_ids)
    return pd.DataFrame({"case_id":ids,"label":labels,"death_risk":risks})


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',type=Path,required=True); parser.add_argument('--output-dir',type=Path,required=True); parser.add_argument('--epochs',type=int,default=30); parser.add_argument('--batch-size',type=int,default=8); parser.add_argument('--max-patches',type=int,default=1024); args=parser.parse_args()
    config=read_config(args.config); seed_all(int(config['seed'])); args.output_dir.mkdir(parents=True,exist_ok=True)
    datasets={s:PatchDataset(Path(config[f'{s}_manifest']),args.max_patches,int(config['seed'])) for s in ('train','valid','test')}
    loaders={s:torch.utils.data.DataLoader(d,args.batch_size,shuffle=s=='train',pin_memory=True) for s,d in datasets.items()}
    device=torch.device('cuda'); model=AttentionMIL().to(device); labels=np.array([x[1] for x in datasets['train'].items]); counts=np.bincount(labels,minlength=2); loss=nn.CrossEntropyLoss(weight=torch.tensor(len(labels)/(2*counts),device=device,dtype=torch.float32)); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    best=-float('inf'); history=[]
    for epoch in range(1,args.epochs+1):
        model.train(); losses=[]
        for patches,mask,y,_ in loaders['train']:
            optimizer.zero_grad(set_to_none=True); value=loss(model(patches.to(device,dtype=torch.float32),mask.to(device)),y.to(device)); value.backward(); optimizer.step(); losses.append(float(value.detach()))
        valid=predict(model,loaders['valid'],device); auc=float(roc_auc_score(valid.label,valid.death_risk)); history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'valid_auc':auc})
        if auc>best: best=auc; torch.save({'model':model.state_dict(),'epoch':epoch,'max_patches':args.max_patches},args.output_dir/'model_best.pt')
    checkpoint=torch.load(args.output_dir/'model_best.pt',map_location=device,weights_only=False); model.load_state_dict(checkpoint['model']); metrics=save_prediction_artifacts(args.output_dir,{s:predict(model,loaders[s],device) for s in ('train','valid','test')})
    (args.output_dir/'metrics_history.json').write_text(json.dumps(history,indent=2)); (args.output_dir/'run_status.json').write_text(json.dumps({'status':'COMPLETE','model':'patch_level_attention_mil','checkpoint_selection':'validation_auc','selected_epoch':checkpoint['epoch'],'max_patches':args.max_patches,'test_evaluated_once_after_selection':True},indent=2))

if __name__=='__main__': main()
