"""W135: R3 backbone with cancer-conditioned readouts; OS adaptation, not paper replication."""
import torch
from torch import nn
from .survival_network import SurvivalPredictor

CANCERS = ('CRC', 'STAD', 'LUAD', 'PAAD')


class CancerPredictor(SurvivalPredictor):
    def __init__(self, kind, **kwargs):
        super().__init__(**kwargs)
        if kind not in ('shared', 'moe'):
            raise ValueError(kind)
        self.kind = kind
        del self.survival_head
        self.cancer_encoder = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU())
        if kind == 'shared':
            self.conditioned_head = nn.Linear(self.dim + 128, self.intervals)
        else:
            self.cancer_gate = nn.Linear(128, 4)
            self.experts = nn.ModuleList([nn.Sequential(nn.Linear(self.dim + 128, 128), nn.ReLU(),
                nn.Dropout(kwargs.get('dropout', .25)), nn.Linear(128, self.intervals)) for _ in CANCERS])

    def readout(self, zc, zf, q):
        z = self.patient_fusion(torch.cat((zc, zf, q)))
        e = self.cancer_encoder(z.new_tensor([int(i == self._cancer_id) for i in range(4)]))
        x = torch.cat((z, e))
        if self.kind == 'shared':
            return self.conditioned_head(x)
        weights = self.cancer_gate(e).softmax(-1)
        return (weights[:, None] * torch.stack([expert(x) for expert in self.experts])).sum(0)

    def forward(self, *args, cancer_id, **kwargs):
        if not 0 <= cancer_id < 4:
            raise ValueError('Unknown cancer category')
        self._cancer_id = int(cancer_id)
        return super().forward(*args, **kwargs)

    def routing(self, device):
        if self.kind != 'moe':
            return None
        with torch.no_grad():
            e = self.cancer_encoder(torch.eye(4, device=device))
            return self.cancer_gate(e).softmax(-1).cpu().tolist()
