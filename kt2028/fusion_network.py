"""Coarse context remains visible; only real observed fine modifies its risk."""
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from .network import GatedPool


def normalized_boxes(boxes):
    origin = boxes[:, :2].amin(0)
    span = (boxes[:, 2:].amax(0)-origin).clamp_min(1)
    return (boxes-origin.repeat(2))/span.repeat(2)


def context_encoder(dim, heads, dropout):
    layer = nn.TransformerEncoderLayer(dim, heads, 2*dim, dropout,
                                      batch_first=True, norm_first=True, activation='gelu')
    return nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)


class CoarseContext(nn.Module):
    def __init__(self, dim=256, heads=4, dropout=.2):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(1024, dim), nn.GELU())
        self.position = nn.Sequential(nn.Linear(4, 64), nn.GELU(), nn.Linear(64, dim))
        self.encoder = context_encoder(dim, heads, dropout)
        self.pool = GatedPool(dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, coarse, boxes):
        if coarse.ndim != 2 or coarse.shape[1] != 1024 or len(coarse) == 0 or boxes.shape != (len(coarse), 4):
            raise ValueError('Invalid coarse context shape')
        if not torch.isfinite(coarse).all() or not torch.isfinite(boxes).all():
            raise ValueError('Nonfinite coarse context')
        position = self.position(normalized_boxes(boxes))
        tokens = self.encoder((self.projection(coarse)+position)[None])[0]
        z = self.pool(tokens)[0][0]
        return tokens, z, self.head(z).squeeze(-1), position


class CrossScalePredictor(nn.Module):
    def __init__(self, dim=256, heads=4, dropout=.2):
        super().__init__()
        self.dim, self.coarse_frozen = dim, False
        self.coarse = CoarseContext(dim, heads, dropout)
        self.fine_projection = nn.Sequential(nn.Linear(1024, dim), nn.GELU())
        self.fine_position = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, dim))
        self.local_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.fusion = nn.Sequential(nn.Linear(2*dim, dim), nn.GELU())
        self.fine_encoder = context_encoder(dim, heads, dropout)
        self.fine_pool = GatedPool(dim)
        self.delta = nn.Sequential(nn.Linear(2*dim+2, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, 1))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)
        self.aux_pool = GatedPool(dim)
        self.aux_head = nn.Linear(dim, 1)

    def freeze_coarse(self):
        self.coarse_frozen = True
        self.coarse.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        if self.coarse_frozen:
            self.coarse.eval()
        return self

    def forward(self, coarse, boxes, observed_fine, relative_xy, candidate_patch_count):
        tokens, zc, lc, position = self.coarse(coarse, boxes)
        ids = sorted(observed_fine)
        if set(relative_xy) != set(ids) or any(not isinstance(i, int) or not 0 <= i < len(tokens) for i in ids):
            raise ValueError('Fine/position observation keys differ or are invalid')
        if not ids:
            return dict(logit=lc, coarse_logit=lc, delta=lc.new_zeros(()), fine_aux_logit=None,
                        coarse_tokens=tokens, coarse_patient=zc, fine_patient=torch.zeros_like(zc),
                        observed_tokens=tokens[:0], observed_ids=ids)
        pure, positioned, lengths = [], [], []
        for i in ids:
            fine, rel = observed_fine[i], relative_xy[i]
            if fine.ndim != 2 or fine.shape[1] != 1024 or not len(fine) or rel.shape != (len(fine), 2):
                raise ValueError('Observed fine/relative shape invalid')
            if not torch.isfinite(fine).all() or not torch.isfinite(rel).all():
                raise ValueError('Nonfinite observed evidence')
            f = self.fine_projection(fine)
            pure.append(f)
            positioned.append(f+self.fine_position(rel))
            lengths.append(len(f))
        if candidate_patch_count < sum(lengths) or candidate_patch_count <= 0:
            raise ValueError('Invalid total candidate patch count')
        padded = pad_sequence(positioned, batch_first=True)
        mask = torch.arange(padded.shape[1], device=coarse.device)[None] >= torch.tensor(lengths, device=coarse.device)[:, None]
        ids_tensor = torch.tensor(ids, device=coarse.device)
        query = tokens[ids_tensor, None, :]
        local = self.local_attention(query, padded, padded, key_padding_mask=mask, need_weights=False)[0][:, 0]
        observed = self.fusion(torch.cat((tokens[ids_tensor], local), -1))+position[ids_tensor]
        observed = self.fine_encoder(observed[None])[0]
        zf = self.fine_pool(observed)[0][0]
        q = lc.new_tensor([len(ids)/len(tokens), sum(lengths)/candidate_patch_count])
        delta = self.delta(torch.cat((zc, zf, q))).squeeze(-1)
        auxiliary = self.aux_head(self.aux_pool(torch.cat(pure))[0][0]).squeeze(-1)
        return dict(logit=lc+delta, coarse_logit=lc, delta=delta, fine_aux_logit=auxiliary,
                    coarse_tokens=tokens, coarse_patient=zc, fine_patient=zf,
                    observed_tokens=observed, observed_ids=ids)
