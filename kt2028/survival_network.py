"""DSCA-inspired partial-observation dual streams with cross gates and shared head.

Adapted design, not a full DSCA reproduction: variable fine bags, no square Conv2D,
explicit empty observations, independent heads removed, no frozen coarse anchor.
"""
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from .fusion_network import normalized_boxes, context_encoder
from .network import GatedPool
from .survival import curve_from_logits


class SurvivalPredictor(nn.Module):
    def __init__(self, dim=256, heads=4, dropout=.25, intervals=5, fusion='cross_gate'):
        super().__init__()
        if fusion not in ('cross_gate', 'concat') or intervals < 2 or dim % heads:
            raise ValueError('Invalid survival architecture')
        self.dim, self.intervals, self.fusion_mode = dim, intervals, fusion
        self.coarse_projection = nn.Sequential(nn.Linear(1024, dim), nn.GELU())
        self.region_position = nn.Sequential(nn.Linear(4, 64), nn.GELU(), nn.Linear(64, dim))
        self.coarse_encoder = context_encoder(dim, heads, dropout)
        self.coarse_pool = GatedPool(dim)
        self.fine_projection = nn.Sequential(nn.Linear(1024, dim), nn.GELU())
        self.fine_position = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, dim))
        self.local_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.fine_only_query = nn.Parameter(torch.zeros(1, 1, dim))
        self.fine_encoder = context_encoder(dim, heads, dropout)
        self.fine_pool = GatedPool(dim)
        # Gate each observed coarse token from the fine stream, and vice versa.
        self.coarse_gate = nn.Sequential(nn.Linear(dim+3, dim), nn.Sigmoid())
        self.fine_gate = nn.Sequential(nn.Linear(2*dim+3, dim), nn.Sigmoid())
        self.patient_fusion = nn.Sequential(nn.Linear(2*dim+3, dim), nn.GELU(), nn.LayerNorm(dim), nn.Dropout(dropout))
        self.survival_head = nn.Linear(dim, intervals)

    def readout(self, zc, zf, q):
        return self.survival_head(self.patient_fusion(torch.cat((zc, zf, q))))

    def context_by_slide(self, encoder, values, slide_indices):
        if slide_indices is None or len(torch.unique(slide_indices)) == 1:
            return encoder(values[None])[0]
        result = torch.zeros_like(values)
        for sid in torch.unique(slide_indices):
            ix = torch.nonzero(slide_indices == sid, as_tuple=True)[0]
            result = result.index_copy(0, ix, encoder(values[ix][None])[0])
        return result

    def encode_fine(self, queries, padded, mask, positions, slide_indices=None):
        local = self.local_attention(queries, padded, padded, key_padding_mask=mask, need_weights=False)[0][:, 0]
        return self.context_by_slide(self.fine_encoder, local+positions, slide_indices)

    def forward(self, coarse, boxes, observed_fine, relative_xy, candidate_patch_count, *, return_aux=False, slide_indices=None):
        if coarse.ndim != 2 or coarse.shape[1] != 1024 or not len(coarse) or boxes.shape != (len(coarse), 4):
            raise ValueError('Invalid coarse input')
        if not torch.isfinite(coarse).all() or not torch.isfinite(boxes).all():
            raise ValueError('Nonfinite coarse input')
        ids = sorted(observed_fine)
        if set(ids) != set(relative_xy) or any(not isinstance(i, int) or not 0 <= i < len(coarse) for i in ids):
            raise ValueError('Invalid observed-region/position keys')
        if candidate_patch_count < 0:
            raise ValueError('Invalid candidate patch count')
        if slide_indices is not None:
            if slide_indices.shape != (len(coarse),) or slide_indices.dtype != torch.int64 or torch.any(slide_indices < 0):
                raise ValueError('Invalid slide membership')
            normalized = torch.zeros_like(boxes)
            for sid in torch.unique(slide_indices):
                ix = torch.nonzero(slide_indices == sid, as_tuple=True)[0]
                normalized[ix] = normalized_boxes(boxes[ix])
        else:
            normalized = normalized_boxes(boxes)
        pos = self.region_position(normalized)
        hc = self.context_by_slide(self.coarse_encoder, self.coarse_projection(coarse)+pos, slide_indices)
        zc = self.coarse_pool(hc)[0][0]
        zero = zc.new_zeros(self.dim)
        q0 = zc.new_zeros(3)
        coarse_logits = self.readout(zc, zero, q0) if return_aux or not ids else None
        if not ids:
            hazards, survival = curve_from_logits(coarse_logits)
            return dict(logits=coarse_logits, hazards=hazards, survival=survival,
                        coarse_aux_logits=coarse_logits if return_aux else None, fine_aux_logits=None,
                        coarse_tokens=hc, coarse_patient=zc, fine_patient=zero, observed_tokens=hc[:0],
                        observed_ids=ids, observation_state=q0, coarse_gates=hc[:0], fine_gates=hc[:0])
        bags, lengths = [], []
        for i in ids:
            f, rel = observed_fine[i], relative_xy[i]
            if f.ndim != 2 or f.shape[1] != 1024 or not len(f) or rel.shape != (len(f), 2):
                raise ValueError('Invalid observed fine bag/relative positions')
            if not torch.isfinite(f).all() or not torch.isfinite(rel).all() or torch.any((rel < 0) | (rel > 1)):
                raise ValueError('Nonfinite or out-of-parent fine center')
            bags.append(self.fine_projection(f)+self.fine_position(rel))
            lengths.append(len(f))
        if sum(lengths) > candidate_patch_count or candidate_patch_count <= 0:
            raise ValueError('Observed fine exceeds eligible patch count')
        padded = pad_sequence(bags, batch_first=True)
        mask = torch.arange(padded.shape[1], device=coarse.device)[None] >= torch.tensor(lengths, device=coarse.device)[:, None]
        ix = torch.tensor(ids, device=coarse.device)
        fine_slides = slide_indices[ix] if slide_indices is not None else None
        hf = self.encode_fine(hc[ix, None, :], padded, mask, pos[ix], fine_slides)
        q = zc.new_tensor([1., len(ids)/len(hc), sum(lengths)/candidate_patch_count])
        qrows = q.expand(len(ids), -1)
        if self.fusion_mode == 'cross_gate':
            gc = self.coarse_gate(torch.cat((hf, qrows), -1))
            gf = self.fine_gate(torch.cat((hc[ix], zc.expand(len(ids), -1), qrows), -1))
        else:
            gc, gf = torch.ones_like(hf), torch.ones_like(hf)
        fused_coarse = hc.clone()
        fused_coarse[ix] = gc*hc[ix]  # All other coarse tokens remain in the pool.
        zc_fused = self.coarse_pool(fused_coarse)[0][0]
        gated_fine = gf*hf
        zf = self.fine_pool(gated_fine)[0][0]
        logits = self.readout(zc_fused, zf, q)
        fine_aux = None
        if return_aux:
            # Same complete fine trunk, with a learned query independent of coarse values.
            pure = self.encode_fine(self.fine_only_query.expand(len(ids), -1, -1), padded, mask, pos[ix], fine_slides)
            fine_aux = self.readout(zero, self.fine_pool(pure)[0][0], q)
        hazards, survival = curve_from_logits(logits)
        return dict(logits=logits, hazards=hazards, survival=survival,
                    coarse_aux_logits=coarse_logits, fine_aux_logits=fine_aux,
                    coarse_tokens=hc, coarse_patient=zc_fused, fine_patient=zf,
                    observed_tokens=gated_fine, observed_ids=ids, observation_state=q,
                    coarse_gates=gc, fine_gates=gf)
