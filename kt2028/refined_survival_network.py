"""W159: individually switchable R3 repairs; legacy R3 remains unchanged."""
import copy

import torch
from torch import nn

from .survival_network import SurvivalPredictor


class TwiceSigmoid(nn.Module):
    def forward(self, value):
        return 2*value.sigmoid()


class AttentionScaledEncoderLayer(nn.TransformerEncoderLayer):
    """Pre-LN with channelwise scaling on the attention residual only.

    Override forward explicitly: a fused TransformerEncoderLayer eval kernel
    would otherwise omit the new parameter. The FFN and its dropout are intact.
    """
    @classmethod
    def from_layer(cls, original, initial_scale):
        weight=original.linear1.weight
        devices=[weight.device.index] if weight.is_cuda else []
        # Preserve unrelated initial weights and the subsequent training RNG.
        with torch.random.fork_rng(devices=devices):
            result=cls(original.self_attn.embed_dim,original.self_attn.num_heads,
                dim_feedforward=original.linear1.out_features,dropout=original.dropout.p,
                activation=copy.deepcopy(original.activation),layer_norm_eps=original.norm1.eps,
                batch_first=original.self_attn.batch_first,norm_first=original.norm_first,
                bias=original.linear1.bias is not None,device=weight.device,dtype=weight.dtype)
        result.load_state_dict(original.state_dict(),strict=True)
        result.register_parameter('attention_scale',nn.Parameter(torch.full(
            (original.self_attn.embed_dim,),float(initial_scale),
            dtype=original.linear1.weight.dtype,device=original.linear1.weight.device)))
        return result

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        if not self.norm_first:
            raise ValueError('The registered residual repair requires Pre-LN')
        value=src+self.attention_scale*self._sa_block(
            self.norm1(src),src_mask,src_key_padding_mask,is_causal=is_causal)
        return value+self._ff_block(self.norm2(value))


class RefinedSurvivalPredictor(SurvivalPredictor):
    def __init__(self, *, identity_coarse_gate=False, fine_attention_scale=None,
                 normalize_fine_kv=False, **architecture):
        super().__init__(**architecture)
        if self.fusion_mode!='cross_gate':
            raise ValueError('Refinement is registered only for the cross-gate R3')
        self.refinement=dict(identity_coarse_gate=bool(identity_coarse_gate),
            fine_attention_scale=fine_attention_scale,normalize_fine_kv=bool(normalize_fine_kv))
        if identity_coarse_gate:
            # Neutral modulation preserves the complete coarse representation.
            nn.init.zeros_(self.coarse_gate[0].weight)
            nn.init.zeros_(self.coarse_gate[0].bias)
            self.coarse_gate[1]=TwiceSigmoid()
        if fine_attention_scale is not None:
            if not 0<float(fine_attention_scale)<=1:
                raise ValueError('Attention residual initial scale must be in (0,1]')
            self.fine_encoder.layers[0]=AttentionScaledEncoderLayer.from_layer(
                self.fine_encoder.layers[0],fine_attention_scale)
        self.fine_kv_norm=nn.LayerNorm(self.dim,eps=1e-5,bias=False) if normalize_fine_kv else nn.Identity()

    def encode_fine(self, queries, padded, mask, positions, slide_indices=None):
        # Same per-patch normalization for both primary and pure-fine auxiliary
        # paths. Queries, observed support and True=padding semantics are intact.
        values=self.fine_kv_norm(padded)
        return super().encode_fine(queries,values,mask,positions,slide_indices)
