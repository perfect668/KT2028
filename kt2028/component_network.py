"""W151 single-component R3 ablations; the original SurvivalPredictor is unchanged."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from .network import GatedPool
from .survival_network import SurvivalPredictor
from .survival import curve_from_logits
from .eagle_source_adapter import load_source, tiled_mase


def lattice(boxes, step=448.):
    """Preserve physical holes; no token-order square padding or slide packing."""
    xy = boxes[:, :2]
    relative = (xy-xy.amin(0))/step
    ij = relative.round().long()
    if not torch.allclose(relative, ij.to(relative), atol=1e-4, rtol=0):
        raise ValueError('Coarse coordinates are off the registered 448um lattice')
    if len(torch.unique(ij, dim=0)) != len(ij):
        raise ValueError('Duplicate coarse coordinates within one slide')
    return ij


class SpatialContext(nn.Module):
    def __init__(self, source, dropout=.25, tile=16):
        super().__init__()
        self.mase = load_source(source).MASE(dropout_p=dropout)
        self.tile = tile

    def forward(self, values, boxes, slide_indices):
        if slide_indices is None:
            slide_indices = torch.zeros(len(values), dtype=torch.long, device=values.device)
        result = torch.zeros_like(values)
        for sid in torch.unique(slide_indices):
            ix = torch.nonzero(slide_indices == sid, as_tuple=True)[0]
            ij = lattice(boxes[ix])
            width, height = (ij.amax(0)+1).tolist()
            canvas = values.new_zeros(height, width, values.shape[-1])
            canvas[ij[:, 1], ij[:, 0]] = values[ix]
            output = tiled_mase(self.mase, canvas[None, None], tile=self.tile)[0, 0]
            result = result.index_copy(0, ix, output[ij[:, 1], ij[:, 0]])
        return result


class SubspacePool(GatedPool):
    """nnMIL feature selection affects score inputs only, never the pooled values."""
    def __init__(self, dim, keep=128):
        super().__init__(dim)
        self.keep = min(keep, dim)
        self.eval_pooled = []
        self.last_weights = None

    def indices(self, x):
        dim = x.shape[-1]
        if self.training:
            return [torch.randperm(dim, device=x.device)[:self.keep]]
        generator = torch.Generator(device=x.device).manual_seed(42)
        perm = torch.randperm(dim, device=x.device, generator=generator)
        stride = max(1, self.keep//4)
        starts = list(range(0, dim-self.keep+1, stride))
        if starts[-1] != dim-self.keep:
            starts.append(dim-self.keep)
        return [perm[i:i+self.keep] for i in starts]

    def attend(self, x, ix):
        subset = x.index_select(-1, ix)
        a = F.linear(subset, self.a.weight.index_select(1, ix), self.a.bias)
        b = F.linear(subset, self.b.weight.index_select(1, ix), self.b.bias)
        return self.score(a.tanh()*b.sigmoid()).T.softmax(-1)

    def forward(self, x):
        weights = torch.cat([self.attend(x, ix) for ix in self.indices(x)], 0)
        pooled = weights@x
        self.last_weights = weights.detach()
        if not self.training:
            self.eval_pooled.append(pooled)
        return pooled[:1], weights[:1]


def mixture_logits(logits):
    """Average complete survival curves, then recover finite conditional hazards.

    Double precision protects the log survival differences near probability 1.
    This is an explicit discrete-survival adaptation, not Cox logmeanexp risk.
    """
    log_survival = F.logsigmoid(-logits.double()).cumsum(-1)
    mixed = torch.logsumexp(log_survival, dim=0)-math.log(len(logits))
    previous = torch.cat([mixed.new_zeros(1), mixed[:-1]])
    log_no_event = (mixed-previous).clamp(max=-torch.finfo(mixed.dtype).eps)
    return torch.log(-torch.expm1(log_no_event))-log_no_event


class ComponentPredictor(SurvivalPredictor):
    def __init__(self, arm, source=None, tile=16, **architecture):
        super().__init__(**architecture)
        if arm not in ('mase', 'subspace'):
            raise ValueError('Only registered single-component variants')
        self.arm = arm
        self._spatial_boxes = None
        if arm == 'mase':
            self.coarse_encoder = SpatialContext(source, architecture.get('dropout', .25), tile)
        else:
            old = self.fine_pool
            self.fine_pool = SubspacePool(self.dim)
            self.fine_pool.load_state_dict(old.state_dict(), strict=True)

    def context_by_slide(self, encoder, values, slide_indices):
        if self.arm == 'mase' and encoder is self.coarse_encoder:
            return encoder(values, self._spatial_boxes, slide_indices)
        return super().context_by_slide(encoder, values, slide_indices)

    def forward(self, coarse, boxes, observed_fine, relative_xy, candidate_patch_count,
                *, return_aux=False, slide_indices=None):
        self._spatial_boxes = boxes
        if self.arm == 'subspace':
            self.fine_pool.eval_pooled = []
        try:
            out = super().forward(coarse, boxes, observed_fine, relative_xy, candidate_patch_count,
                                  return_aux=return_aux, slide_indices=slide_indices)
            if self.arm == 'subspace' and not self.training and observed_fine:
                pooled = self.fine_pool.eval_pooled
                q = out['observation_state']
                logits = torch.stack([self.readout(out['coarse_patient'], z, q) for z in pooled[0]])
                out['logits'] = mixture_logits(logits)
                out['hazards'], out['survival'] = curve_from_logits(out['logits'])
                out['fine_patient'] = pooled[0].mean(0)
                if return_aux:
                    aux = torch.stack([self.readout(out['coarse_patient'].new_zeros(self.dim), z, q) for z in pooled[1]])
                    out['fine_aux_logits'] = mixture_logits(aux)
            return out
        finally:
            self._spatial_boxes = None
            if self.arm == 'subspace':
                self.fine_pool.eval_pooled = []


def make_predictor(arm, cfg, source):
    if arm == 'baseline':
        return SurvivalPredictor(**cfg['architecture'])
    return ComponentPredictor(arm, source, cfg['mase_tile'], **cfg['architecture'])
