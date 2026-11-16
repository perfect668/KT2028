"""W145: pristine EAGLE modules, exact tiled MASE and explicit survival adapter.

The author's classifier/attention weights stay shared. Five outputs now denote
annual hazard logits, not cancer subtypes. This is a controlled adaptation, not
the paper's unreleased Cox experiment pipeline.
"""
import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .survival import discrete_nll, curve_from_logits


def load_source(folder):
    path = Path(folder) / 'Eagle_net_model.py'
    spec = importlib.util.spec_from_file_location('_kt2028_eagle_official', path)
    module = importlib.util.module_from_spec(spec)
    # The public module seeds Python/NumPy/Torch during import; preserve caller RNG.
    import random
    py, npstate = random.getstate(), np.random.get_state()
    with torch.random.fork_rng():
        try:
            spec.loader.exec_module(module)
        finally:
            random.setstate(py)
            np.random.set_state(npstate)
    return module


def grid_coordinates(boxes, step=112.):
    """Exact lattice from audited physical top-left coordinates, never row order."""
    xy = np.asarray(boxes, dtype=np.float64)[:, :2]
    relative = (xy - xy.min(0)) / step
    ij = np.rint(relative).astype(np.int64)
    if not np.allclose(relative, ij, atol=1e-6, rtol=0):
        raise ValueError('Feature coordinates are off the nominal fine lattice')
    if len(np.unique(ij, axis=0)) != len(ij):
        raise ValueError('Duplicate feature grid position')
    return ij


def pack_grids(parts):
    """Greedy quarter-rotation rectangle packing; no spatial rescaling.

    The public repository supplies prepacked CSVs, not the patient-packing
    implementation. This deterministic coordinate-only adapter follows the
    paper's packing principle and records the resulting geometry separately.
    """
    placements = []
    height = width = 0
    for xy, values in parts:
        xy = np.asarray(xy, dtype=np.int64)
        if len(xy) != len(values) or len(np.unique(xy, axis=0)) != len(xy):
            raise ValueError('Nonunique or misaligned patch positions')
        w, h = xy.max(0) + 1
        candidates = []
        for k in range(4):
            if k == 0:
                pos, rw, rh = xy.copy(), w, h
            elif k == 1:
                pos, rw, rh = np.c_[h-1-xy[:, 1], xy[:, 0]], h, w
            elif k == 2:
                pos, rw, rh = np.c_[w-1-xy[:, 0], h-1-xy[:, 1]], w, h
            else:
                pos, rw, rh = np.c_[xy[:, 1], w-1-xy[:, 0]], h, w
            for dx, dy in [(width, 0), (0, height)]:
                nw, nh = max(width, dx+rw), max(height, dy+rh)
                candidates.append(((nw*nh, max(nw, nh), k, dx, dy), pos, nw, nh))
        key, pos, width, height = min(candidates, key=lambda c: c[0])
        placements.append((pos + np.array(key[3:5]), values))
    cube = torch.zeros(height, width, parts[0][1].shape[1], dtype=torch.float32)
    mask = torch.zeros(height, width, dtype=torch.bool)
    for xy, values in placements:
        x, y = torch.from_numpy(xy[:, 0]), torch.from_numpy(xy[:, 1])
        if mask[y, x].any():
            raise ValueError('Slide packing collision')
        cube[y, x] = torch.as_tensor(values)
        mask[y, x] = True
    if int(mask.sum()) != sum(len(v) for _, v in parts):
        raise ValueError('Lost feature during packing')
    return cube, mask


def pool_canvas(cube, mask, maximum=50000):
    """Official 2x2 feature-average/mask-max pooling, with edge preservation.

    Unlike the public training-only crop, use the same deterministic rule for
    all partitions and retain odd boundary cells. No observed feature is sampled
    away. Pooled features are not relabelled as a newly extracted image scale.
    """
    levels = 0
    while mask.numel() > maximum:
        cube = F.avg_pool2d(cube.permute(2, 0, 1)[None], 2, 2,
                            ceil_mode=True, count_include_pad=False)[0].permute(1, 2, 0).contiguous()
        mask = F.max_pool2d(mask.float()[None, None], 2, 2, ceil_mode=True)[0, 0].bool()
        levels += 1
    return cube, mask, levels


def tiled_mase(module, x, tile=32, use_checkpoint=True):
    """Same learned official operations, halo=3; whole-canvas channel dropout.

    Avoid only official gc/empty_cache side effects. Independent tile dropout
    would change the model: draw each branch's channel mask once and reuse it.
    """
    if x.shape[:2] != (1, 1):
        raise ValueError('EAGLE canvas expects batch/channel = 1')
    masks = []
    for channels in [8, 8, 8, 8, 1]:
        ones = x.new_ones((1, channels, 1, 1, 1))
        masks.append(F.dropout3d(ones, p=module.dropout.p, training=module.training))

    def block(a, *drop):
        outputs = [F.relu(conv(a))*m for conv, m in zip(
            [module.conv1x1, module.conv3x3, module.conv5x5, module.conv7x7], drop[:4])]
        return F.relu(module.conv_reduce(torch.cat(outputs, dim=1)))*drop[4] + a

    rows = []
    height, width = x.shape[2:4]
    for y in range(0, height, tile):
        columns = []
        for z in range(0, width, tile):
            y0, y1 = max(0, y-3), min(height, y+tile+3)
            z0, z1 = max(0, z-3), min(width, z+tile+3)
            a = x[:, :, y0:y1, z0:z1, :]
            b = checkpoint(block, a, *masks, use_reentrant=False) if use_checkpoint and torch.is_grad_enabled() else block(a, *masks)
            columns.append(b[:, :, y-y0:y-y0+min(tile, height-y), z-z0:z-z0+min(tile, width-z), :])
        rows.append(torch.cat(columns, dim=3))
    return torch.cat(rows, dim=2)


class SourceEAGLE(nn.Module):
    def __init__(self, source, *, feature_dim=1024, dim=512, intervals=5, topk=10, tile=32):
        super().__init__()
        self.official = load_source(source)
        self.network = self.official.EagleNet(feature_dim, intervals, dim, topk=topk, device='cpu')
        self.tile = tile
        # Public train_model.initialize_weights, without importing its CLI/data stack.
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, cube, mask, *, auxiliary=False):
        if cube.ndim != 3 or cube.shape[:2] != mask.shape or not bool(mask.any()):
            raise ValueError('Invalid/empty EAGLE canvas')
        n = self.network
        embedded = n.patch_embed(cube)
        spatial = tiled_mase(n.mase, embedded[None, None], self.tile)
        x = n.norm_C(spatial.reshape(-1, embedded.shape[-1]))
        local = n.classifier_att(x)
        scores = (local-local.max()).exp().sum(1)
        alpha = scores/scores.sum()
        logits = n.classifier_att((alpha[:, None]*x).sum(0))
        hazards, survival = curve_from_logits(logits)
        result = dict(logits=logits, hazards=hazards, survival=survival,
                      attention=alpha.reshape(mask.shape), background_attention=alpha[~mask.flatten()].sum())
        if auxiliary:
            indices = torch.topk(alpha, min(n.topk, len(alpha)), sorted=False).indices
            rows, cols = self.official.get_neighbors(indices//mask.shape[1], indices % mask.shape[1], *mask.shape)
            neighbors = rows*mask.shape[1]+cols
            result.update(neighbor_logits=local[neighbors], neighbor_attention=alpha[neighbors])
        return result


def survival_objective(output, target, neighborhood_weight=1., background_weight=1.):
    bag = discrete_nll(output['logits'], target)
    local = output['neighbor_logits']
    # Batched equivalent of the existing completed-interval likelihood.
    losses = F.softplus(local[:, :target['completed_intervals']]).sum(1)
    if target['event']:
        losses = losses + F.softplus(-local[:, target['event_bin']])
    neighborhood = (losses*output['neighbor_attention']).sum()
    background = output['background_attention']
    total = bag + neighborhood_weight*neighborhood + background_weight*background
    return total, dict(bag_nll=bag, neighborhood=neighborhood, background=background)
