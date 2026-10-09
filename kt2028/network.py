"""P1 interface implementation, not an upstream HAFED reproduction or fitted model."""
import torch
from torch import nn
from torch.nn import functional as F


class GatedPool(nn.Module):
    def __init__(self, dim, branches=1):
        super().__init__()
        self.a = nn.Linear(dim, 128)
        self.b = nn.Linear(dim, 128)
        self.score = nn.Linear(128, branches)

    def forward(self, x):
        if x.ndim != 2 or len(x) == 0:
            raise ValueError('Attention requires a nonempty variable-length bag')
        weights = self.score(torch.tanh(self.a(x)) * torch.sigmoid(self.b(x)))
        weights = weights.T.softmax(dim=-1)
        return weights @ x, weights


class EvidencePredictor(nn.Module):
    """Only acquired fine bags may be passed to forward; no label argument."""
    def __init__(self, dim=256, branches=4):
        super().__init__()
        self.dim = dim
        self.fine_projection = nn.Sequential(nn.Linear(1024, dim), nn.GELU())
        self.local_pool = GatedPool(dim)
        self.coarse_projection = nn.Sequential(nn.Linear(512, dim), nn.GELU())
        # Geometry, true-observation flag, estimated flag, global evidence ratio.
        self.context = nn.Linear(7, dim)
        self.patient_pool = GatedPool(dim, branches)
        self.head = nn.Linear(dim * branches, 1)

    def encode_fine(self, bag):
        if bag.ndim != 2 or bag.shape[1] != 1024 or not torch.isfinite(bag).all():
            raise ValueError('Invalid acquired UNI bag')
        return self.local_pool(self.fine_projection(bag))[0].squeeze(0)

    def forward(self, coarse, boxes, observed_fine, estimates=None):
        if coarse.ndim != 2 or coarse.shape[1] != 512 or len(coarse) == 0:
            raise ValueError('Invalid coarse bag')
        if boxes.shape != (len(coarse), 4) or not torch.isfinite(boxes).all():
            raise ValueError('Invalid geometry')
        if not torch.isfinite(coarse).all():
            raise ValueError('Nonfinite coarse')
        estimates = {} if estimates is None else estimates
        r = len(coarse)
        if any(not isinstance(i, int) or not 0 <= i < r for i in (*observed_fine, *estimates)):
            raise ValueError('Invalid evidence region index')
        z = self.coarse_projection(coarse)
        values = list(z.unbind())
        observed = coarse.new_zeros(r)
        estimated = coarse.new_zeros(r)
        for i, value in estimates.items():
            if value.shape != (self.dim,) or not torch.isfinite(value).all():
                raise ValueError('Invalid TSU estimate')
            if i not in observed_fine:
                values[i] = value
                estimated[i] = 1
        for i, bag in observed_fine.items():
            values[i] = self.encode_fine(bag)
            observed[i] = 1
        evidence = torch.stack(values)
        # All-coarse-visible geometry normalization; no fine or outcome required.
        span = (boxes[:, 2:].amax(0) - boxes[:, :2].amin(0)).clamp_min(1)
        origin = boxes[:, :2].amin(0).repeat(2)
        geometry = (boxes - origin) / span.repeat(2)
        ratio = observed.mean().expand(r)
        context = torch.cat((geometry, observed[:, None], estimated[:, None], ratio[:, None]), 1)
        tokens = evidence + self.context(context)
        pooled, attention = self.patient_pool(tokens)
        logit = self.head(pooled.flatten()).squeeze(-1)
        return {'logit': logit, 'probability': logit.sigmoid(), 'tokens': tokens,
                'patient': pooled.flatten(), 'attention': attention, 'coarse_latent': z,
                'evidence': evidence, 'observed': observed.bool(), 'estimated': estimated.bool()}


class TargetedStateUpdater(nn.Module):
    def __init__(self, dim=256, threshold=0.9):
        super().__init__()
        if not -1 <= threshold <= 1:
            raise ValueError('Invalid cosine threshold')
        self.threshold = threshold
        self.mlp = nn.Sequential(nn.Linear(3 * dim, 2 * dim), nn.GELU(),
                                 nn.Linear(2 * dim, dim), nn.LayerNorm(dim))

    def forward(self, z_i, z_j, fine_j):
        return self.mlp(torch.cat((z_i, z_j, fine_j), dim=-1))

    def propagate(self, z, acquired_id, fine_j, observed, previous=None):
        estimates = dict(previous or {})
        similarity = F.cosine_similarity(z, z[acquired_id].unsqueeze(0), dim=-1)
        targets = ((similarity >= self.threshold) & ~observed).nonzero().flatten()
        if len(targets):
            predicted = self(z[targets], z[acquired_id].expand(len(targets), -1),
                             fine_j.expand(len(targets), -1))
            estimates.update({int(i): value for i, value in zip(targets, predicted)})
        return {i: value for i, value in estimates.items() if not observed[i]}


class SharedActorCritic(nn.Module):
    """Variable-R scorer. SB3 padded rollout adapter remains a separate task."""
    def __init__(self, dim=256, branches=4):
        super().__init__()
        context_dim = dim * branches + 4  # risk, spent, remaining, lambda
        self.region_score = nn.Sequential(nn.Linear(dim + context_dim + 1, 128),
                                          nn.Tanh(), nn.Linear(128, 1))
        self.stop_score = nn.Linear(context_dim, 1)
        self.critic = nn.Sequential(nn.Linear(context_dim, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, prediction, costs, spent, remaining, cost_preference, legal):
        context = torch.cat((prediction['patient'], torch.stack((prediction['probability'],
                             costs.new_tensor(spent), costs.new_tensor(remaining),
                             costs.new_tensor(cost_preference)))))
        inputs = torch.cat((prediction['tokens'], context.expand(len(costs), -1), costs[:, None]), 1)
        logits = torch.cat((self.region_score(inputs).flatten(), self.stop_score(context)))
        if legal.shape != logits.shape or not legal[-1] or not legal.any():
            raise ValueError('STOP must remain legal')
        logits = logits.masked_fill(~legal, -torch.inf)
        return torch.distributions.Categorical(logits=logits), self.critic(context).squeeze(-1)
