"""P1 staged objectives and freeze boundaries; no implicit experiment launcher."""
import torch
from torch.nn import functional as F


def configure_stage(model, updater, stage):
    if stage not in ('full', 'align', 'partial'):
        raise ValueError('Unknown predictor stage')
    model.eval()
    updater.eval()
    for parameter in (*model.parameters(), *updater.parameters()):
        parameter.requires_grad_(False)
    if stage == 'full':
        modules = (model,)
    elif stage == 'align':
        modules = (model.coarse_projection, updater)
    else:
        modules = (model.context, model.patient_pool, model.head)
    for module in modules:
        module.train()
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return [p for p in (*model.parameters(), *updater.parameters()) if p.requires_grad]


def stage_objective(model, updater, inputs, target, stage, *, max_pairs=128, generator=None):
    if stage not in ('full', 'align', 'partial'):
        raise ValueError('Unknown predictor stage')
    if stage != 'align':
        estimates = {}
        if stage == 'partial' and inputs['observed_fine']:
            with torch.no_grad():
                z = model.coarse_projection(inputs['coarse'])
                observed = torch.zeros(len(z), dtype=torch.bool, device=z.device)
                for i, bag in inputs['observed_fine'].items():
                    observed[i] = True
                    estimates = updater.propagate(z, i, model.encode_fine(bag), observed, estimates)
        result = model(**inputs, estimates=estimates)
        return F.binary_cross_entropy_with_logits(result['logit'], target), {'tsu_pairs': 0}
    if not isinstance(max_pairs, int) or max_pairs <= 0:
        raise ValueError('A positive pair cap is required')
    ids = list(inputs['observed_fine'])
    if not ids:
        raise ValueError('Alignment requires real Train fine supervision')
    with torch.no_grad():
        fine = torch.stack([model.encode_fine(inputs['observed_fine'][i]) for i in ids])
    z = model.coarse_projection(inputs['coarse'])[ids]
    alignment = F.mse_loss(z, fine)
    with torch.no_grad():
        similarity = F.normalize(z, dim=-1) @ F.normalize(z, dim=-1).T
        mask = similarity >= updater.threshold
        mask.fill_diagonal_(False)
        pairs = mask.nonzero()
        # CPU generator has deterministic semantics independent of device.
        if len(pairs) > max_pairs:
            selection = torch.randperm(len(pairs), generator=generator)[:max_pairs].to(pairs.device)
            pairs = pairs[selection]
    if len(pairs):
        i, j = pairs.unbind(1)
        # TSU optimization does not change coarse alignment through this loss.
        estimates = updater(z[i].detach(), z[j].detach(), fine[j])
        tsu_loss = F.mse_loss(estimates, fine[i])
    else:
        tsu_loss = alignment.new_zeros(())
    return alignment + tsu_loss, {'tsu_pairs': len(pairs)}
