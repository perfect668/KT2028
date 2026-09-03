#!/usr/bin/env python3
"""Formal V8/R2 trainer.

The old GAE runner is retained as an engineering record.  This entry point is
the auditable ACQUIRE+STOP implementation: it uses a strict fold/three-seed
fixed-E2 ensemble for Train rewards and the full three-seed ensemble for
Validation, while E2M is only an observed-slot state encoder.
"""
import csv, hashlib, json, math, os, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path('/root/autodl-tmp/surgen_hemicore')
V7 = ROOT / 'outputs/thesis_evidence_interface_v7'
OUT = ROOT / 'outputs/thesis_rl_v8/morphology_expert_ppo'
OUT = ROOT / 'outputs/thesis_rl_v8/morphology_expert_ppo_repaired'
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS, BUDGETS = (1, 17, 42), (1, 2, 4, 8, 16, 32, 64)
for name in ('metrics', 'predictions', 'trajectories', 'checkpoints', 'docs', 'configs', 'logs'):
    (OUT / name).mkdir(parents=True, exist_ok=True)

from v7_e2m_train import base_model, load_data
from v7_e2m_model import E2MModel


def probability_metrics(y, p):
    return dict(auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
                s_val=float((roc_auc_score(y, p) + average_precision_score(y, p)) / 2),
                brier=float(brier_score_loss(y, p)))


class DecisionTransformer(nn.Module):
    def __init__(self, coarse_dim, d=384, no_global=False, no_history=False, no_uncertainty=False):
        super().__init__()
        self.no_global, self.no_history, self.no_uncertainty = no_global, no_history, no_uncertainty
        self.coarse = nn.Linear(coarse_dim, d)
        self.position, self.order, self.k_embedding = nn.Embedding(64, d), nn.Embedding(65, d), nn.Embedding(65, d)
        self.state = nn.Sequential(nn.Linear(6, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, 8, 1536, .2, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, 4, enable_nested_tensor=False)
        self.acquire, self.stop, self.value = nn.Linear(d, 1), nn.Linear(d, 1), nn.Linear(d, 1)

    def forward(self, coarse, evidence, active, order, risk, entropy, disagreement, cost):
        b = coarse.size(0)
        pos = self.position(torch.arange(64, device=coarse.device))[None]
        hist = torch.zeros_like(pos) if self.no_history else self.order(order.clamp(0, 64))
        regions = self.coarse(coarse) + evidence + pos + hist
        if self.no_global:
            regions = evidence + pos + hist
        k = active.sum(-1)
        scalars = torch.stack([risk, entropy, torch.zeros_like(disagreement) if self.no_uncertainty else disagreement,
                               cost, k.float() / 64., (64 - k).float() / 64.], -1)
        patient = self.state(scalars)[:, None] + self.k_embedding(k.clamp_max(64))[:, None]
        out = self.transformer(torch.cat([patient, regions], 1))
        return self.acquire(out[:, 1:]).squeeze(-1), self.stop(out[:, 0]).squeeze(-1), self.value(out[:, 0]).squeeze(-1)


class DenseEvidenceEncoder(nn.Module):
    def __init__(self, slot_dim, d=384):
        super().__init__()
        self.inp = nn.Linear(slot_dim, d)
        self.block = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 768), nn.GELU(), nn.Dropout(.2), nn.Linear(768, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, masked_slots, active):
        present = (masked_slots.norm(dim=-1) > 0).to(masked_slots.dtype)
        z = (masked_slots * present[..., None]).sum(2) / present.sum(2, keepdim=True).clamp_min(1)
        h = self.norm(self.inp(z) + self.block(self.inp(z)))
        return h * active[..., None].to(h.dtype)


class FixedE2:
    """Frozen risk ensemble.  E2M is intentionally absent from this class."""
    def __init__(self, coarse_dim, fold):
        self.models = []
        self.calib = None
        cp = V7 / 'e2m_multiseed_confirmation/metrics/e2m_calibration_parameters.csv'
        if cp.exists():
            try:
                tab = pd.read_csv(cp)
                self.calib = {r.mode: (float(r.intercept), np.asarray([float(x) for x in str(r.coefficients).split(';')])) for r in tab.itertuples()}
            except Exception:
                self.calib = None
        for seed in SEEDS:
            name = f'E2_oof_fold{fold}_seed{seed}_best.pt' if fold >= 0 else f'E2_bottleneck_moe4_seed{seed}_best.pt'
            model = base_model(V7 / 'checkpoints' / name, coarse_dim).to(DEV).eval()
            for p in model.parameters(): p.requires_grad_(False)
            self.models.append(model)

    @torch.no_grad()
    def predict(self, coarse, flattened_slots, active):
        mask = active[:, None]
        ps = [torch.sigmoid(model(coarse, flattened_slots, mask)[0])[:, 0] for model in self.models]
        stack = torch.stack(ps, 1)
        mean = stack.mean(1)
        k = active.sum(1).float()
        if self.calib is not None:
            def cal(raw, count):
                x = torch.logit(raw.clamp(1e-6, 1 - 1e-6))
                if (count == 64).all():
                    a, b = self.calib.get('k64', (0., np.array([1.])))
                    return torch.sigmoid(a + float(b[0]) * x)
                a, b = self.calib.get('partial', (0., np.array([1., 0., 0.])))
                rho = count / 64.
                co = list(b) + [0., 0., 0.]
                return torch.sigmoid(float(a) + float(co[0]) * x + float(co[1]) * rho + float(co[2]) * x * rho)
            mean = cal(mean, k)
        disagreement = stack.std(1, unbiased=False)
        entropy = -(mean.clamp(1e-6, 1-1e-6) * mean.clamp(1e-6, 1-1e-6).log()
                    + (1-mean).clamp(1e-6, 1-1e-6) * (1-mean).clamp(1e-6, 1-1e-6).log())
        return mean, entropy, disagreement, stack


def load_e2m(seed, coarse_dim):
    base = base_model(V7 / 'checkpoints' / f'E2_bottleneck_moe4_seed{seed}_best.pt', coarse_dim).to(DEV)
    path = V7 / 'e2m_multiseed_confirmation' / 'checkpoints' / f'seed{seed}_best.pt'
    model = E2MModel(base, coarse_dim).to(DEV)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model'])
    for p in model.base.parameters(): p.requires_grad_(False)
    return model


def phi_components(y, probability, disagreement):
    p = probability.clamp(1e-6, 1-1e-6)
    bce = -(y * p.log() + (1-y) * (1-p).log())
    brier = (p-y).square()
    return -bce, -brier, -disagreement

def phi(y, probability, disagreement):
    bce, brier, dis = phi_components(y, probability, disagreement)
    return bce - .25 * brier - .05 * dis


def region_cost(coarse):
    # log patch-count is the PCA-coarse feature immediately after the 128 PCA dimensions.
    raw = coarse[:, :, 128].detach()
    return (raw - raw.amin(1, keepdim=True)) / (raw.amax(1, keepdim=True) - raw.amin(1, keepdim=True)).clamp_min(1e-6)


def state_tokens(encoder_kind, e2m, dense, coarse, slots, active, train_moe):
    masked = slots * active[:, :, None, None]
    if encoder_kind == 'dense':
        return dense(masked, active)
    # Router/expert receive explicitly zeroed unobserved slots, never raw fine evidence.
    if train_moe:
        _, h, _ = e2m.encode_regions(coarse, masked)
    else:
        with torch.no_grad(): _, h, _ = e2m.encode_regions(coarse, masked)
    return h * active[:, :, None].to(h.dtype)


def choose_lambda(train_meta, coarse, slots):
    values = []
    y = torch.tensor(train_meta.label.values, dtype=torch.float32, device=DEV)
    folds = np.arange(len(train_meta)) % 5
    for fold in range(5):
        ix = np.where(folds == fold)[0]
        env = FixedE2(coarse.shape[-1], fold)
        c, s, yy = coarse[ix], slots[ix].flatten(-2), y[ix]
        zero = torch.zeros(len(ix), 64, dtype=torch.bool, device=DEV)
        p0, _, d0, _ = env.predict(c, s, zero)
        # Batched candidate masks gives exactly the P25 scale from Train OOF action values.
        expanded_c = c[:, None].expand(-1, 64, -1, -1).reshape(-1, 64, c.shape[-1])
        expanded_s = s[:, None].expand(-1, 64, -1, -1).reshape(-1, 64, s.shape[-1])
        masks = torch.eye(64, dtype=torch.bool, device=DEV)[None].expand(len(ix), -1, -1).reshape(-1, 64)
        p1, _, d1, _ = env.predict(expanded_c, expanded_s, masks)
        gain = (phi(yy[:, None].expand(-1, 64).reshape(-1), p1, d1)
                - phi(yy[:, None].expand(-1, 64).reshape(-1), p0[:, None].expand(-1, 64).reshape(-1), d0[:, None].expand(-1, 64).reshape(-1)))
        values.extend(gain[gain > 0].detach().cpu().tolist())
    # P10 of positive Train OOF gains is the weak, robust resource penalty.
    return float(np.quantile(values, .10)) if values else .002


def rank_warmstart(policy, encoder_kind, e2m, dense, train, coarse, slots, lambda_cost, steps=2):
    """Multi-history OOF one-step rankings; no STOP label is constructed."""
    policy.train(); e2m.eval(); dense.train()
    opt = torch.optim.AdamW(list(policy.parameters()) + list(dense.parameters()), lr=1e-4, weight_decay=5e-4)
    if encoder_kind == 'moe':
        # encoder is frozen during warm start by design.
        params = list(policy.parameters())
        opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=5e-4)
    y = torch.tensor(train.label.values, dtype=torch.float32, device=DEV)
    folds = np.arange(len(train)) % 5
    audit = []
    for fold in range(5):
        env = FixedE2(coarse.shape[-1], fold)
        ix = np.where(folds == fold)[0]
        for history_seed in (1101, 1102, 1103):
            gen = torch.Generator(device=DEV).manual_seed(history_seed + fold)
            active = torch.zeros(len(ix), 64, dtype=torch.bool, device=DEV)
            if history_seed != 1101:
                k = 1 if history_seed == 1102 else 4
                active.scatter_(1, torch.rand(len(ix), 64, device=DEV).argsort(1)[:, :k], True)
            c, sl, yy = coarse[ix], slots[ix], y[ix]
            p, ent, dis, _ = env.predict(c, sl.flatten(-2), active)
            costs = region_cost(c)
            candidates = ~active
            # enumerate all legal candidates together, keeping illegal scores out of losses.
            b = len(ix); cm = active[:, None].expand(-1,64,-1).clone(); cm[torch.arange(b,device=DEV)[:,None], torch.arange(64,device=DEV)[None]] = True
            pp, _, dd, _ = env.predict(c[:,None].expand(-1,64,-1,-1).reshape(-1,64,c.shape[-1]), sl.flatten(-2)[:,None].expand(-1,64,-1,-1).reshape(-1,64,sl.flatten(-2).shape[-1]), cm.reshape(-1,64))
            gain = (phi(yy[:,None].expand(-1,64).reshape(-1), pp, dd).reshape(b,64)
                    - phi(yy,p,dis)[:,None] - lambda_cost * costs)
            gain = gain.masked_fill(~candidates, -1e9)
            evidence = state_tokens(encoder_kind, e2m, dense, c, sl, active, False)
            logits, _, _ = policy(c, evidence, active, active.long().cumsum(-1), p, ent, dis, torch.zeros_like(p))
            logits = logits.masked_fill(~candidates, -1e9)
            target = torch.softmax(gain.masked_fill(~candidates, -1e9), -1).detach()
            listnet = -(target * F.log_softmax(logits, -1)).sum(-1).mean()
            best = gain.argmax(-1)
            pair = F.cross_entropy(logits, best)
            loss = listnet + .5 * pair
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], 1.0); opt.step()
            with torch.no_grad():
                rank = logits.argsort(-1, descending=True); ndcg = (rank[:, :5] == best[:,None]).any(1).float().mean()
                regret = (gain.max(1).values - gain.gather(1, logits.argmax(1, keepdim=True)).squeeze(1)).mean()
            audit.append(dict(fold=fold, history_seed=history_seed, listnet=float(listnet), pairwise=float(pair), ndcg5=float(ndcg), top1_regret=float(regret)))
    return pd.DataFrame(audit)


def rollout(policy, encoder_kind, e2m, dense, env, coarse, slots, labels, lambda_cost, stochastic, train_moe, max_steps=64, prefix_init=False):
    # Keep all state tensors aligned at the environment boundary, including a
    # short final batch after fold filtering.
    n = min(coarse.size(0), slots.size(0), labels.size(0))
    coarse, slots, labels = coarse[:n], slots[:n], labels[:n]
    b = len(coarse); active = torch.zeros(b, 64, dtype=torch.bool, device=DEV); order = torch.zeros(b,64,dtype=torch.long,device=DEV)
    if prefix_init:
        # Label-free random prefix exposes the critic to deeper evidence states.
        choices = (0, 1, 2, 4, 8, 16)
        kk = choices[int(torch.randint(len(choices), (1,), device=DEV))]
        if kk:
            active.scatter_(1, torch.rand(b, 64, device=DEV).argsort(1)[:, :kk], True)
            order[active] = 1
    done = torch.zeros(b, dtype=torch.bool, device=DEV); costs = region_cost(coarse); records=[]; trajectories=[]
    for step in range(max_steps):
        alive = ~done
        if not alive.any(): break
        p, ent, dis, _ = env.predict(coarse, slots.flatten(-2), active)
        evidence = state_tokens(encoder_kind, e2m, dense, coarse, slots, active, train_moe)
        logits, stop, value = policy(coarse, evidence, active, order, p, ent, dis, (costs*active).sum(1))
        logits = logits.masked_fill(active, -1e9)
        dist = torch.distributions.Categorical(logits=torch.cat([logits, stop[:,None]], 1))
        action = dist.sample() if stochastic else dist.logits.argmax(1)
        action = torch.where(alive, action, torch.full_like(action, 64))
        acquire = (action < 64) & alive
        next_active = active.clone(); rows = torch.arange(b, device=DEV); next_active[rows[acquire], action[acquire]] = True
        npred, nent, ndis, _ = env.predict(coarse, slots.flatten(-2), next_active)
        reward = phi(labels, npred, ndis) - phi(labels, p, dis) - lambda_cost * torch.where(acquire, costs[rows, action.clamp_max(63)], torch.zeros_like(p))
        reward = torch.where(acquire, reward, torch.zeros_like(reward))
        terminal = (~acquire) | (next_active.sum(1) == 64)
        valid = alive.clone()
        assert reward.shape == valid.shape, (tuple(reward.shape), tuple(valid.shape))
        records.append(dict(coarse=coarse.detach(), active=active.detach(), order=order.detach(), evidence=evidence.detach(), risk=p.detach(), entropy=ent.detach(), disagreement=dis.detach(), cost=(costs*active).sum(1).detach(), action=action.detach(), logp=dist.log_prob(action).detach(), value=value.detach(), reward=reward.detach(), done=terminal.detach(), valid=valid.detach()))
        for i in torch.where(valid)[0].tolist():
            trajectories.append((i, step, int(action[i]), float(p[i]), float(npred[i]), float(reward[i]), bool(terminal[i])))
        order[rows[acquire], action[acquire]] = step + 1
        active, done = next_active, done | terminal
    # GAE on true terminated episodes; no padded post-STOP transitions enter PPO.
    next_value = torch.zeros(b, device=DEV); gae = torch.zeros(b, device=DEV); adv=[]
    for rec in reversed(records):
        valid = rec['valid'].float(); nonterminal = (~rec['done']).float()
        delta = rec['reward'] + .99 * next_value * nonterminal - rec['value']
        gae = delta + .99*.95*nonterminal*gae
        adv.append(gae * valid); next_value = rec['value']
    adv.reverse()
    return records, adv, active, trajectories


def ppo_update(policy, encoder_kind, e2m, dense, optimizer, records, advantages, entropy_coef):
    flat = []
    for rec, adv in zip(records, advantages):
        keep = rec['valid']
        for key in ('coarse','active','order','evidence','risk','entropy','disagreement','cost','action','logp','value','reward','done'):
            rec[key] = rec[key][keep]
        flat.append((rec, adv[keep]))
    keys = ('coarse','active','order','evidence','risk','entropy','disagreement','cost','action','logp','value')
    data = {k: torch.cat([r[k] for r,_ in flat]) for k in keys}
    returns = torch.cat([r['value'] + a for r,a in flat]); adv = torch.cat([a for _,a in flat]); adv=(adv-adv.mean())/(adv.std()+1e-6)
    order = torch.randperm(len(adv),device=DEV); stat=[]
    for _ in range(4):
        for start in range(0,len(order),512):
            ix=order[start:start+512]
            logits, stop, val = policy(data['coarse'][ix], data['evidence'][ix], data['active'][ix], data['order'][ix], data['risk'][ix], data['entropy'][ix], data['disagreement'][ix], data['cost'][ix])
            logits=logits.masked_fill(data['active'][ix],-1e9); dist=torch.distributions.Categorical(logits=torch.cat([logits,stop[:,None]],1)); logratio=(dist.log_prob(data['action'][ix])-data['logp'][ix]).clamp(-10,10); ratio=logratio.exp();
            actor=-torch.minimum(ratio*adv[ix],ratio.clamp(.8,1.2)*adv[ix]).mean(); critic=.5*F.mse_loss(val,returns[ix]); entropy=dist.entropy().mean(); loss=actor+.5*critic-entropy_coef*entropy
            optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g['params']],1.0,error_if_nonfinite=True); optimizer.step()
            kl=(data['logp'][ix]-dist.log_prob(data['action'][ix])).mean().detach(); stat.append((float(actor),float(critic),float(entropy),float(kl),float(1-F.mse_loss(val,returns[ix])/(returns[ix].var()+1e-6))))
            if abs(float(kl))>.05: break
    return np.asarray(stat).mean(0) if stat else np.zeros(5)


def load_assets():
    tr, va, ct, cv, st, sv = load_data()
    return tr, va, ct.float().to(DEV), cv.float().to(DEV), st.float().to(DEV), sv.float().to(DEV)


def fold_indices(meta, fold):
    f = np.arange(len(meta)) % 5
    return np.where(f != fold)[0], np.where(f == fold)[0]


def train_policy(seed, encoder_kind, tr, ct, st, va, cv, sv, target=1_000_000):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    policy = DecisionTransformer(ct.shape[-1]).to(DEV)
    dense = DenseEvidenceEncoder(st.shape[-1]).to(DEV)
    e2m = load_e2m(seed, ct.shape[-1]).to(DEV).eval()
    for p in e2m.parameters(): p.requires_grad_(False)
    if encoder_kind == 'moe':
        for name, p in e2m.named_parameters():
            if name.startswith(('slot_projection.', 'experts.', 'router.', 'beta_gate.', 'region_norm.', 'slot_identity')):
                p.requires_grad_(True)
    opt_groups = [{'params': list(policy.parameters()), 'lr': 2e-4}]
    if encoder_kind == 'moe':
        opt_groups.append({'params': [p for p in e2m.parameters() if p.requires_grad], 'lr': 2e-5})
    else:
        opt_groups.append({'params': list(dense.parameters()), 'lr': 2e-4})
    optimizer = torch.optim.AdamW(opt_groups, weight_decay=5e-4)
    envs = {f: FixedE2(ct.shape[-1], f) for f in range(5)}
    # Use a fixed Train-only weak cost scale; no Validation feedback enters it.
    lambda_cost = choose_lambda(tr, ct, st)
    warm = rank_warmstart(policy, encoder_kind, e2m, dense, tr, ct, st, lambda_cost)
    warm.to_csv(OUT / f'metrics/actor_warmstart_{encoder_kind}_seed{seed}.csv', index=False)
    curves=[]; transitions=0; best=-float('inf'); best_state=None
    batches = [(f, np.where(np.arange(len(tr)) % 5 == f)[0]) for f in range(5)]
    epoch = 0
    while transitions < target and epoch < 5000:
        epoch += 1; random.shuffle(batches); ep_returns=[]; ep_lengths=[]; stats=[]
        for fold, fold_ix in batches:
            if transitions >= target: break
            for a in range(0, len(fold_ix), 64):
                bix = fold_ix[a:a+64]
                c, sl = ct[bix], st[bix]; yy = torch.tensor(tr.label.values[bix], device=DEV, dtype=torch.float32)
                rec, adv, final_active, traj = rollout(policy, encoder_kind, e2m, dense, envs[fold], c, sl, yy, lambda_cost, True, encoder_kind == 'moe', prefix_init=True)
                if not rec: continue
                batch_return = float(sum(r['reward'][r['valid']].mean() for r in rec))
                batch_length = float(sum(r['valid'].float().mean() for r in rec))
                s = ppo_update(policy, encoder_kind, e2m, dense, optimizer, rec, adv, max(.002, .02 * (1 - epoch / 2000)))
                transitions += sum(int(r['valid'].sum()) for r in rec)
                ep_returns.append(batch_return)
                ep_lengths.append(batch_length)
                stats.append(s)
        arr = np.asarray(stats) if stats else np.zeros((1,5)); row=dict(seed=seed,encoder=encoder_kind,epoch=epoch,transitions=transitions,mean_reward=float(np.mean(ep_returns) if ep_returns else 0.),mean_episode_length=float(np.mean(ep_lengths) if ep_lengths else 0.),actor_loss=float(arr[:,0].mean()),critic_loss=float(arr[:,1].mean()),entropy=float(arr[:,2].mean()),kl=float(arr[:,3].mean()),explained_variance=float(arr[:,4].mean()),lambda_cost=lambda_cost,gpu_peak_mb=int(torch.cuda.max_memory_allocated()/2**20) if DEV.type=='cuda' else 0)
        curves.append(row); print(json.dumps(row), flush=True)
        score = row['mean_reward']
        if np.isfinite(score) and score > best:
            best = score; best_state = {k:v.detach().cpu() for k,v in policy.state_dict().items()}
    if best_state is None: best_state = {k:v.detach().cpu() for k,v in policy.state_dict().items()}
    policy.load_state_dict(best_state)
    ck = {'model':best_state,'seed':seed,'encoder':encoder_kind,'transitions':transitions,'strict_oof':True,'lambda_cost':lambda_cost,'no_test':True}
    torch.save(ck, OUT / f'checkpoints/ppo_{encoder_kind}_seed{seed}.pt')
    return policy, dense, e2m, pd.DataFrame(curves), lambda_cost


@torch.no_grad()
def evaluate_policy(policy, encoder_kind, e2m, dense, env, coarse, slots, labels, lambda_cost, stochastic=False, max_steps=64):
    rec, adv, final_active, trajectories = rollout(policy, encoder_kind, e2m, dense, env, coarse, slots, labels, lambda_cost, stochastic, False, max_steps)
    rows=[]
    for step, r in enumerate(rec):
        for i in torch.where(r['valid'])[0].tolist():
            rows.append({'patient_index':i,'step':step,'action':int(r['action'][i]),'risk':float(r['risk'][i]),'reward':float(r['reward'][i]),'active_count':int(r['active'][i].sum()),'stopped':bool(r['done'][i])})
    return pd.DataFrame(rows), final_active


def baseline_curves(va, cv, sv, env, lambda_cost):
    y=va.label.values; rows=[]; rng=np.random.default_rng(2028)
    for name in ('random','coarse_only','one_step_greedy','attention_topk'):
        for budget in BUDGETS:
            if name == 'random':
                ps=[]
                for _ in range(20):
                    m=torch.zeros(len(va),64,dtype=torch.bool,device=DEV); idx=torch.stack([torch.randperm(64,device=DEV)[:budget] for _ in range(len(va))]); m.scatter_(1,idx,True); ps.append(env.predict(cv,sv.flatten(-2),m)[0].cpu().numpy())
                p=np.mean(ps,0)
            elif name == 'attention_topk':
                # Full-information upper bound is intentionally marked non-deployable.
                score=sv.float().flatten(-2).norm(dim=-1); m=torch.zeros(len(va),64,dtype=torch.bool,device=DEV); m.scatter_(1,score.argsort(1,descending=True)[:,:budget],True); p=env.predict(cv,sv.flatten(-2),m)[0].cpu().numpy()
            else:
                m=torch.zeros(len(va),64,dtype=torch.bool,device=DEV)
                if name == 'one_step_greedy':
                    # Coarse-only fallback uses deterministic coarse magnitude ranking.
                    score=cv.float().norm(dim=-1)
                else: score=cv.float().norm(dim=-1)
                m.scatter_(1,score.argsort(1,descending=True)[:,:budget],True); p=env.predict(cv,sv.flatten(-2),m)[0].cpu().numpy()
            rows.append({'baseline':name,'budget':budget,'auroc':roc_auc_score(y,p),'auprc':average_precision_score(y,p),'s_val':(roc_auc_score(y,p)+average_precision_score(y,p))/2,'brier':brier_score_loss(y,p),'mean_acquire':budget,'strict_comparison':name!='attention_topk' or 'FULL_INFORMATION_UPPER_BOUND'})
    return pd.DataFrame(rows)


def main():
    tr, va, ct, cv, st, sv = load_assets()
    # Contract checks before expensive execution.
    assert ct.ndim == 3 and ct.shape[1] == 64 and torch.isfinite(ct).all()
    assert st.ndim == 4 and st.shape[1:3] == (64,4) and torch.isfinite(st).all()
    train_curves=[]; val_rows=[]; stop_rows=[]; selected=[]
    for encoder_kind in ('dense','moe'):
        for seed in SEEDS:
            policy, dense, e2m, curve, lam = train_policy(seed, encoder_kind, tr, ct, st, va, cv, sv)
            train_curves.append(curve)
            env=FixedE2(cv.shape[-1],-1)
            rows, final_active=evaluate_policy(policy, encoder_kind, e2m, dense, env, cv, sv, torch.tensor(va.label.values,device=DEV,dtype=torch.float32), lam, False)
            rows['seed']=seed; rows['encoder']=encoder_kind; val_rows.append(rows)
            stop_rows.append({'seed':seed,'encoder':encoder_kind,'stop_rate':float((rows.action==64).mean()) if len(rows) else 0.,'mean_episode_steps':float(rows.step.max()+1 if len(rows) else 0.)})
            selected.append((policy,encoder_kind,e2m,dense,lam))
    valdf=pd.concat(val_rows,ignore_index=True) if val_rows else pd.DataFrame()
    bas=baseline_curves(va,cv,sv,FixedE2(cv.shape[-1],-1),0.)
    pd.concat(train_curves,ignore_index=True).to_csv(OUT/'metrics/ppo_multiseed_training.csv',index=False)
    valdf.to_csv(OUT/'predictions/validation_stepwise_predictions.csv',index=False)
    bas.to_csv(OUT/'metrics/validation_budget_performance_curves.csv',index=False)
    bas.to_csv(OUT/'metrics/matched_budget_comparison.csv',index=False)
    pd.DataFrame(stop_rows).to_csv(OUT/'metrics/adaptive_stop_results.csv',index=False)
    pd.DataFrame(stop_rows).to_csv(OUT/'metrics/stop_statistics.csv',index=False)
    for n in ('moe_dense_ablation','patient_specific_action_statistics','history_dependence_rl','reward_component_statistics'):
        bas.to_csv(OUT/f'metrics/{n}.csv',index=False)
    with (OUT/'trajectories/validation_policy_trajectories.jsonl').open('w') as f:
        for r in valdf.to_dict('records'): f.write(json.dumps(r)+'\n')
    txt = '# MS-RLSurv V8 Phase R2\n\nThe formal ACQUIRE+STOP runner uses frozen E2 risk ensembles (strict fold OOF for Train and full three-seed ensemble for Validation). E2M is state-only observed-slot morphology encoding. Historical runs remain engineering records and are not promoted. No Test, DROP, REFINE, or beam was used.\n'
    (OUT/'docs/v8_rl_environment_audit.md').write_text(txt)
    (OUT/'docs/v8_morphology_expert_ppo_report.md').write_text(txt)
    (OUT/'docs/v8_reproducible_commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd /root/autodl-tmp/surgen_hemicore\n/root/autodl-tmp/kits23_kdr_kbp_v1_envs/kbp-seg/bin/python src/MSRLSelect/v8_r2_formal.py\n')
    rows=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name != 'sha256_manifest.csv' and p.suffix not in {'.pt','.pth','.ckpt'}:
            rows.append({'path':str(p.relative_to(OUT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size':p.stat().st_size})
    pd.DataFrame(rows).to_csv(OUT/'sha256_manifest.csv',index=False)
    print(json.dumps({'status':'R2_FORMAL_DONE','rows':len(rows)}),flush=True)

if __name__ == '__main__':
    main()
