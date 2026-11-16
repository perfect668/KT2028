"""Small data/objective bridge to pinned nnMIL, PANTHER and OTSurv sources."""
import importlib.util
import sys
import types
from pathlib import Path
import numpy as np
import torch


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_package(root):
    # Do not run unrelated model factories importing every optional architecture.
    # Relative imports still resolve to the byte-identical official modules.
    root = Path(root)
    sys.path.insert(0, str(root/'src'))
    module = types.ModuleType('mil_models')
    module.__path__ = [str(root/'src'/'mil_models')]
    sys.modules['mil_models'] = module


def make_model(method, official, prototype=None):
    official = Path(official)
    if method == 'nnMIL':
        cls = load_file('w147_simple_mil', official/'nnMIL/network_architecture/models/simple_mil.py').SimpleMIL
        return cls(input_dim=1024, hidden_dim=256, pred_num=1, dropout=True)
    source_package(official/method)
    if method == 'OTSurv':
        from mil_models.model_otsurv import OTSurv
        return OTSurv(num_classes=1, patch_dim=1024, hidden_dim=256, num_prototypes=16, dropout_rate=.25)
    from mil_models.model_linear import LinearEmb
    from types import SimpleNamespace
    return LinearEmb(SimpleNamespace(in_dim=32784, n_classes=1), 'survival')


def make_panther(official, prototype):
    source_package(Path(official)/'PANTHER')
    from mil_models.PANTHER.layers import PANTHERBase
    return PANTHERBase(1024, p=16, L=1, tau=.001, out='allcat', ot_eps=.1,
                       load_proto=True, proto_path=str(prototype), fix_proto=True)


def cox_breslow(risk, times, events):
    """Stable Cox partial NLL/event, including all tied-time risk-set members."""
    risk, times, events = risk.reshape(-1), times.reshape(-1), events.reshape(-1)
    if not (len(risk) == len(times) == len(events)):
        raise ValueError('Mismatched Cox vectors')
    if not events.sum():
        return risk.sum()*0
    denominators = torch.logsumexp(risk[None, :].expand(len(risk), -1).masked_fill(
        times[None, :] < times[:, None], -torch.inf), dim=1)
    return ((denominators-risk)*events).sum()/events.sum()


def labels(rows, device='cpu'):
    return (torch.tensor([r['survival_target']['time_days'] for r in rows], dtype=torch.float32, device=device),
            torch.tensor([r['survival_target']['event'] for r in rows], dtype=torch.float32, device=device))


def baseline_curves(train_risk, train_rows, risk, edges):
    """Train-only Breslow baseline. Shift risks jointly to avoid overflow."""
    time = np.array([r['survival_target']['time_days'] for r in train_rows])
    event = np.array([r['survival_target']['event'] for r in train_rows])
    shift = float(np.max(train_risk))
    event_times = np.unique(time[event == 1])
    increments = np.array([np.sum((time == t)&(event == 1))/np.exp(train_risk[time >= t]-shift).sum()
                           for t in event_times])
    base = np.array([increments[event_times <= t].sum() for t in edges[1:]])
    return np.exp(-np.exp(np.clip(np.array(risk)-shift, -700, 700))[:, None]*base[None, :])


def risk_metrics(risk, rows):
    from sksurv.metrics import concordance_index_censored
    t, e = labels(rows)
    c, concordant, discordant, tied_risk, tied_time = concordance_index_censored(e.numpy().astype(bool), t.numpy(), np.asarray(risk))
    loss = cox_breslow(torch.tensor(risk, dtype=torch.float64), t.double(), e.double())
    return dict(c_index=float(c), cox_partial_nll_per_event=float(loss), patients=len(rows), events=int(e.sum()),
                comparable_pairs=int(concordant+discordant+tied_risk), tied_risk=int(tied_risk), tied_time=int(tied_time))
