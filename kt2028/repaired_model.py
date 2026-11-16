"""Load the verified W159 model from a small manifest; never fit or select."""
import hashlib
import json
from pathlib import Path

import torch

from .refined_survival_network import RefinedSurvivalPredictor


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda:source.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def load_repaired_model(manifest, *, checkpoint=None, device='cpu'):
    """Return (eval predictor, public contract) after exact artifact checks.

    Default checkpoint stays at its registered server path. A path override
    changes only location; its content must have the same registered SHA.
    """
    path=Path(manifest).resolve();doc=json.loads(path.read_text(encoding='utf-8'))
    if doc.get('schema')!='KT2028-verified-r3-repair-1':raise ValueError('Unsupported repaired-model manifest')
    receipt=Path(doc['verification_receipt'])
    if not receipt.is_absolute():receipt=path.parent/receipt
    if digest(receipt)!=doc['verification_receipt_sha256']:raise ValueError('Verification receipt changed')
    proof=json.loads(receipt.read_text())
    if proof.get('status')!='final_verified':raise ValueError('Final model has not been verified')
    if doc.get('verification_variant'):
        verified=proof.get('verified_variants',{}).get(doc['verification_variant'],{})
        if verified.get('status')!='verified':raise ValueError('Requested model variant has not been verified')
        for key in ['stage','architecture','interval_edges_days','feature_protocol','refinement']:
            if verified.get(key)!=doc[key]:raise ValueError('Manifest differs from verified model: '+key)
        expected_reload=verified['fresh_reload']
    else:
        for key,expected in [('retained_stage_replayed',doc['stage']),('selected_spec',doc['refinement']),
                             ('architecture',doc['architecture']),('interval_edges_days',doc['interval_edges_days']),
                             ('feature_protocol',doc['feature_protocol'])]:
            if proof.get(key)!=expected:raise ValueError('Manifest differs from verified model: '+key)
        expected_reload=proof.get('final_fresh_reload',proof['baseline_fresh_reload'])
    if expected_reload['checkpoint_sha256']!=doc['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from the independently reloaded model')
    for name,expected in doc['network_source_sha256'].items():
        local=Path(__file__).parent/name
        if name not in {'repaired_model.py','refined_survival_network.py','survival_network.py',
                        'fusion_network.py','network.py','survival.py'} or digest(local)!=expected:
            raise ValueError('Network source differs from verified artifact: '+name)
    weights=Path(checkpoint) if checkpoint is not None else Path(doc['checkpoint'])
    if not weights.is_absolute():weights=path.parent/weights
    if digest(weights)!=doc['checkpoint_sha256']:raise ValueError('Checkpoint content changed')
    spec={k:v for k,v in doc['refinement'].items() if k!='exposure_likelihood'}
    model=RefinedSurvivalPredictor(**spec,**doc['architecture'])
    # best.pt contains tensors plus primitive metadata; do not deserialize RNG
    # or arbitrary Python objects from an optimizer/last checkpoint.
    state=torch.load(weights,map_location='cpu',weights_only=True)
    model.load_state_dict(state['model'],strict=True)
    model.to(device).eval()
    contract={key:doc[key] for key in ['stage','architecture','refinement','interval_edges_days','feature_protocol','checkpoint_sha256']}
    contract.update({key:doc[key] for key in ['verification_variant','selection_role'] if key in doc})
    return model,contract
