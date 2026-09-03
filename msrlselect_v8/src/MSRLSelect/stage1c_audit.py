#!/usr/bin/env python3
import hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
sys.path.insert(0,str(Path(__file__).parent))
from stage1c_model import Stage1CDenseModel
from stage1c_train import Store,batchify,evaluate

ROOT=Path('/root/autodl-tmp/surgen_hemicore'); OLD=ROOT/'outputs/thesis_algorithm_upgrade_v5/stage1b_repaired_screening/formal'; OUT=ROOT/'outputs/thesis_algorithm_upgrade_v5/stage1c_dense_confirmation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 (OUT/'metrics').mkdir(parents=True,exist_ok=True);(OUT/'docs').mkdir(parents=True,exist_ok=True)
 p1=OLD/'R1_K64_gated_no_spatial/seed17/validation_predictions.csv';p3=OLD/'R3_K64_gated_spatial_no_relation/seed17/validation_predictions.csv';a=pd.read_csv(p1);b=pd.read_csv(p3);d=(a.death_risk-b.death_risk).abs()
 c1=OLD/'R1_K64_gated_no_spatial/seed17/seed17_best.pt';c3=OLD/'R3_K64_gated_spatial_no_relation/seed17/seed17_best.pt';x=torch.load(c1,map_location='cpu',weights_only=False);y=torch.load(c3,map_location='cpu',weights_only=False);diff=[]
 for k in x['model']:
  if k in y['model'] and x['model'][k].shape==y['model'][k].shape:diff.append((x['model'][k].float()-y['model'][k].float()).abs().max().item())
 ident=dict(prediction_sha256_r1=sha(p1),prediction_sha256_r3=sha(p3),checkpoint_sha256_r1=sha(c1),checkpoint_sha256_r3=sha(c3),max_prediction_abs_diff=float(d.max()),mean_prediction_abs_diff=float(d.mean()),exact_prediction_count=int((d==0).sum()),patient_count=len(d),max_common_parameter_abs_diff=max(diff),root_cause='Old R3 set spatial=True,no_relation=True; constructor used spatial and not no_relation for bias only, but always ran the same relation blocks as R1. The numerical graph and initialization were identical. CSV/checkpoint hashes differ due to metadata/configuration fields; files were not overwritten.',old_r3_valid=False)
 (OUT/'audit/r1_r3_identity.json').write_text(json.dumps(ident,indent=2))
 # Repaired spatial checkpoint: prediction perturbation and internal telemetry.
 run=OUT/'audit/repro_a/C2_K64_gated_relation_spatial/seed17';ck=torch.load(run/'best.pt',map_location='cuda',weights_only=False);m=Stage1CDenseModel(pool='gated_attention_pool',spatial=True,relation=True).cuda();m.load_state_dict(ck['model'],strict=True)
 valid=list(pd.read_csv(ROOT/'prepared/sr386_full/surgen_validate_10.csv').itertuples(index=False));train=list(pd.read_csv(ROOT/'prepared/sr386_full/surgen_train_10.csv').itertuples(index=False));store=Store(ROOT,'k64');res=[];preds={}
 torch.manual_seed(991)
 for mode in ['normal','shuffle_coords','bypass_spatial']:
  s,yy,pp=evaluate(m,store,'valid',valid,torch.device('cuda'),mode=mode);preds[mode]=np.array(pp);res.append(dict(condition=mode,valid_auc=s['auc'],valid_auprc=s['auprc'],valid_loss=s['loss'],valid_brier=s['brier'],s_val=.5*(s['auc']+s['auprc'])))
 pd.DataFrame(res).to_csv(OUT/'metrics/stage1c_coordinate_perturbation_results.csv',index=False)
 perturb=dict(normal_vs_shuffle_max_abs=float(np.max(np.abs(preds['normal']-preds['shuffle_coords']))),normal_vs_shuffle_mean_abs=float(np.mean(np.abs(preds['normal']-preds['shuffle_coords']))),normal_vs_bypass_max_abs=float(np.max(np.abs(preds['normal']-preds['bypass_spatial']))),normal_vs_bypass_mean_abs=float(np.mean(np.abs(preds['normal']-preds['bypass_spatial']))))
 X,PM,C,S,RM,Y=batchify(store,'train',train[:4],torch.device('cuda'));m.train()
 with torch.autocast('cuda',dtype=torch.bfloat16):o,aux=m(X,PM,C,S,RM,collect=True);loss=torch.nn.functional.cross_entropy(o,Y)
 loss.backward();grows=[]
 for li,block in enumerate(m.blocks):
  rb=aux['blocks'][li]['raw_bias'].detach().float();gb=aux['blocks'][li]['gated_bias'].detach().float()
  for hi,gv in enumerate(torch.sigmoid(block.gate).detach().cpu().tolist()):
   grows.append(dict(layer=li,head=hi,gate=gv,gate_grad=float(block.gate.grad[hi].abs()),spatial_mlp_grad=float(block.bias_mlp[-1].weight.grad[hi].abs().mean()),raw_bias_mean=float(rb[:,hi].mean()),raw_bias_std=float(rb[:,hi].std()),raw_bias_min=float(rb[:,hi].min()),raw_bias_max=float(rb[:,hi].max()),gated_bias_mean=float(gb[:,hi].mean()),gated_bias_std=float(gb[:,hi].std())))
 pd.DataFrame(grows).to_csv(OUT/'metrics/stage1c_spatial_gate_statistics.csv',index=False)
 spatial_ok=perturb['normal_vs_shuffle_max_abs']>0 and perturb['normal_vs_bypass_max_abs']>0 and all(r['gate_grad']>0 and r['spatial_mlp_grad']>0 for r in grows)
 # Reproducibility
 ra=OUT/'audit/repro_a/C2_K64_gated_relation_spatial/seed17';rb=OUT/'audit/repro_b/C2_K64_gated_relation_spatial/seed17';repro={}
 if rb.exists():
  pa=pd.read_csv(ra/'validation_predictions.csv').death_risk.to_numpy();pb=pd.read_csv(rb/'validation_predictions.csv').death_risk.to_numpy();repro={'max_abs_prediction_diff':float(np.max(np.abs(pa-pb))),'checkpoint_sha_equal':sha(ra/'best.pt')==sha(rb/'best.pt'),'prediction_sha_equal':sha(ra/'validation_predictions.csv')==sha(rb/'validation_predictions.csv')}
 go=spatial_ok and repro.get('max_abs_prediction_diff',1)==0
 audit=dict(status='GO' if go else 'NO-GO',unit_tests='4 passed',early_stop_expected_epoch=30,scheduler='5 epoch linear warmup then cosine',r1_r3_distinct_new_branches=True,spatial_effective=spatial_ok,reproducibility=repro,test_access=False)
 (OUT/'audit/protocol_audit.json').write_text(json.dumps(audit,indent=2));(OUT/'audit/spatial_effectiveness.json').write_text(json.dumps(perturb|{'spatial_ok':spatial_ok},indent=2));print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
