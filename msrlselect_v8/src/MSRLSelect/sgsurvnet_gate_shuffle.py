#!/usr/bin/env python3
"""Post-hoc gate and local-feature permutation audit for a frozen SG-SurvNet."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import roc_auc_score,average_precision_score
from train_sgsurvnet import SGDataset,SGSurvNet
from pipeline_common import read_config,seed_all

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True); p.add_argument('--cache-root',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); a=p.parse_args(); c=read_config(a.config); seed_all(int(c['seed'])); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); a.output_dir.mkdir(parents=True,exist_ok=True)
 ds=SGDataset(Path(c['test_manifest']),a.cache_root,'fusion'); ck=torch.load(a.checkpoint,map_location=dev,weights_only=False); m=SGSurvNet(mode='fusion').to(dev); m.load_state_dict(ck['model']); m.eval(); g=[]; ys=[]; ps=[]; gs=[]; ls=[]
 with torch.no_grad():
  for i in range(len(ds)):
   gx,lx,y,case=ds.items[i]; gt=torch.from_numpy(gx[None]).to(dev); lt=torch.from_numpy(lx[None]).to(dev); gp,lp=m.global_proj(gt),m.local_proj(lt); gate=m.gate(torch.cat([gp,lp],1)).item(); logit=m.head(gate*gp+(1-gate)*lp); g.append({'case_id':case,'gate_weight':gate}); ys.append(y); ps.append(float(logit.softmax(1)[0,1])); ls.append(lx)
 perm=np.random.default_rng(2026).permutation(len(ls)); sh=[]
 with torch.no_grad():
  for i in range(len(ds)):
   gx,_,y,case=ds.items[i]; gt=torch.from_numpy(gx[None]).to(dev); lt=torch.from_numpy(ls[perm[i]][None]).to(dev); gp,lp=m.global_proj(gt),m.local_proj(lt); gate=m.gate(torch.cat([gp,lp],1)); sh.append(float(m.head(gate*gp+(1-gate)*lp).softmax(1)[0,1]))
 orig={'auc':roc_auc_score(ys,ps),'auprc':average_precision_score(ys,ps)}; shuffled={'auc':roc_auc_score(ys,sh),'auprc':average_precision_score(ys,sh)}; pd.DataFrame(g).to_csv(a.output_dir/'gate_weights_test.csv',index=False); (a.output_dir/'local_shuffle_audit.json').write_text(json.dumps({'original':orig,'shuffled':shuffled,'delta':{k:shuffled[k]-orig[k] for k in orig},'permutation_seed':2026},indent=2)); print(json.dumps({'original':orig,'shuffled':shuffled},indent=2))
if __name__=='__main__': main()
