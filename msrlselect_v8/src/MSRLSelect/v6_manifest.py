#!/usr/bin/env python3
import csv,hashlib
from pathlib import Path
root=Path('/root/autodl-tmp/surgen_hemicore');out=root/'outputs/thesis_predictor_first_v6';dst=out/'metrics/v6_sha256_manifest.csv'
files=[p for p in out.rglob('*') if p.is_file() and p!=dst]
files += [root/'src/MSRLSelect/v6_oof_train.py',root/'src/MSRLSelect/v6_finalize.py']
rows=[]
for p in sorted(set(files)):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 rows.append({'path':str(p.relative_to(root)),'sha256':h.hexdigest(),'bytes':p.stat().st_size})
with dst.open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['path','sha256','bytes']);w.writeheader();w.writerows(rows)
print({'files':len(rows),'manifest':str(dst)})
