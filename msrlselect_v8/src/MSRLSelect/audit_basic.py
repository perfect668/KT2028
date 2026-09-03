#!/usr/bin/env python3
from pathlib import Path
import argparse,json,pandas as pd
from pipeline_common import read_config

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); c=read_config(a.config)
 dfs={k:pd.read_csv(c[k+'_manifest']) for k in ('train','valid','test')}
 patients={k:set(d.patient_id.astype(str)) for k,d in dfs.items()}; intersections={f'{x}_{y}':len(patients[x]&patients[y]) for x,y in (('train','valid'),('train','test'),('valid','test'))}
 labels={k:{str(int(x)):int(n) for x,n in d.label.value_counts().sort_index().items()} for k,d in dfs.items()}
 all_ids=set().union(*patients.values()); case_ids=set().union(*(set(d.case_id.astype(str)) for d in dfs.values()))
 # Source constants define 5Y_SUR: 1=died within 5y, 0=survived 5y.
 result={'task':c['task'],'label_definition':{'0':'Survived 5y','1':'Died within 5y'},'label_semantics_note':'Confirmed from SurGen-Dataset/reproducibility/datasets/dataset_constants.py LABEL_ENCODINGS for SR386/5Y_SUR.', 'counts':{k:{'rows':len(d),'patients':int(d.patient_id.nunique()),'labels':labels[k]} for k,d in dfs.items()},'intersections':intersections,'total_patients':len(all_ids),'total_cases':len(case_ids),'all_unique':len(all_ids)==423 and len(case_ids)==423 and len(all_ids)==len(case_ids)}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
