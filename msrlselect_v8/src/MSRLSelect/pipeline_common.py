from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np, pandas as pd, torch

def seed_all(seed=1):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def read_config(path):
    try:
        import yaml
        return yaml.safe_load(Path(path).read_text())
    except ImportError:
        out={}
        for line in Path(path).read_text().splitlines():
            if line.strip() and not line.lstrip().startswith('#'):
                k,v=line.split(':',1); v=v.strip()
                if v.lower() in ('true','false'): v=v.lower()=='true'
                else:
                    try: v=float(v) if '.' in v else int(v)
                    except ValueError: pass
                out[k.strip()]=v
        return out

def load_case(row):
    with np.load(row.features_filepath) as z:
        f=z['img_features'].astype(np.float32,copy=True); c=z['coords'].astype(np.float32,copy=True)
    clusters=json.loads(Path(row.clusters_json_filepath).read_text())
    return f,c,clusters

def cluster_patch_indices(clusters, n):
    out=[]
    for cluster in clusters:
        a=np.asarray(cluster,dtype=np.int64); a=a[(a>=0)&(a<n)]
        out.append(np.unique(a))
    return out

def metric_dict(y,p):
    from sklearn.metrics import roc_auc_score,average_precision_score,accuracy_score,f1_score,confusion_matrix
    y=np.asarray(y); p=np.asarray(p); pred=(p>=.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else float('nan'),
            'auprc':float(average_precision_score(y,p)), 'accuracy':float(accuracy_score(y,pred)),
            'f1':float(f1_score(y,pred,zero_division=0)), 'sensitivity':float(tp/(tp+fn)) if tp+fn else 0.0,
            'specificity':float(tn/(tn+fp)) if tn+fp else 0.0, 'n':int(len(y)),
            'positive':int(y.sum()),'negative':int((1-y).sum())}
