from pathlib import Path
import numpy as np,pandas as pd,json
from sklearn.cluster import MiniBatchKMeans
ROOT=Path('/root/autodl-tmp/surgen_hemicore');OUT=ROOT/'outputs/thesis_evidence_interface_v7';tr=pd.read_csv(ROOT/'prepared/sr386_full/surgen_train_10.csv');rng=np.random.default_rng(17);xs=[];patients=[]
for row in tr.itertuples():
 with np.load(row.features_filepath) as z:f=z['img_features'].astype('float32')
 for i in range(10):
  ids=rng.choice(len(f),8,replace=False) if len(f)>=8 else rng.choice(len(f),8,replace=True);xs.append(f[ids]);patients.extend([row.patient_id]*8)
x=np.concatenate(xs);x/=np.linalg.norm(x,axis=1,keepdims=True).clip(1e-8);km=MiniBatchKMeans(n_clusters=4,batch_size=4096,random_state=17,n_init=5,max_iter=100).fit(x);cent=km.cluster_centers_;cent/=np.linalg.norm(cent,axis=1,keepdims=True);np.save(OUT/'assets/train_kmeans4_spherical_centers.npy',cent);lab=km.labels_;pd.DataFrame({'cluster':lab}).to_csv(OUT/'metrics/kmeans_patch_assignments_sample.csv',index=False)
cov=[]
for c in range(4):
 mask=lab==c;cov.append({'cluster':c,'patch_fraction':float(mask.mean()),'patient_coverage':float(len(set(np.asarray(patients)[mask]))/len(tr)),'sampled_region_coverage':float(mask.sum()/(len(tr)*10*8))})
pd.DataFrame(cov).to_csv(OUT/'metrics/moe_routing_statistics.csv',index=False);pd.DataFrame(cent@cent.T).to_csv(OUT/'metrics/kmeans_center_cosine_similarity.csv',index=False);print(json.dumps(cov))
