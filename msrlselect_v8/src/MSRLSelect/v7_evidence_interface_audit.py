#!/usr/bin/env python3
from pathlib import Path
import numpy as np,pandas as pd,torch,json,hashlib
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
ROOT=Path('/root/autodl-tmp/surgen_hemicore');BASE=ROOT/'outputs/thesis_predictor_first_v6';OUT=ROOT/'outputs/thesis_evidence_interface_v7';CACHE=BASE/'p2_variable_evidence_predictor/cache_compact'
def main():
 for d in ['docs','metrics','predictions','checkpoints','configs','assets','logs']:(OUT/d).mkdir(parents=True,exist_ok=True)
 tr=pd.read_csv(ROOT/'prepared/sr386_full/surgen_train_10.csv');va=pd.read_csv(ROOT/'prepared/sr386_full/surgen_validate_10.csv');coarse=torch.load(CACHE/'train/coarse.pt',weights_only=True).float().numpy();fine=torch.load(CACHE/'train/fine_seed17.pt',weights_only=True).float().numpy();
 # E1 PCA is fit only on Train regional UNI means (the 1024 mean slice of coarse).
 x=coarse[:,:,:1024].reshape(-1,1024);pca=PCA(n_components=128,random_state=17);pca.fit(x);np.save(OUT/'assets/train_pca128_components.npy',pca.components_.astype('float32'));np.save(OUT/'assets/train_pca128_mean.npy',pca.mean_.astype('float32'));np.save(OUT/'assets/train_pca128_explained_variance.npy',pca.explained_variance_.astype('float32'))
 # Train-only patch sample for E2 initialization; no Validation access.
 rng=np.random.default_rng(17);sample=[]
 for row in tr.itertuples():
  with np.load(row.features_filepath) as z:f=z['img_features'].astype('float32')
  ids=rng.choice(len(f),min(128,len(f)),replace=False);sample.append(f[ids])
 sample=np.concatenate(sample);km=MiniBatchKMeans(n_clusters=4,batch_size=1024,random_state=17,n_init=3,max_iter=50);km.fit(sample);np.save(OUT/'assets/train_kmeans4_centers.npy',km.cluster_centers_.astype('float32'))
 configs={'E0_current':{'coarse_dim':2053,'fine_dim':384,'teacher_in_forward':False},'E1_bottleneck_single':{'coarse_dim':133,'pca_dim':128,'fine_dim':384,'teacher_in_forward':False},'E2_bottleneck_moe4':{'coarse_dim':133,'experts':4,'top_k':2,'slot_dim':256,'teacher_in_forward':False,'kmeans_init':'train_only'}}
 for n,c in configs.items():(OUT/'configs'/f'{n}.json').write_text(json.dumps(c,indent=2))
 # Existing E0 action separability baseline, clearly labeled historical/provisional.
 av=pd.read_csv(ROOT/'outputs/thesis_rl_upgrade_v6/p3r_complete_pretraining/metrics/counterfactual_action_values.csv');g=av.groupby('k').reward_raw.agg(['std',lambda q:q.quantile(.9)-q.quantile(.1),lambda q:q.max()-q.min()]).reset_index();g.columns=['k','action_std','p90_p10','max_min'];noise=float(fine.std()/np.sqrt(fine.shape[-1]));g['e0_noise_snr']=g.action_std/(noise+1e-9);g.to_csv(OUT/'metrics/e0_action_identifiability_baseline.csv',index=False)
 pd.DataFrame([{'component':'E0','status':'AUDIT_ONLY','reason':'reuses current P2-E interface'},{'component':'E1','status':'ASSET_READY_NOT_TRAINED','reason':'Train-only PCA-128 fitted'},{'component':'E2','status':'ASSET_READY_NOT_TRAINED','reason':'Train-only 4-means initialization fitted; router/loss not trained'}]).to_csv(OUT/'metrics/config_status.csv',index=False)
 (OUT/'docs/v7_evidence_interface_report.md').write_text(f'''# MS-RLSurv V7 Phase E 证据接口审计\n\nE0/E1/E2配置已建立。E1 PCA-128仅使用Train的{len(x)}个区域UNI均值拟合；E2四专家KMeans仅使用Train随机patch样本拟合。未读取Test。\n\n当前 E0 动作可辨识性基线已从P3-R历史动作值计算；E1/E2尚未训练，不能把PCA/KMeans资产当作预测器结果。由于本阶段禁止beam/PPO/STOP，输出仅作为接口重构准备，正式seed17筛选需在新增模型实现完成后执行。\n''')
 (OUT/'docs/v7_reproducible_commands.sh').write_text('#!/usr/bin/env bash\ncd /root/autodl-tmp/surgen_hemicore/src/MSRLSelect\n/root/autodl-tmp/kits23_kdr_kbp_v1_envs/kbp-seg/bin/python v7_evidence_interface_audit.py\n')
 rows=[]
 for p in OUT.rglob('*'):
  if p.is_file() and 'sha256' not in p.name:rows.append({'path':str(p.relative_to(OUT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 pd.DataFrame(rows).to_csv(OUT/'metrics/v7_sha256_manifest.csv',index=False);print(json.dumps({'pca_train_rows':len(x),'kmeans_samples':len(sample),'status':'PHASE_E_INTERFACE_AUDIT_ONLY'}))
if __name__=='__main__':main()
