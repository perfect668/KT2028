#!/usr/bin/env python3
"""Read actual SR386 arrays and audit values/coordinates; do not fit models."""
import argparse, collections, csv, hashlib, json, time
from pathlib import Path
import numpy as np
import zarr
from numcodecs import blosc

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--features',type=Path,required=True)
    ap.add_argument('--s1',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError('Preserve previous audit output')
    args.output.mkdir(parents=True)
    blosc.set_nthreads(4)
    zarr.config.set({'async.concurrency':8,'threading.max_workers':8})
    manifests={part:json.loads((args.s1/(part+'_asset_manifest.json')).read_text())
               for part in ['train','validate','test']}
    labels={}
    for part,m in manifests.items():
        for p in m['patients']:
            labels[p['slides'][0]['slide_id']]=part
    ome=json.loads((args.s1/'ome_file_list.json').read_text())
    image_map={Path(i['path']).name.removesuffix('.ome.zarr.zip'):i
               for i in ome if '/SR386_' in i['path']}
    rows=[]; errors=[];steps=collections.Counter();start=time.monotonic()
    stores=sorted(args.features.glob('*.zarr'))
    for idx,path in enumerate(stores,1):
        slide=path.stem
        try:
            g=zarr.open_group(str(path),mode='r')
            f=np.asarray(g['features'][:]); c=np.asarray(g['coords'][:])
            if f.shape!=(c.shape[0],1024) or c.shape[1:]!=(2,):raise ValueError('shape mismatch')
            finite=np.isfinite(f)
            duplicate=int(len(c)-len(np.unique(c,axis=0)))
            deltas=[]
            for axis in range(2):
                u=np.unique(c[:,axis]);d=np.diff(u);d=d[d>0]
                deltas.extend(map(int,d));steps.update(map(int,d))
            row={'slide_id':slide,'partition':labels.get(slide,'unlabelled'),
                 'patches':len(c),'nonfinite_values':int((~finite).sum()),
                 'zero_feature_rows':int((f==0).all(axis=1).sum()),
                 'duplicate_coordinates':duplicate,
                 'negative_coordinate_entries':int((c<0).sum()),
                 'x_min':int(c[:,0].min()),'x_max':int(c[:,0].max()),
                 'y_min':int(c[:,1].min()),'y_max':int(c[:,1].max()),
                 'feature_min':float(f[finite].min()) if finite.any() else None,
                 'feature_max':float(f[finite].max()) if finite.any() else None,
                 'positive_axis_step_median':float(np.median(deltas)) if deltas else None,
                 'official_image_index_match':slide in image_map}
            rows.append(row)
            if row['nonfinite_values'] or duplicate or row['zero_feature_rows']:
                errors.append({'slide_id':slide,'reason':'nonfinite, duplicate coordinates or zero rows',
                               'nonfinite_values':row['nonfinite_values'],
                               'duplicate_coordinates':duplicate,'zero_feature_rows':row['zero_feature_rows']})
        except Exception as e:errors.append({'slide_id':slide,'error':type(e).__name__+': '+str(e)[:200]})
        if idx%50==0:print('FINE_AUDIT '+str(idx)+'/'+str(len(stores)),flush=True)
    with (args.output/'slide_value_audit.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    train=sorted([r for r in rows if r['partition']=='train'],key=lambda r:r['patches'])
    sample=[train[0],train[len(train)//2],train[-1]]
    with (args.output/'train_image_samples.json').open('x') as f:
        json.dump([{'slide_id':r['slide_id'],'feature_patches':r['patches'],
                    'source':image_map[r['slide_id']],'fine_path':str(args.features/(r['slide_id']+'.zarr'))}
                   for r in sample],f,indent=2)
    summary={'schema':'KT2028-S2-fine-1','arrays_read':len(rows),'stores_expected':len(stores),
             'patches':sum(r['patches'] for r in rows),
             'nonfinite_values':sum(r['nonfinite_values'] for r in rows),
             'zero_feature_rows':sum(r['zero_feature_rows'] for r in rows),
             'duplicate_coordinates':sum(r['duplicate_coordinates'] for r in rows),
             'negative_coordinate_entries':sum(r['negative_coordinate_entries'] for r in rows),
             'official_image_index_matches':sum(r['official_image_index_match'] for r in rows),
             'labelled_case_count':len(labels),
             'labelled_missing_image_index':sum(s not in image_map for s in labels),
             'axis_step_frequency_top10':steps.most_common(10),
             'partition_counts':dict(collections.Counter(r['partition'] for r in rows)),
             'errors':errors,'elapsed_seconds':round(time.monotonic()-start,3),
             'coord_units':'integer level-0 source pixels, inferred from uploaded source; per-image scale to verify',
             'no_training':True,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with (args.output/'fine_audit_summary.json').open('x') as f:json.dump(summary,f,indent=2)
    print('FINE_AUDIT_COMPLETED '+json.dumps(summary),flush=True)
    if errors or len(rows)!=len(stores):raise SystemExit(1)
if __name__=='__main__':main()

