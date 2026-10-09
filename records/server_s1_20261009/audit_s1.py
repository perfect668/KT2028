#!/usr/bin/env python3
"""S1: audit an uploaded SurGen bundle; preserve inputs, never train."""
import argparse, collections, csv, hashlib, importlib.metadata as md
import json, math, os, re, shutil, subprocess, sys, zipfile
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while block:=f.read(16*1024*1024): h.update(block)
    return h.hexdigest()

def case_key(name):
    m=re.fullmatch(r'SR386_40X_HE_T(\d+)(?:_\d+)?',name)
    if not m: raise ValueError('Unexpected SR386 identifier format')
    return 'SR386_40X_HE_T'+str(int(m.group(1)))

def write_json(path,value):
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--bundle',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--extract',type=Path,required=True)
    args=ap.parse_args()
    root=args.bundle.resolve(strict=True)
    out=args.output.resolve(); target=args.extract.resolve()
    if out.exists() or target.exists(): raise FileExistsError('Use fresh audit and extraction directories')
    if out==root or target==root or root in out.parents or root in target.parents:
        raise ValueError('Derived outputs must be outside the input bundle')
    out.mkdir(parents=True); target.mkdir(parents=True)
    report={'schema':'KT2028-S1-1','started_utc':datetime.now(timezone.utc).isoformat(),
            'plan_version':'P1','local_repository_head':'7ed0dcdd023f0dd963dda95a60733729473d4766',
            'input_root':str(root),'output_root':str(out),'extracted_root':str(target),
            'scope':'asset and schema audit; no training or evaluation'}
    report['script_sha256']=digest(Path(__file__))
    report['resources']={}
    for name,cmd in {
        'gpu':['nvidia-smi','--query-gpu=name,memory.total,driver_version,memory.used,utilization.gpu','--format=csv,noheader'],
        'disk':['df','-h','/','/root/autodl-tmp'],
        'kernel':['uname','-sr']}.items():
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=20)
        report['resources'][name]={'exit_code':r.returncode,'stdout':r.stdout.strip(),'stderr':r.stderr.strip()}
    report['resources']['memory_limit_bytes']=int(Path('/sys/fs/cgroup/memory.max').read_text())
    report['resources']['cpu_quota']=Path('/sys/fs/cgroup/cpu.max').read_text().strip()
    report['resources']['python']=sys.version
    report['resources']['python_executable']=sys.executable
    report['packages']={}
    for name in ['torch','torchvision','numpy','pandas','zarr','numcodecs','h5py','tifffile','gymnasium','stable-baselines3','sb3-contrib']:
        try: report['packages'][name]=md.version(name)
        except md.PackageNotFoundError: report['packages'][name]=None
    archive=root/'embeddings/SurGen_UNI_patch_embeddings.zip'
    print('S1 checking archive SHA256',flush=True)
    archive_hash=digest(archive)
    expected=(root/'SHA256SUMS').read_text().split()[0]
    if archive_hash!=expected: raise ValueError('Archive SHA256 mismatch')
    report['archive']={'bytes':archive.stat().st_size,'sha256':archive_hash,
                       'uploaded_sha256_match':True,
                       'md5_from_independent_preflight':'6668206cc5e0ae0cee00ee554cb2e410'}
    report['source_copy']={'git_checkout':(root/'src/SurGen-Dataset/.git').is_dir(),
        'commit':'unverified; uploaded source copy has no .git'}
    important=['README.md','reproducibility/README.md','reproducibility/datasets/dataset_constants.py',
               'reproducibility/datasets/dataset_loader.py','reproducibility/wsi_core/WholeSlideImage.py']
    report['source_copy']['file_sha256']={p:digest(root/'src/SurGen-Dataset'/p) for p in important}
    report['raw_wsi_files']=sum(p.is_file() for p in (root/'raw_wsi').rglob('*'))
    stats=[]; by_case={}; manifests={}; summary={}
    with zipfile.ZipFile(archive) as z:
        infos=z.infolist(); names={i.filename for i in infos}
        slides=sorted({n.split('/')[0][:-5] for n in names if n.startswith('SR386_') and n.split('/')[0].endswith('.zarr')})
        cohort_counts=collections.Counter(n.split('/')[0].split('_')[0] for n in names if n.endswith('.zarr/'))
        report['archive']['cohort_store_counts']=dict(cohort_counts)
        report['archive']['total_uncompressed_bytes']=sum(i.file_size for i in infos)
        selected=[i for i in infos if i.filename.split('/')[0] in {s+'.zarr' for s in slides}]
        selected_bytes=sum(i.file_size for i in selected)
        if shutil.disk_usage(target).free < selected_bytes+5*1024**3: raise OSError('Insufficient extraction headroom')
        for slide in slides:
            group=slide+'.zarr'; f=json.loads(z.read(group+'/features/.zarray')); c=json.loads(z.read(group+'/coords/.zarray'))
            if f['shape'][0]!=c['shape'][0] or f['shape'][1]!=1024 or c['shape'][1]!=2:
                raise ValueError('Feature/coordinate shape mismatch')
            for arr,meta in [('features',f),('coords',c)]:
                if meta['zarr_format']!=2 or meta['chunks'][1]!=meta['shape'][1]:
                    raise ValueError('Unexpected Zarr layout')
                expected_chunks={group+'/'+arr+'/'+str(i)+'.0' for i in range(math.ceil(meta['shape'][0]/meta['chunks'][0]))}
                present={n for n in names if n.startswith(group+'/'+arr+'/') and not n.split('/')[-1].startswith('.') and not n.endswith('/')}
                if present!=expected_chunks: raise ValueError('Missing or unexpected data chunks')
            key=case_key(slide)
            if key in by_case: raise ValueError('Ambiguous case-to-slide mapping')
            by_case[key]=slide
            stats.append({'case_id':key,'slide_id':slide,'patch_count':f['shape'][0],
                          'feature_dim':f['shape'][1],'feature_dtype':f['dtype'],'coordinate_dtype':c['dtype'],
                          'coords_columns':2,'zarr_format':2,'compressor':f['compressor']['id']})
        partitions={}
        for split in ['train','validate','test']:
            csv_path=root/('metadata/dataset_csv/SR386_5y_sur_'+split+'.csv')
            source_csv=root/('src/SurGen-Dataset/reproducibility/dataset_csv/SR386_5y_sur_'+split+'.csv')
            with csv_path.open(newline='',encoding='utf-8-sig') as handle:
                reader=csv.DictReader(handle); rows=list(reader)
            if reader.fieldnames!=['case_id','slide_id','label']: raise ValueError('Unexpected label CSV columns')
            ids=[r['case_id'] for r in rows]
            if len(set(ids))!=len(ids): raise ValueError('Duplicate case ids within split')
            patients=[]
            for row in rows:
                key=case_key(row['case_id'])
                if row['label'] not in ['0','1'] or key not in by_case: raise ValueError('Invalid label or missing features')
                slide=by_case[key]
                patients.append({'case_id':row['case_id'],'canonical_case_key':key,'label':int(row['label']),
                                 'source_slide_id_field':row['slide_id'],
                                 'slides':[{'slide_id':slide,'features_path':str(target/(slide+'.zarr'))}]})
            partitions[split]={p['canonical_case_key'] for p in patients}
            manifests[split]={'schema':'KT2028-S1-1','endpoint':'five_year_death',
                'endpoint_evidence':'uploaded dataset_constants.py: Died within 5y=1; Survived 5y=0; raw follow-up unavailable',
                'partition':split,'source_csv':str(csv_path),'source_csv_sha256':digest(csv_path),'patients':patients,
                'status':'asset manifest; not a training-ready dual-scale manifest'}
            summary[split]={'rows':len(rows),'unique_case_ids':len(ids),
                'label_counts':dict(collections.Counter(r['label'] for r in rows)),
                'missing_features':0,'metadata_matches_source_copy':digest(csv_path)==digest(source_csv),
                'csv_sha256':digest(csv_path)}
        overlaps={a+'__'+b:len(partitions[a]&partitions[b]) for a,b in [('train','validate'),('train','test'),('validate','test')]}
        if any(overlaps.values()): raise ValueError('Case overlap across splits')
        unlabelled=sorted(set(by_case)-set.union(*partitions.values()))
        report['mapping']={'rule':'CSV case_id integer T key -> zero-padded archive T key; source slide_id is a split-local placeholder',
                           'case_split_overlap':overlaps,'split_summary':summary,
                           'unlabelled_feature_cases':len(unlabelled),
                           'patient_identity_scope':'case_id separation verified; independent clinical patient linkage unavailable'}
        report['features']={'sr386_stores':len(slides),'patches':sum(s['patch_count'] for s in stats),
            'patch_min':min(s['patch_count'] for s in stats),'patch_max':max(s['patch_count'] for s in stats),
            'sr386_uncompressed_bytes':selected_bytes,'metadata_shapes_and_chunk_keys_verified':True,
            'array_values_decoded':False,'fine_coordinate_mpp_provenance':'source protocol suggests level-0 coordinates, 224px at target 1 MPP; no array .zattrs',
            'extraction_crc_verified':False}
        write_json(out/'audit_summary.json',report)
        print('S1 extracting SR386 only, entries='+str(len(selected)),flush=True)
        for i,info in enumerate(selected,1):
            p=PurePosixPath(info.filename)
            if p.is_absolute() or '..' in p.parts or (info.external_attr>>16)&0o170000==0o120000:
                raise ValueError('Unsafe archive member')
            dest=target.joinpath(*p.parts)
            if info.is_dir(): dest.mkdir(parents=True,exist_ok=True)
            else:
                dest.parent.mkdir(parents=True,exist_ok=True)
                with z.open(info) as input_stream, dest.open('xb') as output_stream:
                    shutil.copyfileobj(input_stream,output_stream,1024*1024)
            if i%3000==0: print('S1 extraction entries completed='+str(i),flush=True)
    report['features']['extraction_crc_verified']=True
    report['finished_utc']=datetime.now(timezone.utc).isoformat()
    # Preserve the pre-extraction summary and add a separate completion receipt.
    write_json(out/'completion_receipt.json',report)
    for split,manifest in manifests.items(): write_json(out/(split+'_asset_manifest.json'),manifest)
    with (out/'sr386_feature_inventory.csv').open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(stats[0]));writer.writeheader();writer.writerows(stats)
    write_json(out/'unlabelled_feature_cases.json',unlabelled)
    (out/'README.txt').write_text('Use completion_receipt.json for final status. asset manifests contain labels and stay on server. No training performed. Array values not decoded. Raw WSI/cheap RGB missing. Original bundle preserved.\n')
    print('S1_COMPLETED '+json.dumps({'output':str(out),'extract':str(target),'split_counts':{k:v['rows'] for k,v in summary.items()},'sr386_stores':len(stats),'patches':report['features']['patches'],'extracted_bytes':selected_bytes}),flush=True)

if __name__=='__main__': main()

