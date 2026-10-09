"""Prepare real RGB-derived regional features; no prognosis or policy training."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import zarr
from PIL import Image,ImageDraw
from torchvision.models import resnet18,ResNet18_Weights
from torchvision.transforms import functional as TF

from .remote_rgb import read_low_rgb
from .regions import crop_region,map_fine,rgb_regions


def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(sample,config,config_sha,output,cache,model,device,batch,commit):
    started=time.monotonic();folder=output/sample['slide_id']
    if folder.exists():
        receipt_path=folder/'receipt.json'
        if receipt_path.exists():
            receipt=json.loads(receipt_path.read_text())
            if (receipt.get('status')=='completed' and receipt['config_sha256']==config_sha
                    and sha256(folder/'regions.npz')==receipt['regions_sha256']):
                print('REGIONS_RESUME '+sample['slide_id'],flush=True);return receipt
        raise FileExistsError('Incomplete/mismatched output preserved; inspect before retry')
    folder.mkdir(parents=True)
    print('REGIONS_READ_RGB '+sample['slide_id'],flush=True)
    rgb,access=read_low_rgb(sample['source'],config,cache)
    mpp=access['selected_level']['mpp_xy'];l0=access['l0']
    image_wh=[l0['shape'][-1],l0['shape'][-2]]
    image_um=np.array(image_wh)*np.array(l0['mpp_xy'])
    # Define every candidate before opening the cached fine store.
    candidates=rgb_regions(rgb,image_um,mpp,config)
    Image.fromarray(rgb).save(folder/'low_rgb.png')
    fine=zarr.open_group(sample['fine_path'],mode='r')
    coords=np.asarray(fine['coords'][:])
    if fine['features'].shape != (len(coords),1024):raise ValueError('Fine contract mismatch')
    mapping=map_fine(coords,candidates,image_wh,l0['mpp_xy'],config)
    image=Image.fromarray(rgb); vectors=[];examples=[]
    mean=[0.485,0.456,0.406];std=[0.229,0.224,0.225]
    with torch.inference_mode():
        for start in range(0,len(candidates),batch):
            crops=[crop_region(image,c,mpp,config) for c in candidates[start:start+batch]]
            if len(examples)<8:examples.extend(crops[:8-len(examples)])
            x=torch.stack([TF.normalize(TF.to_tensor(c),mean,std) for c in crops]).to(device)
            vectors.append(model(x).cpu().numpy())
    coarse=np.concatenate(vectors).astype(np.float32)
    if coarse.shape != (len(candidates),512) or not np.isfinite(coarse).all():raise ValueError('Invalid coarse vectors')
    np.savez_compressed(folder/'regions.npz',coarse=coarse,
        bbox_um=np.array([c['bbox_um'] for c in candidates],dtype=np.float32),
        cell=np.array([c['cell'] for c in candidates],dtype=np.int64),
        tissue_fraction=np.array([c['tissue_fraction'] for c in candidates],dtype=np.float32),
        image_valid_fraction=np.array([c['image_valid_fraction'] for c in candidates],dtype=np.float32),**mapping)
    sheet=Image.new('RGB',(224*4,224*2),'white')
    for i,c in enumerate(examples):sheet.paste(c,((i%4)*224,(i//4)*224))
    sheet.save(folder/'region_examples.png')
    thumb=image.copy();thumb.thumbnail((900,700));draw=ImageDraw.Draw(thumb)
    sx,sy=thumb.width/image_um[0],thumb.height/image_um[1]
    for c in candidates:
        x0,y0,x1,y1=c['bbox_um'];draw.rectangle((x0*sx,y0*sy,x1*sx,y1*sy),outline=(0,180,0),width=1)
    thumb.save(folder/'regions_overlay.png')
    count=mapping['fine_counts'];retained=len(mapping['fine_indices'])
    receipt={'schema':'KT2028-region-1','status':'completed','slide_id':sample['slide_id'],
        'source_commit':commit,'config_sha256':config_sha,'config':config,'image_access':access,
        'fine_path':sample['fine_path'],'fine_patch_count':len(coords),'fine_patches_assigned':retained,
        'fine_patches_outside_rgb_candidates':len(coords)-retained,
        'fine_patches_partially_padded':int(((mapping['fine_image_valid_fraction']>0)&(mapping['fine_image_valid_fraction']<1)).sum()),
        'fine_patches_fully_outside_image':int((mapping['fine_image_valid_fraction']==0).sum()),
        'fine_coords_sha256':hashlib.sha256(coords.tobytes()).hexdigest(),
        'rgb_candidate_count':len(candidates),'acquirable_region_count':int((count>0).sum()),
        'fine_patches_per_acquirable_region':{'min':int(count[count>0].min()) if (count>0).any() else 0,
            'median':float(np.median(count[count>0])) if (count>0).any() else 0,
            'max':int(count.max())},'coarse_shape':list(coarse.shape),
        'regions_sha256':sha256(folder/'regions.npz'),
        'preprocessing':'physical crop -> 224 square with white padding -> ImageNet normalize; no additional crop',
        'cost_scope':'cached access simulation; fine coordinate index preprocessing not claimed free online',
        'candidate_status':'interface trial; not final hyperparameters',
        'elapsed_seconds':round(time.monotonic()-started,3),'no_model_training_or_evaluation':True}
    (folder/'receipt.json').write_text(json.dumps(receipt,indent=2))
    print('REGIONS_COMPLETE '+json.dumps({k:receipt[k] for k in ['slide_id','rgb_candidate_count','fine_patch_count','fine_patches_assigned','fine_patches_per_acquirable_region','elapsed_seconds']}),flush=True)
    return receipt


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--samples',type=Path,required=True)
    ap.add_argument('--config',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--range-cache',type=Path,required=True);ap.add_argument('--batch',type=int,default=32)
    ap.add_argument('--device',default='cuda');args=ap.parse_args()
    config=json.loads(args.config.read_text());config_sha=sha256(args.config)
    if args.batch<=0:raise ValueError('batch must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    zarr.config.set({'async.concurrency':1,'threading.max_workers':1})
    result=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True)
    commit=result.stdout.strip()
    model=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1);model.fc=torch.nn.Identity()
    model.eval().requires_grad_(False).to(args.device)
    rows=[];errors=[]
    for sample in json.loads(args.samples.read_text()):
        try: rows.append(prepare(sample,config,config_sha,args.output,args.range_cache,model,args.device,args.batch,commit))
        except Exception as e:
            error={'slide_id':sample['slide_id'],'error':type(e).__name__+': '+str(e)}
            errors.append(error);print('REGIONS_FAILED '+json.dumps(error),flush=True)
        summary={'schema':'KT2028-region-summary-1','config_sha256':config_sha,'source_commit':commit,
            'completed_images':len(rows),'errors':errors,'regions':sum(r['rgb_candidate_count'] for r in rows),
            'fine_patches_total':sum(r['fine_patch_count'] for r in rows),
            'fine_patches_assigned':sum(r['fine_patches_assigned'] for r in rows),
            'fine_patches_partially_padded':sum(r['fine_patches_partially_padded'] for r in rows),
            'fine_patches_fully_outside_image':sum(r['fine_patches_fully_outside_image'] for r in rows),
            'network_bytes':sum(r['image_access']['network_bytes'] for r in rows),
            'interface_trial':True,'no_training_or_evaluation':True}
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
