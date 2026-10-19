"""A verified local CZI -> the same physical/UNI stores as the mixed TCGA input.

Extraction/preflight only. No predictor optimizer, training, or model evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
import numpy as np
from PIL import ImageDraw

from .local_czi import LocalCZI
from .dual_scale_geometry import build_nested_grid
from .extract_dual_scale import atomic_json,write_or_resume
from .prepare_tcga import verify_output
from .audit_tcga import validate_geometry
from .uni_encoder import OriginalUNI,sha256_file


def extract(args):
    root=args.root.resolve(strict=True)
    if (root/'_TASK_OWNER').read_text().strip()!='W067':
        raise ValueError('Not the owned SurGen feature workspace')
    if not re.fullmatch(r'SR386_40X_HE_T\d+_\d+',args.slide_id):
        raise ValueError('Invalid official slide identity')
    if args.source.is_symlink(): raise ValueError('Do not adopt a symlinked source')
    source=args.source.resolve(strict=True)
    if source.parent!=root/'incoming':
        raise ValueError('Source must be a task-owned working copy')
    if source.stat().st_size!=args.source_size or sha256_file(source)!=args.source_sha256:
        raise ValueError('Source transfer size/SHA256 mismatch; preserve source')
    cfg=json.loads(args.config.read_text())
    if (cfg['coarse_mpp_um'],cfg['fine_mpp_um'],cfg['input_pixels'],cfg['feature_dimension'])!=(2.,.5,224,1024):
        raise ValueError('Require the adopted physical UNI input protocol')
    folder=root/'features'/args.slide_id; folder.mkdir(parents=True,exist_ok=True)
    source_dir=Path(__file__).parent
    hashes={n:sha256_file(source_dir/n) for n in ('local_czi.py','prepare_surgen.py',
        'dual_scale_geometry.py','extract_dual_scale.py','uni_encoder.py')}
    protocol=hashlib.sha256(json.dumps(dict(config=sha256_file(args.config),source=hashes),sort_keys=True).encode()).hexdigest()
    run=dict(protocol=protocol,source_sha256=args.source_sha256,source_size=args.source_size)
    registration=folder/'run.json'
    if registration.exists() and json.loads(registration.read_text())!=run:
        raise ValueError('Different extraction protocol; preserve prior output')
    if not registration.exists(): atomic_json(registration,run)
    complete=folder/'receipt.json'
    if complete.exists():
        receipt=json.loads(complete.read_text())
        if receipt['protocol']!=protocol: raise ValueError('Completed protocol differs')
        verify_output(folder,receipt)
        return receipt
    started=time.time()
    with LocalCZI(source,cfg['maximum_native_upsampling_fraction']) as reader:
        thumb=reader.thumbnail(cfg['thumbnail_maximum_edge'])
        grid=build_nested_grid(np.asarray(thumb),reader.image_wh_um,
            tuple(a/b for a,b in zip(reader.image_wh_um,thumb.size)),reader.mpp,cfg)
        if not len(grid['fine_coords']): raise ValueError('No physical tissue fine patches')
        validate_geometry(grid,np.asarray(reader.mpp))
        geometry=folder/'geometry.npz'
        if geometry.exists():
            with np.load(geometry,allow_pickle=False) as old:
                for key in grid: np.testing.assert_array_equal(old[key],grid[key])
        else:
            np.savez_compressed(geometry,**grid)
        thumb.save(folder/'thumbnail.png')
        overlay=thumb.copy(); draw=ImageDraw.Draw(overlay)
        ratio=np.asarray(thumb.size)/reader.image_wh_um
        for box in grid['coarse_bbox_um']:
            draw.rectangle(tuple(box*np.tile(ratio,2)),outline=(255,180,0),width=1)
        overlay.save(folder/'coarse_overlay.png')
        if shutil.disk_usage(root).free<cfg['minimum_free_disk_bytes']:
            raise OSError('Insufficient reserved server space')
        encoder=OriginalUNI(args.checkpoint,cfg['uni'],args.device)
        encoder.torch.set_num_threads(4)
        import zarr
        zarr.config.set({'async.concurrency':1,'threading.max_workers':1})
        meta=dict(schema='KT2028-SurGen-dual-scale-1',protocol=protocol,
            source_sha256=args.source_sha256,source_size=args.source_size,
            config_sha256=sha256_file(args.config),weight_sha256=encoder.checkpoint_sha256,
            geometry_sha256=sha256_file(geometry),original_uni_frozen=True,
            inference_precision='float32',feature_dimension=1024,input_pixels=224,
            coords_definition='CZI image-local level0 top-left; document origin in slide_metadata')
        cg=dict(bbox=grid['coarse_bbox_um'],coords=grid['coarse_coords'],
                fine_counts=grid['fine_counts'],fine_ptr=grid['fine_ptr'])
        fg=dict(bbox=grid['fine_bbox_um'],coords=grid['fine_coords'],
                parent_indices=grid['fine_parent_indices'],relative_xy=grid['fine_relative_xy'])
        c=write_or_resume(folder/'coarse.zarr',cg,dict(meta,mpp_um=cfg['coarse_mpp_um']),
            224,cfg['coarse_mpp_um'],reader,encoder,cfg['batch_size'],cfg['minimum_free_disk_bytes'])
        f=write_or_resume(folder/'fine.zarr',fg,dict(meta,mpp_um=cfg['fine_mpp_um']),
            224,cfg['fine_mpp_um'],reader,encoder,cfg['batch_size'],cfg['minimum_free_disk_bytes'])
        receipt=dict(meta,status='completed',slide_id=args.slide_id,slide_metadata=reader.metadata(),
            coarse=c,fine=f,elapsed_seconds=time.time()-started,no_predictor_training=True)
        verify_output(folder,receipt)
        atomic_json(complete,receipt)
        return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--source-sha256',required=True)
    p.add_argument('--source-size',type=int,required=True)
    p.add_argument('--slide-id',required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    import fcntl
    lock=(args.root/'extraction.lock').open('a+b')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receipt=extract(args)
    print(json.dumps(dict(status=receipt['status'],coarse_rows=receipt['coarse']['rows'],
        fine_rows=receipt['fine']['rows'],elapsed_seconds=receipt['elapsed_seconds'],predictor_training=False)),flush=True)


if __name__=='__main__':main()
