"""Frozen original UNI extraction on real physical ROIs; restart per batch.

No labels, predictor, policy, or synthetic embeddings. Completed cases remove
only task-owned temporary HTTP ranges; source images and all historical caches
are preserved. Noncomplete stores must not be used for model training.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import zarr

from .remote_pyramid import RemotePyramid
from .uni_encoder import OriginalUNI, sha256_file


def atomic_json(path, value):
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def write_or_resume(path, geometry, metadata, pixels, mpp, reader, encoder, batch, min_free):
    bbox = geometry['bbox']
    n = len(bbox)
    if path.exists():
        group = zarr.open_group(str(path), mode='a')
        if any(group.attrs.get(k) != v for k, v in metadata.items()):
            raise ValueError('Existing feature source/protocol/geometry mismatch')
        if group['features'].shape != (n, 1024) or str(group['features'].dtype) != 'float32':
            raise ValueError('Existing feature shape/dtype mismatch')
        np.testing.assert_array_equal(group['coords'][:], geometry['coords'])
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        group = zarr.open_group(str(path), mode='w-', zarr_format=2)
        group.attrs.update(dict(metadata, status='in_progress', completed_rows=0))
        group.create_array('features', shape=(n, 1024), dtype='float32',
                           chunks=(max(1, min(batch, n)), 1024), fill_value=np.nan)
        for name, data in geometry.items():
            if name == 'bbox':
                name = 'bbox_um'
            group.create_array(name, data=np.asarray(data))
    start = int(group.attrs['completed_rows'])
    if not 0 <= start <= n:
        raise ValueError('Invalid feature progress')
    digest = hashlib.sha256()
    for offset in range(0, start, batch):
        values = np.asarray(group['features'][offset:min(offset+batch, start)])
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite completed feature rows')
        digest.update(values.tobytes())
    if group.attrs.get('status') == 'completed':
        if start != n or digest.hexdigest() != group.attrs['features_sha256']:
            raise ValueError('Completed feature checksum mismatch')
        return dict(rows=n, features_sha256=digest.hexdigest(), resumed_complete=True)
    for offset in range(start, n, batch):
        if shutil.disk_usage(path).free < min_free:
            raise OSError('Disk headroom below extraction contract')
        images = [reader.read_patch(float(b[0]), float(b[1]), mpp, pixels)
                  for b in bbox[offset:offset+batch]]
        values = encoder.encode(images)
        group['features'][offset:offset+len(values)] = values
        # Verify the persisted batch before committing its progress marker.
        stored = np.asarray(group['features'][offset:offset+len(values)])
        if not np.array_equal(stored, values):
            raise ValueError('Feature save/reload mismatch')
        digest.update(values.tobytes())
        group.attrs['completed_rows'] = offset + len(values)
        print(json.dumps({'event': 'encoded_batch', 'scale_mpp': mpp,
                          'completed': offset+len(values), 'total': n}), flush=True)
    group.attrs.update(status='completed', features_sha256=digest.hexdigest())
    return dict(rows=n, features_sha256=digest.hexdigest(), resumed_complete=False)


def owned_cache(root, source, config_sha):
    # CachedRangeFile's canonical identity; no old cache root is adopted here.
    canonical = 'https://www.ebi.ac.uk/biostudies/files/S-BIAD1285/' + source['path']
    identity = hashlib.sha256(f"{canonical}\n{source['size']}".encode()).hexdigest()
    folder = root / identity
    owner = dict(schema='KT2028-R2-temporary-ranges', config_sha256=config_sha,
                 source_identity=identity, source_zip_bytes=source['size'])
    marker = folder / '_OWNER.json'
    if folder.exists():
        if not marker.is_file() or json.loads(marker.read_text()) != owner:
            raise ValueError('Preserve non-owned range cache')
    else:
        folder.mkdir(parents=True)
        with marker.open('x') as f:
            json.dump(owner, f)
    return folder, owner


def clear_completed_cache(root, folder, owner):
    # Never remove a root, historical cache, symlink, or unverified computed path.
    root = root.resolve(strict=True)
    if folder.is_symlink() or folder.resolve(strict=True).parent != root:
        raise ValueError('Cache cleanup escaped task root')
    if folder.name != owner['source_identity'] or len(folder.name) != 64:
        raise ValueError('Invalid owned cache identity')
    if json.loads((folder / '_OWNER.json').read_text()) != owner:
        raise ValueError('Cache ownership changed')
    shutil.rmtree(folder)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--samples', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--geometry', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--fallback-cache', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max-cases', type=int)
    ap.add_argument('--case-deadline-seconds', type=float, default=28800)
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    config_sha = sha256_file(args.config)
    samples = json.loads(args.samples.read_text())
    if args.max_cases is not None:
        if args.max_cases <= 0:
            raise ValueError('Invalid case limit')
        samples = samples[:args.max_cases]
    if not samples or len({s['slide_id'] for s in samples}) != len(samples):
        raise ValueError('Empty or duplicate source mapping')
    # Fail before image traffic or feature-store allocation when weights are absent.
    encoder = OriginalUNI(args.checkpoint, config['uni'], args.device)
    zarr.config.set({'async.concurrency': 1, 'threading.max_workers': 1})
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    run_protocol = dict(config_sha256=config_sha, samples_sha256=sha256_file(args.samples),
                        weight_sha256=encoder.checkpoint_sha256,
                        code_sha256={name: sha256_file(Path(__file__).with_name(name)) for name in
                                     ['extract_dual_scale.py', 'remote_pyramid.py', 'remote_rgb.py', 'uni_encoder.py']})
    run_path = args.output / 'run.json'
    if run_path.exists():
        previous = json.loads(run_path.read_text())
        if any(previous.get(k) != v for k, v in run_protocol.items()):
            raise ValueError('Extraction run protocol/source files changed; preserve existing run')
    else:
        try:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        except (OSError, subprocess.SubprocessError):
            raise ValueError('Extraction must run from a traceable Git checkout')
        from importlib.metadata import version
        atomic_json(run_path, dict(run_protocol, source_commit=commit, device=args.device,
                    environment={name: version(name) for name in ['torch', 'timm', 'zarr', 'numpy']},
                    scope='frozen original UNI; real Train RGB; no prognosis/policy training'))
    lock = args.output / '.extract.lock'
    with lock.open('x') as f:
        json.dump({'pid': os.getpid(), 'config_sha256': config_sha}, f)
    rows = []
    started = time.monotonic()
    try:
        for sample in samples:
            sid = sample['slide_id']
            if Path(sid).name != sid or sid in ('.', '..'):
                raise ValueError('Invalid slide identifier')
            geometry_dir = args.geometry / sid
            g_receipt = json.loads((geometry_dir / 'receipt.json').read_text())
            if g_receipt.get('status') != 'geometry_only_prepared' or g_receipt['config_sha256'] != config_sha:
                raise ValueError('Geometry protocol not ready')
            geometry_path = geometry_dir / 'geometry.npz'
            if sha256_file(geometry_path) != g_receipt['geometry_sha256']:
                raise ValueError('Geometry checksum mismatch')
            case_receipt = args.output / 'receipts' / (sid + '.json')
            case_receipt.parent.mkdir(parents=True, exist_ok=True)
            # Completed cases are validated from their stores and skipped without network.
            if case_receipt.exists():
                old = json.loads(case_receipt.read_text())
                if old.get('status') != 'completed' or old.get('weight_sha256') != encoder.checkpoint_sha256 or old.get('config_sha256') != config_sha or old.get('geometry_sha256') != g_receipt['geometry_sha256']:
                    raise ValueError('Existing completion receipt mismatch')
                for scale, expected in [('mpp_2p0', old['coarse']), ('mpp_0p5', old['fine'])]:
                    group = zarr.open_group(str(args.output / scale / (sid+'.zarr')), mode='r')
                    h = hashlib.sha256()
                    for i in range(0, expected['rows'], config['batch_size']):
                        value = np.asarray(group['features'][i:i+config['batch_size']])
                        if not np.isfinite(value).all():
                            raise ValueError('Completed feature rows are corrupt')
                        h.update(value.tobytes())
                    if group.attrs.get('status') != 'completed' or group['features'].shape != (expected['rows'], 1024) or h.hexdigest() != expected['features_sha256']:
                        raise ValueError('Completed feature checksum mismatch')
                rows.append(old)
                continue
            folder, owner = owned_cache(args.cache, sample['source'], config_sha)
            with np.load(geometry_path, allow_pickle=False) as grid, RemotePyramid(
                    sample['source'], args.cache, config['max_network_bytes_per_image'],
                    config['max_roi_pixels'], args.case_deadline_seconds, args.fallback_cache) as reader:
                if g_receipt['source_zip_bytes'] != reader.raw.size:
                    raise ValueError('Source geometry size mismatch')
                if g_receipt['root_metadata_sha256'] is not None and g_receipt['root_metadata_sha256'] != reader.root_sha:
                    raise ValueError('Source metadata changed since RGB preparation')
                meta = dict(schema='KT2028-dual-scale-2', config_sha256=config_sha,
                            weight_sha256=encoder.checkpoint_sha256,
                            geometry_sha256=g_receipt['geometry_sha256'], source_root_sha256=reader.root_sha,
                            coords_definition=g_receipt['coords_definition'], original_uni_frozen=True,
                            inference_precision='float32', feature_dimension=1024)
                coarse_geo = dict(bbox=grid['coarse_bbox_um'], coords=grid['coarse_coords'],
                                  fine_counts=grid['fine_counts'], fine_ptr=grid['fine_ptr'])
                fine_geo = dict(bbox=grid['fine_bbox_um'], coords=grid['fine_coords'],
                                parent_indices=grid['fine_parent_indices'], relative_xy=grid['fine_relative_xy'])
                coarse = write_or_resume(args.output/'mpp_2p0'/(sid+'.zarr'), coarse_geo,
                                         dict(meta, mpp_um=2.0), config['input_pixels'], 2.0,
                                         reader, encoder, config['batch_size'], config['minimum_free_disk_bytes'])
                fine = write_or_resume(args.output/'mpp_0p5'/(sid+'.zarr'), fine_geo,
                                       dict(meta, mpp_um=0.5), config['input_pixels'], 0.5,
                                       reader, encoder, config['batch_size'], config['minimum_free_disk_bytes'])
                receipt = dict(meta, status='completed', slide_id=sid, coarse=coarse, fine=fine,
                               image_access=reader.receipt(), no_predictor_training=True)
                atomic_json(case_receipt, receipt)
                rows.append(receipt)
            clear_completed_cache(args.cache, folder, owner)
            atomic_json(args.output/'summary.json', dict(status='running', expected=len(samples),
                        completed=len(rows), coarse_patches=sum(r['coarse']['rows'] for r in rows),
                        fine_patches=sum(r['fine']['rows'] for r in rows),
                        elapsed_seconds=round(time.monotonic()-started, 3)))
        atomic_json(args.output/'summary.json', dict(status='completed', expected=len(samples),
                    completed=len(rows), coarse_patches=sum(r['coarse']['rows'] for r in rows),
                    fine_patches=sum(r['fine']['rows'] for r in rows),
                    elapsed_seconds=round(time.monotonic()-started, 3)))
    except Exception as e:
        atomic_json(args.output/'failure.json', dict(status='failed', completed=len(rows),
                    error=type(e).__name__+': '+str(e)[:400], resumable=True))
        raise
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
