"""TCGA local SVS -> outcome-independent aligned 5x/20x original-UNI features.

This command never trains a predictor/policy, creates patient splits, or deletes
originals. Each completed slide carries its source hash and resumable stores.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import tarfile

import numpy as np
from PIL import Image, ImageDraw
import zarr

from .dual_scale_geometry import build_nested_grid
from .extract_dual_scale import atomic_json, write_or_resume
from .local_svs import LocalSVS
from .uni_encoder import OriginalUNI, sha256_file


def md5_file(path):
    h = hashlib.md5()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8*1024**2), b''):
            h.update(part)
    return h.hexdigest()


def source_path(root, sample):
    import uuid
    if str(uuid.UUID(sample['id'])) != sample['id']:
        raise ValueError('Invalid GDC file UUID')
    return root/'incoming'/(sample['id']+'.svs')


def stage(root, sample):
    dest = source_path(root, sample)
    candidate = dest if dest.exists() else dest.with_suffix('.svs.partial')
    if candidate.is_symlink() or candidate.stat().st_size != sample['size']:
        raise ValueError('Source size/type mismatch')
    if md5_file(candidate) != sample['md5']:
        raise ValueError('GDC source MD5 mismatch')
    if candidate != dest:
        candidate.rename(dest)
    receipt = dict(file_id=sample['id'], size=sample['size'], md5=sample['md5'],
                   verified_at_unix=time.time(), source_status='md5_verified')
    folder = root/'staged'; folder.mkdir(exist_ok=True)
    atomic_json(folder/(sample['id']+'.json'), receipt)
    return receipt


def prepare_geometry(root, output, sample, config):
    dest = source_path(root, sample)
    folder = output/sample['id']; folder.mkdir(parents=True, exist_ok=True)
    with LocalSVS(dest, config['maximum_native_upsampling_fraction']) as reader:
        thumb = reader.thumbnail(config['thumbnail_maximum_edge'])
        rgb = np.asarray(thumb)
        mask_mpp = tuple(a/b for a, b in zip(reader.image_wh_um, thumb.size))
        grid = build_nested_grid(rgb, reader.image_wh_um, mask_mpp, reader.mpp, config)
        if len(grid['fine_coords']) == 0:
            raise ValueError('No fine patches after tissue filter')
        metadata = reader.metadata()
        metadata.update(coarse_read_level=reader.select_level(config['coarse_mpp_um']),
                        fine_read_level=reader.select_level(config['fine_mpp_um']))
        geometry = folder/'geometry.npz'
        if geometry.exists():
            with np.load(geometry, allow_pickle=False) as old:
                for name, values in grid.items():
                    np.testing.assert_array_equal(old[name], values)
        else:
            with (folder/'geometry.partial').open('wb') as f:
                np.savez_compressed(f, **grid)
            (folder/'geometry.partial').rename(geometry)
        thumb.save(folder/'thumbnail.png')
        overlay = thumb.copy(); draw = ImageDraw.Draw(overlay)
        for box in grid['coarse_bbox_um']:
            draw.rectangle(tuple((box/np.asarray([*mask_mpp, *mask_mpp])).tolist()),
                           outline=(255, 180, 0), width=1)
        overlay.save(folder/'coarse_overlay.png')
        # Actual physical pixels for a deterministic central candidate; no label selection.
        idx = len(grid['coarse_coords'])//2
        x, y = grid['coarse_bbox_um'][idx, :2]
        coarse = reader.read_patch(x, y, config['coarse_mpp_um'])
        panel = Image.new('RGB', (224*5, 224*4), 'white')
        panel.paste(coarse, (0, 0))
        for k, box in enumerate(grid['fine_bbox_um'][grid['fine_parent_indices'] == idx]):
            panel.paste(reader.read_patch(*box[:2], config['fine_mpp_um']),
                        (224*(1+k%4), 224*(k//4)))
        panel.save(folder/'paired_pixels.png')
    return grid, metadata, sha256_file(geometry)


def extract_one(root, output, sample, config, config_hash, encoder, protocol):
    started = time.monotonic()
    dest = source_path(root, sample)
    if dest.stat().st_size != sample['size']:
        raise ValueError('Staged source size changed')
    stage_receipt = json.loads((root/'staged'/(sample['id']+'.json')).read_text())
    if stage_receipt['md5'] != sample['md5'] or stage_receipt['size'] != sample['size']:
        raise ValueError('Staged source fingerprint changed')
    folder = output/sample['id']
    complete = folder/'receipt.json'
    if complete.exists():
        old = json.loads(complete.read_text())
        if old['protocol'] != protocol or old['source_md5'] != sample['md5']:
            raise ValueError('Preserve output from different extraction protocol')
        if old['status'] == 'completed':
            verify_output(folder, old)
            return old
    grid, slide_meta, geometry_hash = prepare_geometry(root, output, sample, config)
    folder.mkdir(exist_ok=True)
    atomic_json(folder/'geometry_receipt.json', dict(source_md5=sample['md5'],
                geometry_sha256=geometry_hash, metadata=slide_meta,
                coarse_patches=len(grid['coarse_coords']), fine_patches=len(grid['fine_coords']),
                feature_uncompressed_bytes=(len(grid['coarse_coords'])+len(grid['fine_coords']))*4096))
    meta = dict(schema='KT2028-TCGA-dual-scale-1', protocol=protocol,
                source_md5=sample['md5'], source_size=sample['size'],
                config_sha256=config_hash, weight_sha256=encoder.checkpoint_sha256,
                geometry_sha256=geometry_hash, original_uni_frozen=True,
                inference_precision='float32', feature_dimension=1024,
                coords_definition='level0 top-left floor; exact bbox_um authoritative')
    with LocalSVS(dest, config['maximum_native_upsampling_fraction']) as reader:
        coarse_geo = dict(bbox=grid['coarse_bbox_um'], coords=grid['coarse_coords'],
                          fine_counts=grid['fine_counts'], fine_ptr=grid['fine_ptr'])
        fine_geo = dict(bbox=grid['fine_bbox_um'], coords=grid['fine_coords'],
                        parent_indices=grid['fine_parent_indices'], relative_xy=grid['fine_relative_xy'])
        coarse = write_or_resume(folder/'coarse.zarr', coarse_geo,
                    dict(meta, mpp_um=config['coarse_mpp_um']), 224, config['coarse_mpp_um'],
                    reader, encoder, config['batch_size'], config['minimum_free_disk_bytes'])
        fine = write_or_resume(folder/'fine.zarr', fine_geo,
                    dict(meta, mpp_um=config['fine_mpp_um']), 224, config['fine_mpp_um'],
                    reader, encoder, config['batch_size'], config['minimum_free_disk_bytes'])
    receipt = dict(meta, status='completed', file_id=sample['id'],
                   patient_ids=sample['patient_ids'], project=sample['project'],
                   slide_metadata=slide_meta, coarse=coarse, fine=fine,
                   elapsed_seconds=round(time.monotonic()-started, 3), no_predictor_training=True)
    atomic_json(complete, receipt)
    verify_output(folder, receipt)
    return receipt


def verify_output(folder, receipt):
    if sha256_file(folder/'geometry.npz') != receipt['geometry_sha256']:
        raise ValueError('Saved geometry checksum mismatch')
    for name in ['coarse', 'fine']:
        group = zarr.open_group(str(folder/(name+'.zarr')), mode='r')
        h = hashlib.sha256()
        if group.attrs['status'] != 'completed' or group['features'].shape != (receipt[name]['rows'], 1024):
            raise ValueError('Incomplete feature store')
        for offset in range(0, receipt[name]['rows'], 256):
            value = np.asarray(group['features'][offset:offset+256])
            if not np.isfinite(value).all():
                raise ValueError('Nonfinite stored feature')
            h.update(value.tobytes())
        if h.hexdigest() != receipt[name]['features_sha256']:
            raise ValueError('Stored feature checksum mismatch')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['stage', 'inspect', 'extract', 'watch', 'status', 'pack', 'retire'])
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--config', type=Path)
    ap.add_argument('--checkpoint', type=Path)
    ap.add_argument('--file-id')
    ap.add_argument('--max-slides', type=int)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--poll-seconds', type=int, default=30)
    ap.add_argument('--backup-sha256')
    args = ap.parse_args()
    samples = json.loads(args.manifest.read_text(encoding='utf-8-sig'))
    if len({s['id'] for s in samples}) != len(samples):
        raise ValueError('Duplicate file UUID')
    if args.file_id:
        samples = [s for s in samples if s['id'] == args.file_id]
        if len(samples) != 1:
            raise ValueError('File UUID absent from manifest')
    for sample in samples:
        source_path(args.root, sample)
    if args.action == 'status':
        ready = []
        for sample in samples:
            folder = args.root/'features'/sample['id']
            p = folder/'receipt.json'
            if p.exists():
                r = json.loads(p.read_text())
                if r['status'] == 'completed':
                    ready.append(dict(id=sample['id'], coarse_patches=r['coarse']['rows'],
                                      fine_patches=r['fine']['rows'],
                                      output_bytes=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())))
        incoming = sum(p.stat().st_size for p in (args.root/'incoming').glob('*') if p.is_file())
        print(json.dumps(dict(completed=ready, staged_bytes=incoming,
                              free_bytes=shutil.disk_usage(args.root).free)), flush=True)
        return
    if args.action in ['pack', 'retire']:
        if len(samples) != 1 or not (args.root/'_TASK_OWNED_W059').is_file():
            raise ValueError('Archive/retire needs one UUID and task ownership marker')
        sample = samples[0]
        folder = args.root/'features'/sample['id']
        receipt = json.loads((folder/'receipt.json').read_text())
        verify_output(folder, receipt)
        if receipt['source_md5'] != sample['md5']:
            raise ValueError('Source/feature fingerprint mismatch')
        archives = args.root/'archives'; archives.mkdir(exist_ok=True)
        archive = archives/(sample['id']+'.tar.gz')
        if args.action == 'pack':
            if not archive.exists():
                if shutil.disk_usage(args.root).free < 32*1024**3:
                    raise OSError('Insufficient archival headroom')
                tmp = archive.with_suffix('.gz.partial')
                with tarfile.open(tmp, 'w:gz', compresslevel=1) as tar:
                    tar.add(folder, arcname=sample['id'])
                tmp.rename(archive)
            print(json.dumps(dict(id=sample['id'], sha256=sha256_file(archive),
                                  bytes=archive.stat().st_size)), flush=True)
        else:
            if not args.backup_sha256 or sha256_file(archive) != args.backup_sha256:
                raise ValueError('Require verified off-server backup SHA before retire')
            # Only the newly uploaded regular working copy under the exact task root.
            source = source_path(args.root, sample)
            if source.exists():
                if source.is_symlink() or source.resolve().parent != (args.root/'incoming').resolve():
                    raise ValueError('Unsafe working-copy path')
                if source.stat().st_size != sample['size'] or md5_file(source) != sample['md5']:
                    raise ValueError('Working copy changed; preserve it')
                source.unlink()
            atomic_json(args.root/'staged'/(sample['id']+'.retired.json'),
                        dict(source_md5=sample['md5'], backup_sha256=args.backup_sha256,
                             local_original_preserved_by_transfer_client=True, at_unix=time.time()))
            # Archive is also a regenerable task-owned temporary; features stay on server.
            if archive.is_symlink() or archive.resolve().parent != archives.resolve():
                raise ValueError('Unsafe archive path')
            archive.unlink()
            print(json.dumps(dict(id=sample['id'], retired=True)), flush=True)
        return
    if args.action == 'stage':
        for sample in samples:
            print(json.dumps(stage(args.root, sample)), flush=True)
        return
    config = json.loads(args.config.read_text())
    if (config['coarse_mpp_um'], config['fine_mpp_um'], config['input_pixels']) != (2, 0.5, 224):
        raise ValueError('TCGA scale contract mismatch')
    args.root.mkdir(parents=True, exist_ok=True)
    output = args.root/'features'; output.mkdir(exist_ok=True)
    if args.action == 'inspect':
        for sample in samples:
            if not (args.root/'staged'/(sample['id']+'.json')).exists():
                continue
            grid, meta, gh = prepare_geometry(args.root, output, sample, config)
            estimate = dict(file_id=sample['id'], metadata=meta, geometry_sha256=gh,
                            coarse_patches=len(grid['coarse_coords']), fine_patches=len(grid['fine_coords']),
                            feature_uncompressed_bytes=(len(grid['coarse_coords'])+len(grid['fine_coords']))*4096)
            atomic_json(output/sample['id']/'geometry_receipt.json', estimate)
            print(json.dumps(estimate), flush=True)
        return
    import fcntl
    lock_handle = (args.root/'extract.lock').open('a')
    fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config_hash = sha256_file(args.config)
    protocol = dict(config_sha256=config_hash, manifest_sha256=sha256_file(args.manifest),
                    code_sha256={n: sha256_file(Path(__file__).with_name(n)) for n in
                    ['prepare_tcga.py', 'local_svs.py', 'dual_scale_geometry.py', 'regions.py',
                     'uni_encoder.py', 'extract_dual_scale.py']})
    existing = args.root/'run.json'
    if existing.exists():
        if json.loads(existing.read_text())['protocol'] != protocol:
            raise ValueError('Existing TCGA run code/config/manifest mismatch')
    else:
        atomic_json(existing, dict(protocol=protocol, source_commit=subprocess.check_output(
                    ['git', 'rev-parse', 'HEAD'], text=True).strip(), started_at_unix=time.time(),
                    scope='all available slides, fixed pretrained encoder; no outcome fitting'))
    encoder = OriginalUNI(args.checkpoint, config['uni'], args.device)
    encoder.torch.set_num_threads(4)
    zarr.config.set({'async.concurrency': 1, 'threading.max_workers': 1})
    completed, failed = {}, {}
    while True:
        for sample in samples:
            sid = sample['id']
            if sid in completed or sid in failed or not (args.root/'staged'/(sid+'.json')).exists():
                continue
            if not source_path(args.root, sample).exists():
                continue
            if shutil.disk_usage(args.root).free < config['minimum_free_disk_bytes']:
                print(json.dumps(dict(event='waiting_for_disk_space')), flush=True)
                break
            try:
                print(json.dumps(dict(event='starting_slide', file_id=sid)), flush=True)
                receipt = extract_one(args.root, output, sample, config, config_hash, encoder, protocol)
                completed[sid] = receipt
                print(json.dumps(dict(event='completed_slide', file_id=sid,
                            coarse_patches=receipt['coarse']['rows'], fine_patches=receipt['fine']['rows'],
                            seconds=receipt['elapsed_seconds'])), flush=True)
            except Exception as error:
                failed[sid] = dict(error=type(error).__name__+': '+str(error)[:500])
                (output/sid).mkdir(exist_ok=True)
                atomic_json(output/sid/'failure.json', failed[sid])
                print(json.dumps(dict(event='failed_slide', file_id=sid, **failed[sid])), flush=True)
            atomic_json(args.root/'summary.json', dict(status='running', expected=len(samples),
                        completed=len(completed), failed=failed,
                        coarse_patches=sum(r['coarse']['rows'] for r in completed.values()),
                        fine_patches=sum(r['fine']['rows'] for r in completed.values()),
                        completed_ids=list(completed), updated_at_unix=time.time()))
            if args.max_slides and len(completed) >= args.max_slides:
                return
        if args.action == 'extract' or len(completed)+len(failed) == len(samples) or (args.root/'STOP').exists():
            return
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
