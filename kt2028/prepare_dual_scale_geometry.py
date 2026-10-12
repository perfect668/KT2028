"""Prepare real nested geometry from existing source-matched low RGB.

This does not read old UNI features, fabricate new embeddings, or use outcomes.
Each receipt explicitly says geometry_only; new RGB / UNI extraction follows.
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .dual_scale_geometry import build_nested_grid
from .uni_encoder import sha256_file


def prepare_one(sample, config, config_sha, legacy_root, output):
    sid = sample['slide_id']
    if Path(sid).name != sid or sid in ('.', '..'):
        raise ValueError('Invalid slide identifier')
    source = sample['source']
    if Path(source['path']).name != sid + '.ome.zarr.zip':
        raise ValueError('Source filename / slide identity mismatch')
    old = legacy_root / sid
    prior = json.loads((old / 'receipt.json').read_text())
    access = prior['image_access']
    if prior['slide_id'] != sid or access['source_zip_bytes'] != source['size']:
        raise ValueError('Existing RGB source identity / size mismatch')
    if Path(access['source_url']).name != sid + '.ome.zarr.zip':
        raise ValueError('Existing RGB belongs to different source image')
    selected = access['selected_level']
    image_sha = sha256_file(old / 'low_rgb.png')
    identity = dict(config_sha256=config_sha, mask_rgb_sha256=image_sha,
                    root_metadata_sha256=access.get('root_metadata_sha256'),
                    source_zip_bytes=source['size'])
    folder = output / sid
    receipt_path = folder / 'receipt.json'
    if receipt_path.exists():
        completed = json.loads(receipt_path.read_text())
        if completed.get('status') != 'geometry_only_prepared':
            raise ValueError('Unexpected geometry receipt state')
        if any(completed.get(k) != v for k, v in identity.items()):
            raise ValueError('Existing geometry protocol/source changed; preserve it')
        if sha256_file(folder / 'geometry.npz') != completed['geometry_sha256']:
            raise ValueError('Existing geometry hash mismatch')
        return completed
    if folder.exists():
        # Do not silently overwrite an interrupted or user-owned folder.
        raise FileExistsError('Unfinished geometry folder requires explicit inspection')
    image = Image.open(old / 'low_rgb.png').convert('RGB')
    if list(image.size) != [selected['shape'][-1], selected['shape'][-2]]:
        raise ValueError('Existing low RGB dimensions differ from source receipt')
    l0 = access['l0']
    wh = [l0['shape'][-1] * l0['mpp_xy'][0], l0['shape'][-2] * l0['mpp_xy'][1]]
    grid = build_nested_grid(np.asarray(image), wh, selected['mpp_xy'], l0['mpp_xy'], config)
    folder.mkdir()
    with (folder / 'geometry.npz').open('xb') as f:
        np.savez_compressed(f, **grid)
    counts = grid['fine_counts']
    receipt = dict(identity, status='geometry_only_prepared', slide_id=sid,
                   coarse_regions=len(grid['coarse_bbox_um']),
                   fine_patches=len(grid['fine_bbox_um']),
                   nonempty_fine_regions=int((counts > 0).sum()),
                   fine_count_min=int(counts.min()), fine_count_max=int(counts.max()),
                   coarse_mpp_um=config['coarse_mpp_um'], fine_mpp_um=config['fine_mpp_um'],
                   input_pixels=config['input_pixels'], physical_origin_um=[0, 0],
                   coords_definition='floor of physical top-left / OME level-0 MPP; bbox_um retains exact physical positions',
                   geometry_sha256=sha256_file(folder / 'geometry.npz'),
                   mask_scope='Existing low RGB only; tissue/background; not prognosis or tumor selection',
                   original_fine_coordinates_used=False, new_features_generated=False,
                   source_mapping_scope='filename, source size, existing source receipt; not clinical patient identity validation')
    receipt['historical_root_metadata_hash_available'] = identity['root_metadata_sha256'] is not None
    with receipt_path.open('x') as f:
        json.dump(receipt, f, indent=2)
    return receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--samples', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--legacy-regions', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    samples = json.loads(args.samples.read_text())
    if len({s['slide_id'] for s in samples}) != len(samples):
        raise ValueError('Duplicate source slides')
    config = json.loads(args.config.read_text())
    config_sha = sha256_file(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    rows, errors = [], []
    started = time.monotonic()
    for sample in samples:
        try:
            if shutil.disk_usage(args.output).free < config['minimum_free_disk_bytes']:
                raise OSError('Insufficient output disk headroom')
            row = prepare_one(sample, config, config_sha, args.legacy_regions, args.output)
            rows.append(row)
            print('GEOMETRY_READY ' + sample['slide_id'], flush=True)
        except Exception as e:
            errors.append({'slide_id': sample['slide_id'], 'error': type(e).__name__ + ': ' + str(e)[:300]})
        summary = {'scope': 'geometry only; no high-layer whole-cohort download / UNI features / training / labels',
                   'config_sha256': config_sha, 'expected': len(samples), 'completed': len(rows),
                   'errors': errors, 'coarse_regions': sum(r['coarse_regions'] for r in rows),
                   'fine_patches': sum(r['fine_patches'] for r in rows),
                   'elapsed_seconds': round(time.monotonic()-started, 3)}
        temp = args.output / 'summary.partial'
        temp.write_text(json.dumps(summary, indent=2))
        temp.replace(args.output / 'summary.json')
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
