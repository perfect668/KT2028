"""Encode existing physical coarse RGB with frozen UNI; no network image reads."""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import zarr
from PIL import Image

from .regions import crop_region
from .uni_encoder import OriginalUNI, sha256_file


def write_json(path, value):
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def prepare_case(slide, legacy, output, config, config_sha, encoder):
    sid = slide['slide_id']
    if Path(sid).name != sid:
        raise ValueError('Unsafe slide identifier')
    old = legacy / sid
    prior = json.loads((old / 'receipt.json').read_text())
    region_sha = sha256_file(old / 'regions.npz')
    if prior['status'] != 'completed' or prior['slide_id'] != sid or region_sha != prior['regions_sha256']:
        raise ValueError('Legacy region identity/hash mismatch')
    if prior['config_sha256'] != config['legacy_geometry_config_sha256']:
        raise ValueError('Legacy geometry protocol differs')
    if Path(prior['fine_path']) != Path(slide['features_path']):
        raise ValueError('Fine source differs from official manifest')
    rgb_sha = sha256_file(old / 'low_rgb.png')
    identity = dict(schema='KT2028-R2-coarse-UNI', config_sha256=config_sha,
                    rgb_sha256=rgb_sha, original_regions_sha256=region_sha,
                    original_receipt_sha256=sha256_file(old / 'receipt.json'),
                    weight_sha256=encoder.checkpoint_sha256, fine_path=prior['fine_path'],
                    fine_coords_sha256=prior['fine_coords_sha256'])
    folder = output / sid
    receipt_path = folder / 'receipt.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if any(receipt.get(k) != v for k, v in identity.items()) or receipt['status'] != 'completed':
            raise ValueError('Preserve different existing encoding')
        if sha256_file(folder / 'regions.npz') != receipt['regions_sha256']:
            raise ValueError('Existing coarse asset hash mismatch')
        return receipt
    folder.mkdir(exist_ok=True)
    owner = folder / 'owner.json'
    if owner.exists() and json.loads(owner.read_text()) != identity:
        raise ValueError('Preserve unrelated incomplete folder')
    write_json(owner, identity)
    with np.load(old / 'regions.npz', allow_pickle=False) as source:
        arrays = {k: source[k].copy() for k in source.files if k != 'coarse'}
    boxes, counts, ptr, indices = (arrays[k] for k in ('bbox_um', 'fine_counts', 'fine_ptr', 'fine_indices'))
    if len(ptr) != len(boxes)+1 or ptr[-1] != len(indices) or not np.array_equal(np.diff(ptr), counts):
        raise ValueError('Invalid fine CSR')
    if len(np.unique(indices)) != len(indices):
        raise ValueError('Duplicate fine ownership')
    fine = zarr.open_group(prior['fine_path'], mode='r')
    coords = np.asarray(fine['coords'][:])
    import hashlib
    if hashlib.sha256(coords.tobytes()).hexdigest() != prior['fine_coords_sha256']:
        raise ValueError('Official fine coordinates changed')
    if fine['features'].shape != (len(coords), 1024) or np.any(indices < 0) or np.any(indices >= len(coords)):
        raise ValueError('Invalid official fine shape/index')
    access = prior['image_access']
    mpp = access['selected_level']['mpp_xy']
    if max(mpp) > config['coarse_mpp_um']:
        raise ValueError('Cached RGB is coarser than requested pixels')
    with Image.open(old / 'low_rgb.png') as image:
        rgb = image.convert('RGB')
    if list(rgb.size) != [access['selected_level']['shape'][-1], access['selected_level']['shape'][-2]]:
        raise ValueError('Cached RGB source size mismatch')
    values = []
    for i in range(0, len(boxes), config['batch_size']):
        patches = [crop_region(rgb, {'bbox_um': b}, mpp, prior['config'])
                   for b in boxes[i:i+config['batch_size']]]
        values.append(encoder.encode(patches))
    arrays['coarse'] = np.concatenate(values).astype(np.float32)
    l0_mpp = np.asarray(access['l0']['mpp_xy'])
    selected_coords = coords[indices].astype(np.float64)
    full_wh = np.asarray(access['l0']['shape'][-2:][::-1])
    sizes = np.maximum(np.minimum(selected_coords + arrays['fine_pixel_size_xy'], full_wh)-selected_coords, 0)
    centers = (selected_coords + sizes/2)*l0_mpp
    parents = np.repeat(np.arange(len(boxes)), counts)
    relative = (centers-boxes[parents, :2])/config['coarse_field_um']
    if np.any(relative < -1e-5) or np.any(relative > 1+1e-5):
        raise ValueError('Fine centers do not match assigned coarse cells')
    arrays['fine_relative_xy'] = relative.astype(np.float32)
    temp = folder / 'regions.partial.npz'
    np.savez_compressed(temp, **arrays)
    with np.load(temp, allow_pickle=False) as saved:
        for key, value in arrays.items():
            if not np.array_equal(value, saved[key]):
                raise ValueError('Coarse save/reload differs')
    temp.replace(folder / 'regions.npz')
    receipt = dict(identity, status='completed', regions_sha256=sha256_file(folder / 'regions.npz'),
                   coarse_shape=list(arrays['coarse'].shape), fine_patch_count=len(coords),
                   fine_assigned_count=len(indices), coarse_mpp_um=4.0, fine_mpp_um=1.0,
                   fine_field_um=224.0, coarse_field_um=896.0, input_pixels=224,
                   image_network_bytes=0, original_uni_frozen=True, no_predictor_training=True)
    write_json(receipt_path, receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', action='append', required=True, type=Path)
    p.add_argument('--legacy-regions', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--max-cases', type=int)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    if (config['coarse_mpp_um'], config['fine_mpp_um'], config['input_pixels']) != (4.0, 1.0, 224):
        raise ValueError('This input contract is 4/1MPP; do not mix extraction runs')
    slides, case_keys = [], []
    for path in args.manifest:
        manifest = json.loads(path.read_text())
        if manifest['partition'] not in ('train', 'validate') or manifest['endpoint'] != 'five_year_death':
            raise ValueError('Only registered Train/Validation data preparation; no Test')
        for row in manifest['patients']:
            if len(row['slides']) != 1:
                raise ValueError('Multi-slide aggregation requires its own contract')
            case_keys.append(row['canonical_case_key'])
            slides.extend(row['slides'])
    if len(case_keys) != len(set(case_keys)) or len({s['slide_id'] for s in slides}) != len(slides):
        raise ValueError('Duplicate cases/slides or cross-partition overlap')
    if args.max_cases is not None:
        if args.max_cases <= 0:
            raise ValueError('Positive case limit required')
        slides = slides[:args.max_cases]
    encoder = OriginalUNI(args.checkpoint, config['uni'], args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = dict(config_sha256=sha256_file(args.config),
                    manifest_shas=[sha256_file(x) for x in args.manifest],
                    source_shas={n: sha256_file(Path(__file__).with_name(n)) for n in
                                 ('prepare_coarse_uni.py', 'uni_encoder.py', 'regions.py')})
    run = args.output/'run.json'
    if run.exists():
        previous = json.loads(run.read_text())
        if any(previous.get(k) != v for k, v in protocol.items()):
            raise ValueError('Encoding code/manifest/config differs from existing run')
    else:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        write_json(run, dict(protocol, source_commit=commit, scope='real coarse UNI data preparation only'))
    records = []
    for slide in slides:
        records.append(prepare_case(slide, args.legacy_regions, args.output, config,
                                    protocol['config_sha256'], encoder))
        summary = dict(status='running', expected=len(slides), completed=len(records),
                       coarse_regions=sum(r['coarse_shape'][0] for r in records), image_network_bytes=0)
        write_json(args.output/'summary.json', summary)
        print(json.dumps(summary), flush=True)
    write_json(args.output/'summary.json', dict(summary, status='completed'))


if __name__ == '__main__':
    main()
