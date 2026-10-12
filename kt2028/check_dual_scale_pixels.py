"""Real-image physical ROI engineering check; no predictor or UNI required."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import zarr
from PIL import Image, ImageDraw

from .remote_pyramid import RemotePyramid


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--samples', type=Path, required=True)
    ap.add_argument('--slide-id', required=True)
    ap.add_argument('--xy-um', type=float, nargs=2, required=True)
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--network-cap-mib', type=int, default=64)
    args = ap.parse_args()
    samples = json.loads(args.samples.read_text())
    matches = [s for s in samples if s['slide_id'] == args.slide_id]
    if len(matches) != 1:
        raise ValueError('Source slide mapping must be unique')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    zarr.config.set({'async.concurrency': 1, 'threading.max_workers': 1})
    receipt = {'scope': 'real physical nested ROI engineering; no features/training/labels',
               'status': 'started', 'xy_um': args.xy_um, 'input_pixels': 224,
               'coarse_mpp_um': 2.0, 'fine_mpp_um': 0.5}
    receipt['source_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    receipt['source_code_sha256'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                     for name in ['check_dual_scale_pixels.py', 'remote_pyramid.py', 'remote_rgb.py']}
    try:
        with RemotePyramid(matches[0]['source'], args.cache, args.network_cap_mib * 1024**2) as reader:
            x, y = args.xy_um
            if x + 448 > reader.image_wh_um[0] or y + 448 > reader.image_wh_um[1]:
                raise ValueError('Use a full parent ROI for this nested-grid check')
            coarse = reader.read_patch(x, y, 2.0)
            coarse.save(args.output / 'coarse_2mpp.png')
            grid = coarse.copy()
            draw = ImageDraw.Draw(grid)
            for i in range(1, 4):
                draw.line([(56*i, 0), (56*i, 223)], fill=(0, 150, 0))
                draw.line([(0, 56*i), (223, 56*i)], fill=(0, 150, 0))
            grid.save(args.output / 'coarse_nested_grid.png')
            mosaic = Image.new('RGB', (896, 896), 'white')
            for row in range(4):
                for col in range(4):
                    fine = reader.read_patch(x + 112*col, y + 112*row, 0.5)
                    fine.save(args.output / f'fine_{row}_{col}_0p5mpp.png')
                    mosaic.paste(fine, (224*col, 224*row))
            mosaic.save(args.output / 'fine_same_field_mosaic.png')
            mosaic.resize((224, 224), Image.Resampling.BILINEAR).save(args.output / 'fine_mosaic_at_coarse_size.png')
            receipt.update(reader.receipt(), status='pixels_read', fine_patches=16,
                           coarse_level=reader.select_level(2.0), fine_level=reader.select_level(0.5),
                           parent_field_um=448.0, child_field_um=112.0,
                           visual_registration_checked=False)
    except Exception as e:
        receipt.update(status='failed', error=type(e).__name__ + ': ' + str(e)[:400])
        raise
    finally:
        receipt['elapsed_seconds'] = round(time.monotonic() - started, 3)
        receipt['image_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in args.output.glob('*.png')}
        (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
