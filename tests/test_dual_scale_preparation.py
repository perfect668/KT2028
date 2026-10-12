"""Software fixtures only; no simulated prognosis or experiment results."""
import asyncio
import io
import zipfile

import numpy as np
import pytest
import zarr
from zarr.abc.store import RangeByteRequest, SuffixByteRequest
from zarr.core.buffer import default_buffer_prototype

from kt2028.remote_pyramid import RangeZipStore
from kt2028.dual_scale_geometry import build_nested_grid
from kt2028.uni_encoder import verify_checkpoint


def zip_bytes(entries, compression=zipfile.ZIP_STORED):
    value = io.BytesIO()
    with zipfile.ZipFile(value, 'w', compression=compression) as archive:
        for key, payload in entries.items():
            archive.writestr(key, payload)
    value.seek(0)
    return value


def test_partial_zip_range_and_getsize_do_not_materialize_full_shard():
    raw = zip_bytes({'large': bytes(range(256))*1024})
    store = RangeZipStore(raw, max_value_bytes=32)
    prototype = default_buffer_prototype()
    async def check():
        assert await store.getsize('large') == 256*1024
        value = await store.get('large', prototype, RangeByteRequest(200, 220))
        assert value.to_bytes() == bytes(range(200, 220))
        value = await store.get('large', prototype, SuffixByteRequest(8))
        assert value.to_bytes() == bytes(range(248, 256))
        assert await store.get('missing', prototype) is None
        with pytest.raises(ValueError, match='bounded ROI'):
            await store.get('large', prototype)
        assert store.complete_crc_checks == 0 and store.partial_requests == 2
    asyncio.run(check())
    store.close()


def test_outer_deflate_is_rejected_for_partial_roi():
    store = RangeZipStore(zip_bytes({'data': b'abcdef'}, zipfile.ZIP_DEFLATED))
    with pytest.raises(ValueError, match='ZIP_STORED'):
        asyncio.run(store.get('data', default_buffer_prototype(), RangeByteRequest(0, 2)))
    store.close()


def test_zarr_native_shard_decoder_uses_partial_zip_store(tmp_path):
    root = tmp_path / 'native'
    original = np.arange(64*64, dtype=np.uint16).reshape(64, 64)
    array = zarr.create_array(store=str(root), shape=(64, 64), chunks=(16, 16),
                              shards=(64, 64), dtype='uint16')
    array[:] = original
    raw = zip_bytes({p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()})
    store = RangeZipStore(raw)
    remote = zarr.open_array(store=store, mode='r')
    np.testing.assert_array_equal(remote[18:25, 35:42], original[18:25, 35:42])
    assert store.partial_requests > 0
    store.close()


def test_new_grid_has_unique_variable_children_and_preserves_edge_geometry():
    # Synthetic mask fixture, not a patient image or generated feature cache.
    rgb = np.full((460, 600, 3), [120, 60, 130], dtype=np.uint8)
    config = dict(input_pixels=224, coarse_mpp_um=2.0, fine_mpp_um=0.5,
                  minimum_tissue_fraction=0.01, tissue_channel_spread_min=10,
                  tissue_mean_min=20, tissue_mean_max=245)
    grid = build_nested_grid(rgb, [600, 460], [1, 1], [0.1, 0.1], config)
    assert len(grid['coarse_bbox_um']) == 4
    assert grid['fine_counts'].tolist() == [16, 8, 4, 2]
    assert len(set(map(tuple, grid['fine_coords']))) == 30
    np.testing.assert_array_equal(grid['fine_ptr'], [0, 16, 24, 28, 30])
    assert grid['fine_image_valid_fraction'].min() < 1
    parents = grid['coarse_bbox_um'][grid['fine_parent_indices']]
    assert np.all(grid['fine_bbox_um'][:, :2] >= parents[:, :2])
    assert np.all(grid['fine_bbox_um'][:, 2:] <= parents[:, 2:])


def test_missing_or_mismatched_uni_never_allows_random_features(tmp_path):
    spec = {'size_bytes': 12, 'sha256': 'not-real'}
    with pytest.raises(FileNotFoundError):
        verify_checkpoint(tmp_path/'missing.bin', spec)
    bad = tmp_path/'wrong.bin'
    bad.write_bytes(b'not-a-model')
    with pytest.raises(ValueError):
        verify_checkpoint(bad, spec)
