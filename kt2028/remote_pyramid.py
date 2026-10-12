"""Read bounded physical ROIs from official stored ZIP / sharded OME-Zarr.

Only ZIP addressing is adapted here. Zarr's codecs decode shards and validate
their index CRC32C. Partial reads cannot validate a whole ZIP-member CRC; receipts
explicitly distinguish that from the complete metadata-member CRC checks.
"""
import asyncio
import hashlib
import json
import math
import struct
import threading
import zipfile

import numpy as np
from PIL import Image
import zarr
from zarr.abc.store import Store, RangeByteRequest, OffsetByteRequest, SuffixByteRequest

from .remote_rgb import CachedRangeFile


class RangeZipStore(Store):
    """Read-only Zarr store using offsets into uncompressed outer ZIP members."""
    supports_writes = False
    supports_deletes = False
    supports_listing = True

    def __init__(self, fileobj, max_value_bytes=64 * 1024**2):
        super().__init__(read_only=True)
        self.raw = fileobj
        self.archive = zipfile.ZipFile(fileobj)
        members = [i for i in self.archive.infolist() if not i.is_dir()]
        self.members = {i.filename: i for i in members}
        if len(self.members) != len(members):
            raise ValueError('Duplicate ZIP members')
        if any(k.startswith('/') or '..' in k.split('/') for k in self.members):
            raise ValueError('Unsafe ZIP member')
        self.max_value_bytes = int(max_value_bytes)
        self.lock = threading.RLock()
        self.offsets = {}
        self.partial_requests = 0
        self.complete_crc_checks = 0

    def __eq__(self, other):
        return self is other

    def _bounds(self, size, request):
        if request is None:
            return 0, size
        if isinstance(request, RangeByteRequest):
            start, end = request.start, min(request.end, size)
        elif isinstance(request, OffsetByteRequest):
            start, end = request.offset, size
        elif isinstance(request, SuffixByteRequest):
            if request.suffix < 0:
                raise ValueError('Negative suffix')
            start, end = max(0, size - request.suffix), size
        else:
            raise TypeError('Unknown Zarr byte request')
        if start < 0 or end < start or start > size:
            raise ValueError('Invalid ZIP byte request')
        return start, end

    def _get_bytes(self, key, request):
        with self.lock:
            info = self.members.get(key)
            if info is None:
                return None
            start, end = self._bounds(info.file_size, request)
            if end - start > self.max_value_bytes:
                raise ValueError('Whole-shard/value read exceeds bounded ROI contract')
            if request is None:
                # Metadata and genuinely small complete values retain ZIP CRC validation.
                data = self.archive.read(info)
                self.complete_crc_checks += 1
                return data
            if info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 1:
                raise ValueError('Partial access requires unencrypted ZIP_STORED members')
            if key not in self.offsets:
                self.raw.seek(info.header_offset)
                header = self.raw.read(30)
                if len(header) != 30:
                    raise ValueError('Short ZIP local header')
                values = struct.unpack('<4s5H3I2H', header)
                if values[0] != b'PK\x03\x04' or values[3] != zipfile.ZIP_STORED:
                    raise ValueError('Invalid stored ZIP local header')
                self.offsets[key] = info.header_offset + 30 + values[-2] + values[-1]
            if end == start:
                return b''
            self.raw.seek(self.offsets[key] + start)
            data = self.raw.read(end - start)
            if len(data) != end - start:
                raise ValueError('Short ZIP member byte range')
            self.partial_requests += 1
            return data

    async def get(self, key, prototype, byte_range=None):
        data = await asyncio.to_thread(self._get_bytes, key, byte_range)
        return None if data is None else prototype.buffer.from_bytes(data)

    async def get_partial_values(self, prototype, key_ranges):
        return [await self.get(key, prototype, request) for key, request in key_ranges]

    async def getsize(self, key):
        if key not in self.members:
            raise FileNotFoundError(key)
        return self.members[key].file_size

    async def exists(self, key):
        return key in self.members

    async def list(self):
        for key in self.members:
            yield key

    async def list_prefix(self, prefix):
        for key in self.members:
            if key.startswith(prefix):
                yield key

    async def list_dir(self, prefix):
        prefix = prefix.rstrip('/')
        base = prefix + '/' if prefix else ''
        seen = set()
        for key in self.members:
            if key.startswith(base):
                part = key[len(base):].split('/')[0]
                if part and part not in seen:
                    seen.add(part)
                    yield part

    async def set(self, *args, **kwargs):
        raise PermissionError('Read-only official source')

    async def delete(self, *args, **kwargs):
        raise PermissionError('Read-only official source')

    def close(self):
        self.archive.close()
        super().close()


class RemotePyramid:
    """Explicit micrometer coordinates; no objective-power / MPP conversion."""
    def __init__(self, source, cache_root, cap_bytes, max_roi_pixels=4_000_000):
        canonical = 'https://www.ebi.ac.uk/biostudies/files/S-BIAD1285/' + source['path']
        url = source.get('url', canonical)
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if parsed.scheme != 'https' or parsed.hostname not in ('www.ebi.ac.uk', 'ftp.ebi.ac.uk'):
            raise ValueError('Only official HTTPS image sources accepted')
        self.raw = CachedRangeFile(url, source['size'], cache_root, cap_bytes,
                                   canonical, source.get('cache_only', False))
        self.store = None
        try:
            self.store = RangeZipStore(self.raw)
            roots = [k[:-len('/zarr.json')] for k in self.store.members
                     if k.count('/') == 1 and k.endswith('/zarr.json')]
            if len(roots) != 1:
                raise ValueError('Ambiguous OME root')
            self.root = roots[0]
            raw_meta = self.store._get_bytes(self.root + '/zarr.json', None)
            self.root_sha = hashlib.sha256(raw_meta).hexdigest()
            meta = json.loads(raw_meta)
            mult = meta['attributes'].get('ome', meta['attributes'])['multiscales']
            if len(mult) != 1:
                raise ValueError('Multiple OME images need explicit source mapping')
            mult = mult[0]
            if [a['name'] for a in mult['axes']] != ['t', 'c', 'z', 'y', 'x']:
                raise ValueError('Unsupported image axes')
            if any(a.get('unit') not in ('micrometer', 'µm') for a in mult['axes'][-2:]):
                raise ValueError('Unverified spatial units')
            # Sources checked so far have scale only and zero physical origin.
            # Reject translations rather than silently changing registration.
            if mult.get('coordinateTransformations'):
                raise ValueError('Global OME transforms require explicit handling')
            self.levels = []
            self.arrays = {}
            for dataset in mult['datasets']:
                transforms = dataset['coordinateTransformations']
                if len(transforms) != 1 or transforms[0]['type'] != 'scale':
                    raise ValueError('Only explicitly verified scale-only pyramids supported')
                scale = transforms[0]['scale']
                am = json.loads(self.store._get_bytes(self.root + '/' + dataset['path'] + '/zarr.json', None))
                if am['shape'][:3] != [1, 3, 1] or am['data_type'] != 'uint8':
                    raise ValueError('Unsupported RGB image layout')
                mpp = [float(scale[-1]), float(scale[-2])]
                if not all(math.isfinite(v) and v > 0 for v in mpp):
                    raise ValueError('Invalid spatial scale')
                self.levels.append({'path': dataset['path'], 'shape': am['shape'], 'mpp_xy': mpp})
            self.max_roi_pixels = int(max_roi_pixels)
            self.l0 = min(self.levels, key=lambda l: max(l['mpp_xy']))
            self.image_wh_um = [self.l0['shape'][-1] * self.l0['mpp_xy'][0],
                                self.l0['shape'][-2] * self.l0['mpp_xy'][1]]
        except BaseException:
            self.close()
            raise

    def select_level(self, target_mpp):
        if not math.isfinite(target_mpp) or target_mpp <= 0:
            raise ValueError('Invalid target MPP')
        valid = [l for l in self.levels if max(l['mpp_xy']) <= target_mpp]
        if not valid:
            raise ValueError('Native image is coarser than requested target')
        return max(valid, key=lambda l: max(l['mpp_xy']))

    def read_patch(self, x_um, y_um, target_mpp, pixels=224):
        if not all(math.isfinite(v) and v >= 0 for v in (x_um, y_um)) or pixels <= 0:
            raise ValueError('Invalid physical ROI')
        if x_um >= self.image_wh_um[0] or y_um >= self.image_wh_um[1]:
            raise ValueError('ROI lies outside source image')
        level = self.select_level(target_mpp)
        mx, my = level['mpp_xy']
        field = pixels * target_mpp
        x0, y0 = math.floor(x_um / mx), math.floor(y_um / my)
        x1 = min(math.ceil((x_um + field) / mx), level['shape'][-1])
        y1 = min(math.ceil((y_um + field) / my), level['shape'][-2])
        if (x1 - x0) * (y1 - y0) > self.max_roi_pixels:
            raise ValueError('Decoded ROI memory cap exceeded')
        if level['path'] not in self.arrays:
            self.arrays[level['path']] = zarr.open_array(
                store=self.store, path=self.root + '/' + level['path'], mode='r')
        array = self.arrays[level['path']]
        rgb = np.transpose(np.asarray(array[0, :, 0, y0:y1, x0:x1]), (1, 2, 0))
        if rgb.dtype != np.uint8 or rgb.shape[-1] != 3:
            raise ValueError('Decoded RGB contract mismatch')
        extent = (x_um / mx - x0, y_um / my - y0,
                  (x_um + field) / mx - x0, (y_um + field) / my - y0)
        return Image.fromarray(rgb).transform(
            (pixels, pixels), Image.Transform.EXTENT, extent,
            resample=Image.Resampling.BILINEAR, fillcolor=(255, 255, 255))

    def receipt(self):
        return {'source_url': self.raw.url, 'source_zip_bytes': self.raw.size,
                'root_metadata_sha256': self.root_sha, 'levels': self.levels,
                'image_wh_um': self.image_wh_um, 'physical_origin_um': [0, 0],
                'network_bytes': self.raw.bytes_received, 'http_requests': self.raw.requests_made,
                'cache_hits': self.raw.cache_hits, 'partial_zip_requests': self.store.partial_requests,
                'complete_zip_crc_checks': self.store.complete_crc_checks,
                'partial_shard_zip_crc_verified': False,
                'integrity_scope': 'HTTP ranges/length; complete metadata ZIP CRC; native Zarr shard-index CRC32C',
                'pixel_backend': 'native Zarr codecs; partial ZIP_STORED shard ranges',
                'sampling': 'physical EXTENT; bilinear; white edge padding; no extra center crop'}

    def close(self):
        if self.store is not None:
            self.store.close()
        self.raw.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
