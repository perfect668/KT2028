"""Bounded, persistent HTTPS range cache for official OME-Zarr ZIPs."""
import hashlib
import io
import json
import threading
import time
import zipfile
import zlib
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
import zarr


class CachedRangeFile(io.RawIOBase):
    def __init__(self, url, size, cache_root, cap_bytes, cache_identity=None):
        self.url, self.size, self.cap = url, int(size), int(cap_bytes)
        self.pos = self.bytes_received = self.requests_made = self.cache_hits = 0
        identity = hashlib.sha256(f'{cache_identity or url}\n{size}'.encode()).hexdigest()
        self.cache = Path(cache_root) / identity
        self.cache.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers['Accept-Encoding'] = 'identity'
        self.lock = threading.RLock()
        self.started = time.monotonic()

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, n, whence=0):
        with self.lock:
            p = n if whence == 0 else self.pos+n if whence == 1 else self.size+n
            if p < 0: raise ValueError('Negative seek')
            self.pos = p
            return p

    def _fetch(self, start, count):
        path = self.cache / f'{start}_{count}.bin'
        if path.exists():
            value = path.read_bytes()
            if len(value) != count: raise ValueError('Incomplete cached range')
            self.cache_hits += 1
            return value
        if count > self.cap or self.bytes_received+count > self.cap:
            raise ValueError('Network byte cap exceeded; no full ZIP download allowed')
        for attempt in range(3):
            if time.monotonic()-self.started > 1200:
                raise TimeoutError('Per-image 20 minute deadline exceeded')
            try:
                self.requests_made += 1
                if count >= 1024**2: print(f'RANGE_FETCH bytes={count} attempt={attempt+1}',flush=True)
                with self.session.get(self.url, headers={'Range':f'bytes={start}-{start+count-1}'},
                                      timeout=(15,40), stream=True) as response:
                    if response.status_code in (429,500,502,503,504):
                        response.raise_for_status()
                    if response.status_code != 206: raise ValueError('HTTP Range not honored')
                    if response.headers.get('Content-Range') != f'bytes {start}-{start+count-1}/{self.size}':
                        raise ValueError('Source size/range changed')
                    chunks = []; received = 0
                    for block in response.iter_content(256*1024):
                        received += len(block); self.bytes_received += len(block)
                        if received > count or self.bytes_received > self.cap:
                            raise ValueError('Range response exceeds cap')
                        if time.monotonic()-self.started > 1200: raise TimeoutError('Image deadline exceeded')
                        chunks.append(block)
                    value = b''.join(chunks)
                    if len(value) != count: raise requests.exceptions.ChunkedEncodingError('Incomplete range')
                tmp = path.with_suffix('.partial'); tmp.write_bytes(value); tmp.replace(path)
                return value
            except requests.RequestException:
                if attempt == 2: raise
                time.sleep(attempt+1)

    def read(self, n=-1):
        with self.lock:
            n = min(n if n >= 0 else self.size-self.pos, self.size-self.pos)
            if n <= 0: return b''
            # Coalesce the numerous small ZIP/metadata reads; large members remain bounded.
            if n < 128*1024:
                start = self.pos//(128*1024)*(128*1024)
                count = min(((self.pos+n-start+131071)//131072)*131072, self.size-start)
                data = self._fetch(start,count)[self.pos-start:self.pos-start+n]
            else: data = self._fetch(self.pos,n)
            self.pos += n
            return data

    def close(self):
        self.session.close()
        super().close()


def read_low_rgb(source, config, cache_root):
    """Read a whole low-resolution RGB plane, never a whole WSI archive."""
    canonical = 'https://www.ebi.ac.uk/biostudies/files/S-BIAD1285/'+source['path']
    url = source.get('url',canonical)
    parsed=urlparse(url)
    if parsed.scheme!='https' or parsed.hostname not in ('www.ebi.ac.uk','ftp.ebi.ac.uk'):
        raise ValueError('Only verified official HTTPS sources accepted')
    raw = CachedRangeFile(url,source['size'],cache_root,config['max_network_bytes_per_image'],canonical)
    try:
        with zipfile.ZipFile(raw) as archive:
            roots = [n[:-len('/zarr.json')] for n in archive.namelist()
                     if n.count('/') == 1 and n.endswith('/zarr.json')]
            if len(roots) != 1: raise ValueError('Ambiguous OME root')
            root = roots[0]; root_bytes=archive.read(root+'/zarr.json')
            root_meta = json.loads(root_bytes)
            attrs = root_meta['attributes']; mult = attrs.get('ome',attrs)['multiscales'][0]
            axes = mult['axes']
            if [a['name'] for a in axes] != ['t','c','z','y','x']:
                raise ValueError('Unsupported axes')
            if any(a.get('unit') not in ('micrometer','µm') for a in axes[-2:]):
                raise ValueError('Unverified spatial units')
            consolidated=root_meta.get('consolidated_metadata',{}).get('metadata',{})
            levels=[];metas={}
            for d in mult['datasets']:
                meta=consolidated.get(d['path'])
                if meta is None:meta=json.loads(archive.read(root+'/'+d['path']+'/zarr.json'))
                transforms=d['coordinateTransformations']
                if any(t['type']!='scale' for t in transforms):
                    raise ValueError('Spatial translation requires explicit handling')
                scale=next(t['scale'] for t in transforms if t['type']=='scale')
                if meta['shape'][:3]!=[1,3,1] or meta['data_type']!='uint8':raise ValueError('Unsupported RGB layout')
                metas[d['path']]=meta
                levels.append({'path':d['path'],'shape':meta['shape'],'mpp_xy':[scale[-1],scale[-2]]})
            fine_enough=[l for l in levels if max(l['mpp_xy'])<=config['target_coarse_mpp_um']]
            if not fine_enough:raise ValueError('No suitable native resolution')
            selected=max(fine_enough,key=lambda l:max(l['mpp_xy']))
            if np.prod(selected['shape'][-2:])>config['max_low_rgb_pixels']:raise ValueError('Low RGB memory cap exceeded')
            # Materialize ONLY the chosen low array. Native Zarr reads local shard ranges,
            # avoiding ZipStore re-reading a large whole member for each inner chunk.
            target=raw.cache/'selected_arrays'/selected['path'];target.mkdir(parents=True,exist_ok=True)
            (target/'zarr.json').write_text(json.dumps(metas[selected['path']]))
            prefix=root+'/'+selected['path']+'/'
            members=[i for i in archive.infolist() if i.filename.startswith(prefix+'c/') and not i.is_dir()]
            if not members:raise ValueError('Selected level contains no shards')
            for info in members:
                rel=Path(info.filename[len(prefix):])
                if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe ZIP member')
                dest=target/rel
                if dest.exists():
                    if dest.stat().st_size==info.file_size and zlib.crc32(dest.read_bytes())==info.CRC:continue
                    raise ValueError('Local selected shard CRC mismatch')
                value=archive.read(info)
                dest.parent.mkdir(parents=True,exist_ok=True)
                temp=dest.with_suffix('.partial');temp.write_bytes(value);temp.replace(dest)
        array=zarr.open_array(str(target),mode='r')
        rgb=np.transpose(np.asarray(array[0,:,0,:,:]),(1,2,0))
        receipt = {'source_url':url,'source_zip_bytes':source['size'],'root':root,'levels':levels,
                   'selected_level':selected,'l0':levels[0],'network_bytes':raw.bytes_received,
                   'requests':raw.requests_made,'cache_hits':raw.cache_hits,
                   'root_metadata_sha256':hashlib.sha256(root_bytes).hexdigest(),
                   'pixel_backend':'native Zarr local selected array; ZIP CRC and shard index codecs',
                   'elapsed_seconds':round(time.monotonic()-raw.started,3)}
        return rgb, receipt
    finally:
        raw.close()
