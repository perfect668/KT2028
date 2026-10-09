#!/usr/bin/env python3
"""Bounded CPU-only thumbnail probe for the verified official OME-Zarr ZIP."""
import argparse, ctypes, ctypes.util, hashlib, io, json, math, zipfile
from pathlib import Path
import requests
import numpy as np
from PIL import Image

def crc32c(data):
    table=[]
    for n in range(256):
        c=n
        for _ in range(8): c=(c>>1)^0x82F63B78 if c&1 else c>>1
        table.append(c)
    crc=0xffffffff
    for b in data: crc=table[(crc^b)&255]^(crc>>8)
    return crc^0xffffffff

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source-report',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    src=json.loads(args.source_report.read_text())
    out=args.output
    if out.exists(): raise FileExistsError('Use a fresh thumbnail output directory')
    out.mkdir(parents=True)
    report={'url':src['url'],'source_report':str(args.source_report),'bytes_received':0,
            'scope':'one real image thumbnail on CPU; not coarse model extraction or training'}
    class RemoteFile(io.RawIOBase):
        def __init__(self): self.pos=0
        def seekable(self): return True
        def readable(self): return True
        def tell(self): return self.pos
        def seek(self,n,whence=0):
            self.pos=n if whence==0 else self.pos+n if whence==1 else src['example_size']+n
            if self.pos<0: raise ValueError('Negative seek')
            return self.pos
        def read(self,n=-1):
            n=src['example_size']-self.pos if n<0 else min(n,src['example_size']-self.pos)
            if not n:return b''
            if n>2*1024*1024 or report['bytes_received']+n>4*1024*1024:
                raise ValueError('Bounded probe read cap exceeded')
            with requests.get(src['url'],headers={'Range':f'bytes={self.pos}-{self.pos+n-1}'},
                              timeout=(10,20),stream=True) as r:
                if r.status_code!=206: raise ValueError('HTTP range not honored')
                data=r.raw.read(n+1)
                if len(data)!=n: raise ValueError('Partial response size mismatch')
                self.pos+=n; report['bytes_received']+=n; return data
    try:
        with zipfile.ZipFile(RemoteFile()) as z:
            metas={name:json.loads(z.read(name)) for name in z.namelist()
                   if name.endswith('/zarr.json') and '/'+name.split('/')[-2]+'/' in name
                   and name.split('/')[-2].isdigit()}
            name,meta=min(metas.items(),key=lambda kv:math.prod(kv[1]['shape'][-2:]))
            shape=meta['shape']
            if shape[:3]!=[1,3,1] or meta['data_type']!='uint8':
                raise ValueError('Expected single RGB plane')
            codec=meta['codecs'][0]
            if codec['name']!='sharding_indexed': raise ValueError('Expected indexed shard')
            config=codec['configuration']; inner=config['chunk_shape']
            if inner[:3]!=[1,1,1] or inner[-2:]!=shape[-2:] or config['index_location']!='end':
                raise ValueError('Probe requires one full-plane inner chunk')
            if [c['name'] for c in config['codecs']]!=['bytes','zstd']:
                raise ValueError('Unsupported inner codec')
            if [c['name'] for c in config['index_codecs']]!=['bytes','crc32c']:
                raise ValueError('Unsupported index codec')
            shard_shape=meta['chunk_grid']['configuration']['chunk_shape']
            chunks=tuple(a//b for a,b in zip(shard_shape,inner))
            entry=name.removesuffix('zarr.json')+'c/'+'/'.join(['0']*len(shape))
            blob=z.read(entry)
            index_size=math.prod(chunks)*16+4
            index_bytes=blob[-index_size:]
            if crc32c(index_bytes[:-4])!=int.from_bytes(index_bytes[-4:],'little'):
                raise ValueError('Shard index CRC32C mismatch')
            index=np.frombuffer(index_bytes[:-4],dtype='<u8').reshape(chunks+(2,))
            lib=ctypes.CDLL(ctypes.util.find_library('zstd'))
            lib.ZSTD_decompress.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t]
            lib.ZSTD_decompress.restype=ctypes.c_size_t
            lib.ZSTD_isError.argtypes=[ctypes.c_size_t];lib.ZSTD_isError.restype=ctypes.c_uint
            channels=[]
            for channel in range(3):
                offset,nbytes=map(int,index[0,channel,0,0,0])
                if offset+nbytes>len(blob)-index_size: raise ValueError('Shard offset out of range')
                encoded=blob[offset:offset+nbytes]; source=ctypes.create_string_buffer(encoded)
                size=math.prod(inner); buffer=ctypes.create_string_buffer(size)
                count=lib.ZSTD_decompress(buffer,size,source,len(encoded))
                if lib.ZSTD_isError(count) or count!=size: raise ValueError('Zstd decode failed')
                channels.append(np.frombuffer(buffer.raw,dtype=np.uint8).reshape(tuple(inner))[0,0,0])
            rgb=np.stack(channels,axis=-1)
            png=out/'thumbnail.png';Image.fromarray(rgb).save(png)
            report.update({'status':'verified','pixel_arrays_decoded':True,'shard_index_crc32c_verified':True,
                'zip_crc_verified':True,'level_metadata':name,'shape':list(rgb.shape),
                'source_case_name':name.split('/')[0].removesuffix('.ome.zarr'),
                'thumbnail':str(png),'png_sha256':hashlib.sha256(png.read_bytes()).hexdigest(),
                'rgb_min':int(rgb.min()),'rgb_max':int(rgb.max()),
                'source_zip_bytes':src['example_size'],
                'all_sr386_coarse_views_available':'not yet verified'})
    except Exception as e:
        report.update({'status':'failed','error':type(e).__name__+': '+str(e)[:200]})
    with (out/'receipt.json').open('x') as f:json.dump(report,f,indent=2)
    print('THUMBNAIL_PROBE '+json.dumps(report),flush=True)
    if report['status']!='verified':raise SystemExit(1)

if __name__=='__main__':main()

