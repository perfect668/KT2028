"""Dependency-free range relay for slow server international links.

Download only selected native low-resolution ZIP members, preserving CRC checks.
The cache format matches remote_rgb.CachedRangeFile; full archives are forbidden.
"""
import argparse
import concurrent.futures as cf
import hashlib
import io
import json
import math
import threading
import time
import urllib.request
import zipfile
from pathlib import Path


class RelayFile(io.RawIOBase):
    def __init__(self,source,cache,cap,workers):
        canonical='https://www.ebi.ac.uk/biostudies/files/S-BIAD1285/'+source['path']
        self.url=source.get('url',canonical);self.size=source['size'];self.cap=cap;self.pos=0
        if not self.url.startswith(('https://ftp.ebi.ac.uk/','https://www.ebi.ac.uk/')):
            raise ValueError('Only official source URLs accepted')
        self.folder=Path(cache)/hashlib.sha256(f'{canonical}\n{self.size}'.encode()).hexdigest()
        self.folder.mkdir(parents=True,exist_ok=True)
        self.bytes=0;self.workers=workers;self.lock=threading.Lock()
        self.started=time.monotonic()

    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,n,whence=0):
        self.pos=n if whence==0 else self.pos+n if whence==1 else self.size+n
        if self.pos<0:raise ValueError('Negative seek')
        return self.pos

    def chunk(self,start,count):
        for attempt in range(3):
            try:
                request=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{start+count-1}','Accept-Encoding':'identity'})
                with urllib.request.urlopen(request,timeout=40) as response:
                    if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{start+count-1}/{self.size}':
                        raise ValueError('Range not honored')
                    data=response.read(count+1)
                with self.lock:
                    self.bytes+=len(data)
                    if self.bytes>self.cap:raise ValueError('Network byte cap exceeded')
                if len(data)!=count:raise ValueError('Incomplete range')
                return data
            except (OSError,urllib.error.URLError):
                if attempt==2:raise
                time.sleep(attempt+1)

    def fetch(self,start,count):
        path=self.folder/f'{start}_{count}.bin'
        if path.exists():
            data=path.read_bytes()
            if len(data)!=count:raise ValueError('Cached size mismatch')
            return data
        if count>self.cap or self.bytes+count>self.cap:raise ValueError('Bounded relay cap exceeded')
        step=1024**2
        pieces=[(i,min(step,start+count-i)) for i in range(start,start+count,step)]
        with cf.ThreadPoolExecutor(max_workers=self.workers) as pool:
            data=b''.join(pool.map(lambda p:self.chunk(*p),pieces))
        temp=path.with_suffix('.partial');temp.write_bytes(data);temp.replace(path)
        return data

    def read(self,n=-1):
        n=min(n if n>=0 else self.size-self.pos,self.size-self.pos)
        if n<=0:return b''
        if n<128*1024:
            start=self.pos//131072*131072
            count=min(math.ceil((self.pos+n-start)/131072)*131072,self.size-start)
            data=self.fetch(start,count)[self.pos-start:self.pos-start+n]
        else:data=self.fetch(self.pos,n)
        self.pos+=n;return data


def prefetch(sample,config,cache,workers):
    raw=RelayFile(sample['source'],cache,config['max_network_bytes_per_image'],workers)
    with zipfile.ZipFile(raw) as archive:
        root=next(n[:-len('/zarr.json')] for n in archive.namelist() if n.count('/')==1 and n.endswith('/zarr.json'))
        meta=json.loads(archive.read(root+'/zarr.json'));attrs=meta['attributes']
        mult=attrs.get('ome',attrs)['multiscales'][0];options=[]
        for d in mult['datasets']:
            scale=next(t['scale'] for t in d['coordinateTransformations'] if t['type']=='scale')
            if max(scale[-2:])<=config['target_coarse_mpp_um']:options.append((max(scale[-2:]),d['path']))
        if not options:raise ValueError('No appropriate native level')
        _,level=max(options)
        array_meta=json.loads(archive.read(root+'/'+level+'/zarr.json'))
        if math.prod(array_meta['shape'][-2:])>config['max_low_rgb_pixels']:raise ValueError('Low RGB memory cap')
        members=[i for i in archive.infolist() if i.filename.startswith(root+'/'+level+'/c/') and not i.is_dir()]
        if not members:raise ValueError('No selected level data')
        for member in members:
            if member.file_size>config['max_network_bytes_per_image']:raise ValueError('Selected ZIP member exceeds cap')
            archive.read(member)  # Mature ZIP CRC check, no custom Zarr decoding.
    receipt={'slide_id':sample['slide_id'],'selected_level':level,'zip_members_crc_verified':len(members),
             'network_bytes':raw.bytes,'elapsed_seconds':round(time.monotonic()-raw.started,3),
             'cache_directory':str(raw.folder),'no_full_archive_download':True}
    print('RELAY_COMPLETE '+json.dumps(receipt),flush=True)
    return receipt


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--samples',type=Path,required=True)
    ap.add_argument('--config',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--receipt',type=Path,required=True);ap.add_argument('--chunk-workers',type=int,default=4)
    ap.add_argument('--image-workers',type=int,default=2);args=ap.parse_args()
    config=json.loads(args.config.read_text());samples=json.loads(args.samples.read_text())
    rows=[];errors=[]
    with cf.ThreadPoolExecutor(max_workers=args.image_workers) as pool:
        jobs={pool.submit(prefetch,s,config,args.cache,args.chunk_workers):s for s in samples}
        for future in cf.as_completed(jobs):
            try:rows.append(future.result())
            except Exception as e:
                errors.append({'slide_id':jobs[future]['slide_id'],'error':type(e).__name__+': '+str(e)})
            args.receipt.write_text(json.dumps({'completed':len(rows),'errors':errors,'images':rows,
                                               'network_bytes':sum(r['network_bytes'] for r in rows)},indent=2))
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
