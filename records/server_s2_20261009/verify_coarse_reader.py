#!/usr/bin/env python3
"""Bounded official RGB access using Zarr's production ZIP/codecs; no model training."""
import io,json,time,hashlib,argparse,threading,zipfile
from pathlib import Path
import numpy as np,requests,zarr
from PIL import Image,ImageDraw
class RangeFile(io.RawIOBase):
    def __init__(self,url,size,cap): self.url=url;self.size=size;self.cap=cap;self.pos=0;self.bytes=0;self.cache={};self.lock=threading.RLock()
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,n,whence=0):
        self.pos=n if whence==0 else self.pos+n if whence==1 else self.size+n
        if self.pos<0:raise ValueError('negative seek')
        return self.pos
    def read(self,n=-1):
        n=min(n if n>=0 else self.size-self.pos,self.size-self.pos)
        if n<=0:return b''
        key=(self.pos,n)
        if key in self.cache: self.pos+=n;return self.cache[key]
        if n>32*1024**2 or self.bytes+n>self.cap:raise ValueError('network cap exceeded')
        for attempt in range(3):
            try:
                with requests.get(self.url,headers={'Range':f'bytes={self.pos}-{self.pos+n-1}','Accept-Encoding':'identity'},timeout=(15,40),stream=True) as r:
                    if r.status_code!=206:raise ValueError(f'expected 206, got {r.status_code}')
                    expected=f'bytes {self.pos}-{self.pos+n-1}/{self.size}'
                    if r.headers.get('Content-Range')!=expected:raise ValueError('Content-Range mismatch')
                    data=r.raw.read(n+1);self.bytes+=len(data)
                    if len(data)!=n:raise ValueError('short range response')
                self.cache[key]=data;self.pos+=n;return data
            except requests.RequestException:
                if attempt==2:raise
                time.sleep(attempt+1)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--samples',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    zarr.config.set({'async.concurrency':1,'threading.max_workers':1})
    receipts=[]
    for sample in json.loads(args.samples.read_text()):
        t=time.time();row={'source_zip_bytes':sample['source']['size'],'scope':'three Train examples only; metadata/low RGB/coordinate geometry; no prediction','status':'started'}
        raw=None
        try:
            url='https://www.ebi.ac.uk/biostudies/files/S-BIAD1285/'+sample['source']['path']
            raw=RangeFile(url,sample['source']['size'],64*1024**2)
            folder=args.output/sample['slide_id'];folder.mkdir()
            with zipfile.ZipFile(raw) as zf:
                root=next(n[:-len('/zarr.json')] for n in zf.namelist() if n.count('/')==1 and n.endswith('/zarr.json'))
                attrs=json.loads(zf.read(root+'/zarr.json'))['attributes']
            store=zarr.storage.ZipStore(raw,mode='r')
            group=zarr.open_group(store=store,path=root,mode='r',use_consolidated=False)
            mult=attrs.get('ome',attrs)['multiscales'][0]
            datasets=mult['datasets']; levels=[]
            for d in datasets:
                a=group[d['path']];scale=next(c['scale'] for c in d['coordinateTransformations'] if c['type']=='scale')
                levels.append({'path':d['path'],'shape':list(a.shape),'scale':scale})
            l0=levels[0]; low=levels[-1]
            a=group[low['path']]
            rgb=np.transpose(np.asarray(a[0,:,0,:,:]),(1,2,0))
            if rgb.dtype!=np.uint8 or rgb.shape[-1]!=3:raise ValueError('RGB layout mismatch')
            Image.fromarray(rgb).save(folder/'thumbnail.png')
            fine=zarr.open_group(sample['fine_path'],mode='r');coords=np.asarray(fine['coords'][:])
            x_mpp=l0['scale'][-1];y_mpp=l0['scale'][-2]
            if not all(0.05<v<0.5 for v in (x_mpp,y_mpp)):raise ValueError('unexpected L0 physical scale')
            centers=coords+np.array([224/x_mpp,224/y_mpp])/2
            inside=(centers[:,0]>=0)&(centers[:,0]<l0['shape'][-1])&(centers[:,1]>=0)&(centers[:,1]<l0['shape'][-2])
            overlay=Image.fromarray(rgb).resize((rgb.shape[1]*3,rgb.shape[0]*3))
            draw=ImageDraw.Draw(overlay)
            stride=max(1,len(coords)//300)
            for c in centers[::stride]:
                x=c[0]*x_mpp/low['scale'][-1]*3;y=c[1]*y_mpp/low['scale'][-2]*3
                draw.ellipse((x-1,y-1,x+1,y+1),fill=(0,255,0))
            overlay.save(folder/'coordinate_overlay.png')
            row.update(status='verified',url=url,levels=levels,thumbnail_shape=list(rgb.shape),fine_patches=len(coords),coordinate_min=coords.min(0).tolist(),coordinate_max=coords.max(0).tolist(),fine_centers_inside_image=int(inside.sum()),fine_centers_outside_image=int((~inside).sum()),l0_mpp_xy=[x_mpp,y_mpp],fine_field_um=224,geometry_status='bounds and visual overlay only; does not prove per-patch anatomical registration',thumbnail_sha256=hashlib.sha256((folder/'thumbnail.png').read_bytes()).hexdigest())
            store.close()
        except Exception as e:row.update(status='failed',error=type(e).__name__+': '+str(e)[:400])
        row.update(bytes_received=raw.bytes if raw else 0,elapsed_seconds=round(time.time()-t,3))
        receipts.append(row);(args.output/'receipt.json').write_text(json.dumps(receipts,indent=2))
        print('COARSE_SAMPLE '+json.dumps(row),flush=True)
    if any(r['status']!='verified' for r in receipts):raise SystemExit(1)
if __name__=='__main__':main()
