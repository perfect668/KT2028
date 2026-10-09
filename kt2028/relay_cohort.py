"""Bounded local download + authenticated SCP relay; no stored passwords/tokens."""
import argparse
import concurrent.futures as cf
import json
import shlex
import shutil
import subprocess
from pathlib import Path

from .prefetch_rgb_cache import prefetch


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--samples',type=Path,required=True)
    ap.add_argument('--config',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--receipt',type=Path,required=True);ap.add_argument('--ssh-key',type=Path,required=True)
    ap.add_argument('--host',required=True);ap.add_argument('--port',type=int,required=True)
    ap.add_argument('--remote-cache',required=True);ap.add_argument('--image-workers',type=int,default=4)
    ap.add_argument('--chunk-workers',type=int,default=4);args=ap.parse_args()
    samples=json.loads(args.samples.read_text());config=json.loads(args.config.read_text())
    args.cache.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(args.cache).free<8*1024**3:raise OSError('Local disk headroom below 8 GiB')
    def transfer(sample):
        if shutil.disk_usage(args.cache).free<5*1024**3:raise OSError('Local disk headroom below 5 GiB')
        row=prefetch(sample,config,args.cache,args.chunk_workers)
        folder=Path(row['cache_directory']);destination=f'{args.host}:{args.remote_cache.rstrip("/")}/'
        subprocess.run(['scp','-q','-i',str(args.ssh_key),'-o','BatchMode=yes','-P',str(args.port),
                        '-r',str(folder),destination],check=True,timeout=1200)
        marker=args.remote_cache.rstrip('/')+'/'+folder.name+'/.relay_ready'
        subprocess.run(['ssh','-i',str(args.ssh_key),'-o','BatchMode=yes','-p',str(args.port),args.host,
                        'touch -- '+shlex.quote(marker)],check=True,timeout=60)
        row['uploaded_to_server']=True
        print('RELAY_UPLOADED '+sample['slide_id'],flush=True)
        return row
    rows=[];errors=[]
    with cf.ThreadPoolExecutor(max_workers=args.image_workers) as pool:
        jobs={pool.submit(transfer,s):s for s in samples}
        for future in cf.as_completed(jobs):
            try:rows.append(future.result())
            except Exception as e:errors.append({'slide_id':jobs[future]['slide_id'],'error':type(e).__name__+': '+str(e)})
            report={'scope':'selected low RGB cache relay; no complete WSI archives',
                    'expected':len(samples),'uploaded':len(rows),'pending':len(samples)-len(rows)-len(errors),
                    'errors':errors,'images':rows,'network_bytes':sum(r['network_bytes'] for r in rows)}
            temp=args.receipt.with_suffix('.partial');temp.write_text(json.dumps(report,indent=2));temp.replace(args.receipt)
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
