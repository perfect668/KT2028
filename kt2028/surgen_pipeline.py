"""Bounded CZI staging/feature-watch/archive interface; no prognosis fitting."""
import argparse
import json
from pathlib import Path
import re
import shutil
import tarfile
import time

from .extract_dual_scale import atomic_json
from .prepare_surgen import extract
from .prepare_tcga import verify_output
from .uni_encoder import sha256_file


def source(root,sid):
    p=root/'incoming'/(sid+'.czi')
    if p.is_symlink() or p.with_suffix('.czi.partial').is_symlink():
        raise ValueError('Symlinked working copy; preserve and investigate')
    return p if p.exists() else p.with_suffix('.czi.partial')


def status(root,samples):
    ready=[]
    for s in samples:
        folder=root/'features'/s['id']; p=folder/'receipt.json'
        if p.exists():
            r=json.loads(p.read_text())
            if r['status']=='completed':
                ready.append(dict(id=s['id'],output_bytes=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())))
    return dict(completed=ready,staged_bytes=sum(p.stat().st_size for p in (root/'incoming').glob('*') if p.is_file()),
                free_bytes=shutil.disk_usage(root).free)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['status','stage','pack','retire','watch'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--file-id')
    p.add_argument('--source-sha256')
    p.add_argument('--backup-sha256')
    p.add_argument('--config',type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--device',default='cuda')
    p.add_argument('--poll-seconds',type=int,default=30)
    args=p.parse_args()
    root=args.root.resolve(strict=True)
    if (root/'_TASK_OWNER').read_text().strip()!='W067': raise ValueError('Unowned CZI working directory')
    samples=json.loads(args.manifest.read_text(encoding='utf-8-sig'))
    if len({s['id'] for s in samples})!=len(samples): raise ValueError('Duplicate source identity')
    for s in samples:
        if not re.fullmatch(r'SR386_40X_HE_T\d+_\d+',s['id']) or s['filename']!=s['id']+'.czi':
            raise ValueError('Invalid source identity/path')
    if args.file_id:
        samples=[s for s in samples if s['id']==args.file_id]
        if len(samples)!=1: raise ValueError('Unknown source identity')
    if args.action=='status':
        print(json.dumps(status(root,samples))); return
    (root/'staged').mkdir(exist_ok=True)
    if args.action in ('stage','pack','retire'):
        if len(samples)!=1: raise ValueError('Exactly one source required')
        s=samples[0]; sid=s['id']; src=source(root,sid)
        if args.action=='stage':
            if not args.source_sha256 or not re.fullmatch('[0-9a-f]{64}',args.source_sha256):
                raise ValueError('A locally verified transfer SHA256 is mandatory')
            if src.is_symlink() or src.stat().st_size!=s['size'] or sha256_file(src)!=args.source_sha256:
                raise ValueError('CZI working-copy size/SHA mismatch')
            if src.suffix=='.partial': src.rename(root/'incoming'/(sid+'.czi'))
            record=dict(id=sid,size=s['size'],source_sha256=args.source_sha256,at_unix=time.time(),
                        official_checksum_supplied=False)
            atomic_json(root/'staged'/(sid+'.json'),record)
            print(json.dumps(record)); return
        folder=root/'features'/sid
        if folder.is_symlink(): raise ValueError('Unsafe feature directory')
        receipt=json.loads((folder/'receipt.json').read_text())
        verify_output(folder,receipt)
        archives=root/'archives'; archives.mkdir(exist_ok=True)
        archive=archives/(sid+'.tar.gz')
        if archives.is_symlink() or archive.is_symlink(): raise ValueError('Unsafe archive path')
        if args.action=='pack':
            if not archive.exists():
                if shutil.disk_usage(root).free<2*receipt['fine']['rows']*4096+2*1024**3:
                    raise OSError('Insufficient archive headroom')
                tmp=archive.with_suffix('.gz.partial')
                with tarfile.open(tmp,'w:gz',compresslevel=1) as tar: tar.add(folder,arcname=sid)
                tmp.rename(archive)
            print(json.dumps(dict(id=sid,bytes=archive.stat().st_size,sha256=sha256_file(archive),
                                  source_sha256=receipt['source_sha256']))); return
        if not args.backup_sha256 or sha256_file(archive)!=args.backup_sha256:
            raise ValueError('Matching verified off-server archive required')
        if src.exists():
            if src.is_symlink() or src.resolve().parent!=(root/'incoming') or src.stat().st_size!=s['size']:
                raise ValueError('Unsafe or changed CZI working copy')
            if sha256_file(src)!=receipt['source_sha256']: raise ValueError('Changed source; preserve')
            src.unlink()  # Only a verified task-owned server working copy; local original stays.
        if archive.is_symlink() or archive.resolve().parent!=archives: raise ValueError('Unsafe archive')
        atomic_json(root/'staged'/(sid+'.retired.json'),dict(source_sha256=receipt['source_sha256'],
            backup_sha256=args.backup_sha256,local_original_preserved_by_transfer_client=True,at_unix=time.time()))
        archive.unlink()
        print(json.dumps(dict(id=sid,retired=True))); return
    if not args.config or not args.checkpoint: raise ValueError('Extraction config/checkpoint required')
    import fcntl
    lock=(root/'extraction.lock').open('a+b'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=json.loads(args.config.read_text())
    source_dir=Path(__file__).parent
    pinned=dict(config_sha256=sha256_file(args.config),manifest_sha256=sha256_file(args.manifest),
        source_shas={n:sha256_file(source_dir/n) for n in ('surgen_pipeline.py','prepare_surgen.py',
            'local_czi.py','dual_scale_geometry.py','regions.py','extract_dual_scale.py','uni_encoder.py')})
    registration=root/'pipeline_run.json'
    if registration.exists() and json.loads(registration.read_text())!=pinned:
        raise ValueError('Pipeline source/config/manifest differs; preserve prior queue')
    if not registration.exists(): atomic_json(registration,pinned)
    failed=set()
    while True:
        ready={s['id'] for s in status(root,samples)['completed']}
        if len(ready)+len(failed)==len(samples) or (root/'STOP').exists(): return
        for s in samples:
            sid=s['id']; staged=root/'staged'/(sid+'.json')
            if sid in ready or sid in failed or not staged.is_file(): continue
            if shutil.disk_usage(root).free<cfg['minimum_free_disk_bytes']: break
            record=json.loads(staged.read_text())
            args.source=source(root,sid); args.source_size=s['size']; args.slide_id=sid
            args.source_sha256=record['source_sha256']
            try:
                result=extract(args)
                print(json.dumps(dict(event='completed_slide',id=sid,coarse=result['coarse']['rows'],fine=result['fine']['rows'])),flush=True)
            except Exception as error:
                failed.add(sid)
                folder=root/'features'/sid; folder.mkdir(exist_ok=True)
                record=dict(error=type(error).__name__+': '+str(error)[:500])
                atomic_json(folder/'failure.json',record)
                print(json.dumps(dict(event='failed_slide',id=sid,**record)),flush=True)
        time.sleep(args.poll_seconds)


if __name__=='__main__':main()
