"""Windows/local OpenSSH SFTP relay: verified complete WSI -> bounded server queue.

Resume uploads, back up completed features, then retire task-owned server working
copies. Local originals are never removed. No credentials stored in manifests.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def log(**record):
    print(json.dumps(dict(at=time.time(), **record), ensure_ascii=True), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--host', required=True)
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--user', default='root')
    ap.add_argument('--key', type=Path, required=True)
    ap.add_argument('--server-repo', required=True)
    ap.add_argument('--server-python', required=True)
    ap.add_argument('--server-root', required=True)
    ap.add_argument('--local-wsi', type=Path, required=True)
    ap.add_argument('--gdc-manifest', type=Path, required=True)
    ap.add_argument('--samples', type=Path, required=True)
    ap.add_argument('--state', type=Path, required=True)
    ap.add_argument('--backup', type=Path, required=True)
    ap.add_argument('--queue-gib', type=float, default=20)
    ap.add_argument('--minimum-free-gib', type=float, default=30)
    ap.add_argument('--local-minimum-free-gib', type=float, default=15)
    ap.add_argument('--watch', action='store_true')
    ap.add_argument('--poll-seconds', type=int, default=60)
    args = ap.parse_args()
    args.backup.mkdir(parents=True, exist_ok=True)
    args.state.parent.mkdir(parents=True, exist_ok=True)
    import os
    # Exclusive OS-backed lock, released on process exit; never infer liveness from a file.
    lock = args.state.with_suffix('.lock').open('a+b')
    if os.name == 'nt':
        import msvcrt
        lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = json.loads(args.state.read_text()) if args.state.exists() else dict(verified={}, staged=[], backed_up={})
    state.setdefault('retired', [])
    samples = json.loads(args.samples.read_text(encoding='utf-8-sig'))
    with args.gdc_manifest.open(encoding='utf-8-sig', newline='') as f:
        original = list(csv.DictReader(f, delimiter='\t'))
    lookup = {s['id']: s for s in samples}
    if len(lookup) != len(samples):
        raise ValueError('Duplicate source UUID')
    for r in original:
        if r['id'] in lookup and (lookup[r['id']]['md5'] != r['md5'] or lookup[r['id']]['size'] != int(r['size'])):
            raise ValueError('Pinned source manifest disagreement')
    base = ['-i', str(args.key), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
            '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=6']
    target = args.user+'@'+args.host

    def save():
        tmp = args.state.with_suffix('.partial')
        tmp.write_text(json.dumps(state, indent=2), encoding='utf-8')
        tmp.replace(args.state)

    def ssh(command):
        p = subprocess.run(['ssh', '-n', *base, '-p', str(args.port), target, command],
                           check=True, capture_output=True, text=True, encoding='utf-8',
                           stdin=subprocess.DEVNULL, timeout=120)
        return p.stdout.strip()

    def action(name, sid=None, backup_hash=None):
        words = [args.server_python, '-m', 'kt2028.prepare_tcga', name,
                 '--root', args.server_root, '--manifest', args.server_root+'/metadata/samples.json']
        if sid:
            words += ['--file-id', sid]
        if backup_hash:
            words += ['--backup-sha256', backup_hash]
        return json.loads(ssh('cd '+shlex.quote(args.server_repo)+' && '+shlex.join(words)))

    def sftp(command):
        # UTF-8 batch files preserve Chinese local paths; OpenSSH uses the existing host key pin.
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.sftp', delete=False) as f:
            batch = Path(f.name); f.write(command+'\n')
        try:
            subprocess.run(['sftp', *base, '-P', str(args.port), '-b', str(batch), target],
                           check=True, stdin=subprocess.DEVNULL, timeout=7200)
        finally:
            batch.unlink()

    def local_source(s):
        return args.local_wsi/s['id']/s['filename']

    def verified(s):
        p = local_source(s)
        if not p.is_file() or p.is_symlink() or p.stat().st_size != s['size']:
            return False
        signature = [p.stat().st_size, p.stat().st_mtime_ns]
        old = state['verified'].get(s['id'])
        if old and old['signature'] == signature and old['md5'] == s['md5']:
            return True
        log(event='checking_local_md5', file_id=s['id'], bytes=s['size'])
        h = digest(p, 'md5')
        if p.stat().st_size != signature[0] or p.stat().st_mtime_ns != signature[1]:
            return False
        if h != s['md5']:
            log(event='local_md5_mismatch_preserved', file_id=s['id'])
            return False
        state['verified'][s['id']] = dict(signature=signature, md5=h)
        save()
        return True

    log(event='relay_started', expected=len(samples), queue_gib=args.queue_gib)
    while True:
        if args.state.with_suffix('.stop').exists():
            log(event='stop_requested'); return
        try:
            status = action('status')
            ready = {s['id']: s for s in status['completed']}
            # Back up and free working copies before admitting further uploads.
            for sid, info in ready.items():
                if sid not in lookup:
                    continue
                sample = lookup[sid]
                final = args.backup/(sid+'.tar.gz')
                saved = state['backed_up'].get(sid)
                if saved is not None:
                    if not final.is_file():
                        # A missing backup must be rebuilt while server features still exist.
                        del state['backed_up'][sid]
                        if sid in state['retired']:
                            state['retired'].remove(sid)
                        save()
                    else:
                        signature = [final.stat().st_size, final.stat().st_mtime_ns]
                        if saved.get('signature') != signature:
                            if final.stat().st_size != saved['bytes'] or digest(final) != saved['sha256']:
                                raise ValueError('Previously verified local archive changed; preserve all copies')
                            saved['signature'] = signature
                            save()
                        if sid in state['retired']:
                            # Avoid one SSH round trip per past slide on every new upload.
                            # Final dataset audit remains independent of these cached acknowledgements.
                            continue
                if sid not in state['backed_up']:
                    if not verified(sample):
                        log(event='local_original_unverified_no_retire', file_id=sid); continue
                    pending_wsi = sum(int(r['size']) for r in original
                        if not ((args.local_wsi/r['id']/r['filename']).is_file()
                                and (args.local_wsi/r['id']/r['filename']).stat().st_size == int(r['size'])))
                    free = shutil.disk_usage(args.backup).free
                    if free - pending_wsi - info['output_bytes'] < args.local_minimum_free_gib*1024**3:
                        log(event='waiting_local_backup_space', free_bytes=free, reserved_download_bytes=pending_wsi)
                        continue
                    archive = action('pack', sid)
                    temp = final.with_suffix('.gz.partial')
                    if not final.exists():
                        sftp('get -a '+json.dumps(args.server_root+'/archives/'+sid+'.tar.gz')+' '+
                             json.dumps(temp.as_posix(), ensure_ascii=False))
                        if temp.stat().st_size != archive['bytes'] or digest(temp) != archive['sha256']:
                            raise ValueError('Downloaded feature archive checksum mismatch')
                        with tarfile.open(temp, 'r:gz') as tar:
                            members = tar.getmembers()
                            if any(Path(m.name).is_absolute() or '..' in Path(m.name).parts for m in members):
                                raise ValueError('Unsafe archive member')
                            if not any(m.name == sid+'/receipt.json' for m in members):
                                raise ValueError('Missing archived completion receipt')
                        temp.rename(final)
                    if digest(final) != archive['sha256']:
                        raise ValueError('Existing local backup fingerprint mismatch')
                    state['backed_up'][sid] = dict(sha256=archive['sha256'], bytes=archive['bytes'],
                        signature=[final.stat().st_size, final.stat().st_mtime_ns], at=time.time())
                    save(); log(event='feature_backup_verified', file_id=sid, bytes=archive['bytes'])
                retired = args.server_root+'/staged/'+sid+'.retired.json'
                # Retire is idempotent only while its temporary archive is present.
                exists = ssh('test -f '+shlex.quote(retired)+' && echo yes || echo no')
                if exists == 'no' and verified(sample):
                    action('retire', sid, state['backed_up'][sid]['sha256'])
                    log(event='server_working_copy_retired', file_id=sid)
                    exists = 'yes'
                if exists == 'yes':
                    state['retired'].append(sid)
                    save()
            if len(state['backed_up']) == len(samples):
                ssh('touch '+shlex.quote(args.server_root+'/STOP'))
                log(event='relay_completed', completed=len(samples)); return
            status = action('status')
            admitted = False
            # Source order is fixed; eligibility is based on completeness, never outcome.
            for sample in samples:
                sid = sample['id']
                if sid in state['staged'] or sid in ready:
                    continue
                if status['staged_bytes']+sample['size'] > args.queue_gib*1024**3:
                    continue
                if status['free_bytes']-sample['size'] < args.minimum_free_gib*1024**3:
                    continue
                if not verified(sample):
                    continue
                source = local_source(sample)
                remote = args.server_root+'/incoming/'+sid+'.svs.partial'
                # OpenSSH reput/put -a requires an existing remote target on Windows.
                # Only resume partials; new transfers use ordinary put.
                resume = ssh('test -f '+shlex.quote(remote)+' && echo yes || echo no') == 'yes'
                started = time.monotonic()
                log(event='upload_started', file_id=sid, bytes=sample['size'])
                sftp(('put -a ' if resume else 'put ')+
                     json.dumps(source.as_posix(), ensure_ascii=False)+' '+json.dumps(remote))
                action('stage', sid)
                seconds = time.monotonic()-started
                state['staged'].append(sid); save()
                log(event='upload_md5_verified', file_id=sid, bytes=sample['size'],
                    elapsed_seconds=round(seconds, 2), effective_mb_s=round(sample['size']/seconds/1e6, 3))
                admitted = True
                # Re-check space and back up completions after every file.
                break
            if not admitted:
                log(event='waiting', completed_features=len(ready), backed_up=len(state['backed_up']),
                    staged_files=len(state['staged']), free_server_bytes=status['free_bytes'])
                if not args.watch:
                    return
                time.sleep(args.poll_seconds)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
            log(event='transient_transfer_error', error=str(error)[:500])
            if not args.watch:
                raise
            time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
