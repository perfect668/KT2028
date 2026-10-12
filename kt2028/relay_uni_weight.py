"""One-shot local download staging and server extraction launcher.

No browser credentials or tokens are read. The user's completed official file is
accepted only with the pinned size/SHA256. It then stages the model via existing
SSH authorization, checks a real frozen GPU forward, and launches Train-only
feature preparation. Status/logs live outside the public repository.
"""
import argparse
import datetime
import json
import shlex
import shutil
import subprocess
import time
from pathlib import Path, PurePosixPath

from .uni_encoder import verify_checkpoint


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--download-dir', type=Path, required=True)
    ap.add_argument('--local-model-dir', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--ssh-key', type=Path, required=True)
    ap.add_argument('--host', required=True)
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--remote-root', required=True)
    ap.add_argument('--status', type=Path, required=True)
    ap.add_argument('--wait-hours', type=float, default=12)
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    spec = config['uni']
    base = PurePosixPath(args.remote_root)
    if not base.is_absolute() or '..' in base.parts or not args.ssh_key.is_file():
        raise ValueError('Invalid explicit server root or SSH key')
    if not 1 <= args.port <= 65535 or args.wait_hours <= 0:
        raise ValueError('Invalid relay limits')
    args.status.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    phase = 'waiting_for_completed_browser_download'

    def status(state, **fields):
        value = dict(status=state, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     elapsed_seconds=round(time.monotonic()-started, 1), **fields)
        temp = args.status.with_suffix('.partial')
        temp.write_text(json.dumps(value, indent=2))
        temp.replace(args.status)
        print(json.dumps(value), flush=True)

    ssh = ['ssh', '-i', str(args.ssh_key), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
           '-p', str(args.port), args.host]
    repo = str(base / 'repo')
    python = str(base / '.venv/bin/python')

    def remote_python(code, timeout=600):
        cmd = 'cd ' + shlex.quote(repo) + ' && ' + shlex.quote(python) + ' -'
        result = subprocess.run(ssh + [cmd], input=code, text=True, capture_output=True, timeout=timeout)
        if result.returncode:
            # Our code/commands contain no tokens; retain the actionable last error only.
            raise RuntimeError((result.stderr or result.stdout).splitlines()[-1][:350])
        return json.loads(result.stdout.strip().splitlines()[-1])

    try:
        status(phase, expected_bytes=spec['size_bytes'])
        completed = None
        while time.monotonic()-started < args.wait_hours*3600:
            for candidate in args.download_dir.glob('pytorch_model*.bin'):
                if candidate.stat().st_size == spec['size_bytes']:
                    verify_checkpoint(candidate, spec)
                    completed = candidate
                    break
            if completed:
                break
            time.sleep(30)
        if completed is None:
            raise TimeoutError('No complete verified original UNI file before relay deadline')
        phase = 'staging_verified_weight'
        status(phase, weight_sha256=spec['sha256'])
        args.local_model_dir.mkdir(parents=True, exist_ok=True)
        local = args.local_model_dir / spec['filename']
        if local.exists():
            verify_checkpoint(local, spec)
        else:
            temporary = local.with_suffix('.uploading')
            shutil.copyfile(completed, temporary)
            verify_checkpoint(temporary, spec)
            temporary.replace(local)
        model_dir = str(base / 'cache/uni_original')
        subprocess.run(ssh + ['mkdir -p ' + shlex.quote(model_dir)], check=True, timeout=60)
        remote_weight = str(PurePosixPath(model_dir) / spec['filename'])
        remote_tmp = remote_weight + '.uploading'
        already = remote_python('from pathlib import Path\nimport json\nfrom kt2028.uni_encoder import verify_checkpoint\n'
                                f'p=Path({remote_weight!r});spec={spec!r}\n'
                                'if p.exists():verify_checkpoint(p,spec)\n'
                                'print(json.dumps({"verified":p.exists()}))\n')
        if not already['verified']:
            for attempt in range(3):
                try:
                    subprocess.run(['scp', '-q', '-i', str(args.ssh_key), '-o', 'BatchMode=yes',
                                    '-o', 'ConnectTimeout=15', '-P', str(args.port), str(local),
                                    args.host + ':' + remote_tmp], check=True, timeout=1800)
                    break
                except (subprocess.SubprocessError, OSError):
                    if attempt == 2:
                        raise
                    time.sleep(10)
            remote_python('from pathlib import Path\nimport json\nfrom kt2028.uni_encoder import verify_checkpoint\n'
                          f'tmp=Path({remote_tmp!r});p=Path({remote_weight!r});spec={spec!r}\n'
                          'verify_checkpoint(tmp,spec)\n'
                          'if p.exists():verify_checkpoint(p,spec)\n'
                          'else:tmp.replace(p)\n'
                          'print(json.dumps({"verified":True}))\n')
        phase = 'verifying_original_uni_real_gpu_forward'
        status(phase)
        receipt = str(base / 'audit/r2_preparation_20261012/uni_verified.json')
        image = str(base / 'audit/r2_preparation_20261012/real_pixels_T103/fine_0_0_0p5mpp.png')
        remote_python('import json,subprocess,sys\nfrom pathlib import Path\n'
                      f'receipt=Path({receipt!r})\n'
                      'if not receipt.exists():\n'
                      f' subprocess.run([sys.executable,"-m","kt2028.uni_encoder","--config","configs/dual_scale_r2.json","--checkpoint",{remote_weight!r},"--image",{image!r},"--receipt",str(receipt)],check=True)\n'
                      'r=json.loads(receipt.read_text())\n'
                      f'if r.get("weight_sha256")!={spec["sha256"]!r} or r.get("status")!="weight_and_real_forward_verified" or r.get("shape")!=[1,1024]:raise ValueError("UNI verification receipt mismatch")\n'
                      'print(json.dumps({"verified":True,"shape":r["shape"]}))\n')
        phase = 'waiting_for_train_geometry'
        status(phase)
        geometry = str(base / 'data/r2_geometry_20261012_train')
        while time.monotonic()-started < args.wait_hours*3600:
            progress = remote_python('import json\nfrom pathlib import Path\n'
                                      f'print(json.dumps(json.loads(Path({geometry!r},"summary.json").read_text())))\n')
            if progress['expected'] != 255 or progress['errors']:
                raise ValueError('Official Train geometry has an unresolved mapping/preparation error')
            if progress['completed'] == 255:
                break
            time.sleep(60)
        else:
            raise TimeoutError('Train geometry not complete before relay deadline')
        phase = 'launching_train_feature_queue'
        status(phase)
        output = str(base / 'data/r2_features_20261012_train')
        log = str(base / 'audit/r2_preparation_20261012/extraction.log')
        pid_path = str(base / 'audit/r2_preparation_20261012/extraction_pid.json')
        command = [python, '-m', 'kt2028.extract_dual_scale', '--samples',
                   str(base / 'audit/r2_preparation_20261012/train_extraction_order.json'),
                   '--config', 'configs/dual_scale_r2.json', '--geometry', geometry,
                   '--checkpoint', remote_weight, '--cache', str(base / 'cache/r2_extraction_20261012'),
                   '--fallback-cache', str(base / 'cache/r2_roi_ranges'), '--output', output]
        result = remote_python('import json,subprocess,time\nfrom pathlib import Path\n'
                               f'pid_path=Path({pid_path!r})\n'
                               'if pid_path.exists():raise FileExistsError("Existing extraction process record; inspect before relaunch")\n'
                               f'log=Path({log!r})\n'
                               'with log.open("x") as f:\n'
                               f' p=subprocess.Popen({command!r},cwd={repo!r},stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)\n'
                               'time.sleep(2)\n'
                               'if p.poll() is not None:raise RuntimeError("Extraction exited at startup; inspect log")\n'
                               f'r={{"pid":p.pid,"log":str(log),"output":{output!r},"command":{command!r}}}\n'
                               'pid_path.write_text(json.dumps(r,indent=2))\n'
                               'print(json.dumps(r))\n')
        status('server_extraction_process_started', **{k: result[k] for k in ['pid', 'log', 'output']},
               weight_sha256=spec['sha256'], scope='Train only; frozen UNI; no prognosis/PPO training')
    except Exception as e:
        status('failed', failed_phase=phase, error=type(e).__name__+': '+str(e)[:400])
        raise


if __name__ == '__main__':
    main()
