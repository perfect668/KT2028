"""Anonymous W160 progress/figures; never modify training or patient files."""
import argparse
import html
import json
import subprocess
import tarfile
import time
from pathlib import Path

import numpy as np
from .run_multicancer import write
from .uni_encoder import sha256_file

KEYS = ['c_index', 'nll', 'integrated_brier_annual_grid']
MODES = ['zero', '0.2', 'full']


def build(root):
    public = root / 'public'; public.mkdir(exist_ok=True)
    cfg = json.loads((root/'config.json').read_text())
    signature = sha256_file(root/'provenance.json')
    records = []; failures = []
    last_updates = {}
    for worker in range(3):
        p = root/f'worker_{worker}'/'failure.json'
        if p.exists(): failures.append(dict(worker=worker, failure=True))
        log = root/f'worker_{worker}.log'
        if log.exists():
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size-200000))
                lines=stream.read().decode('utf-8',errors='replace').splitlines()
            for line in lines:
                try:item=json.loads(line)
                except (ValueError,TypeError):continue
                if 'fold' in item and 'optimizer_steps' in item:
                    last_updates[item['fold']]=max(last_updates.get(item['fold'],0),item['optimizer_steps'])
    for fold in range(5):
        folder = root/'folds'/str(fold)
        record = dict(fold=fold, state='queued', optimizer_steps=0, history=[])
        if (folder/'status.json').exists(): record.update(json.loads((folder/'status.json').read_text()))
        record['optimizer_steps']=max(record.get('optimizer_steps',0),last_updates.get(fold,0))
        if (folder/'history.json').exists():
            raw = json.loads((folder/'history.json').read_text())
            # Explicit aggregate whitelist: no patient identity or individual curves.
            record['history'] = [dict(epoch=h['epoch'], train_objective=h['train_objective'],
                validation={k:h['validation']['full'][k] for k in KEYS},
                train_monitor={k:h['monitor_full'][k] for k in KEYS}) for h in raw]
        if (folder/'completion.json').exists():
            done = json.loads((folder/'completion.json').read_text())
            assert done['provenance_sha256'] == signature
            assert done['best_sha256'] == sha256_file(folder/'best.pt')
            record.update(state='completed', best_epoch=done['best_epoch'], optimizer_steps=done['optimizer_steps'])
            scores = json.loads((folder/'outer_metrics.json').read_text())
            record['outer'] = {mode:{key:scores[mode][key] for key in KEYS} for mode in MODES}
            record['best_sha256'] = done['best_sha256']
        records.append(record)
    complete = [r for r in records if r['state']=='completed']
    aggregate = {}
    if complete:
        for mode in MODES:
            aggregate[mode] = {key:dict(mean=float(np.mean([r['outer'][mode][key] for r in complete])),
                sd=float(np.std([r['outer'][mode][key] for r in complete],ddof=1)) if len(complete)>1 else None)
                for key in KEYS}
    state = 'failed' if failures else 'completed' if len(complete)==5 else 'running'
    summary = dict(state=state, updated_unix=time.time(), completed_folds=len(complete),
        fits=5, folds=records, aggregate=aggregate, failures=failures,
        historical_outer_reuse=True, provenance_sha256=signature,
        warning='Fold0 and historical outer outcomes were used in prior research; repeated internal evaluation, not external validation.')
    write(public/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.7),constrained_layout=True)
    for ax,key,title in zip(axes,KEYS,['C-index (higher)','Annual NLL (lower)','IPCW IBS (lower)']):
        for r in records:
            h=r['history']
            if not h:continue
            line,=ax.plot([a['epoch'] for a in h],[a['validation'][key] for a in h],label=f"Fold {r['fold']+1}: inner")
            ax.plot([a['epoch'] for a in h],[a['train_monitor'][key] for a in h],'--',color=line.get_color(),alpha=.45)
        ax.set(title=title,xlabel='Epoch')
        if ax.lines:ax.legend(fontsize=7)
        else:ax.text(.5,.5,'First epoch in progress',ha='center',transform=ax.transAxes)
    fig.suptitle('R3 identity gate + exposure | solid: inner validation; dashed: fixed Train subset')
    for ext in ['png','svg','pdf']:fig.savefig(public/('learning_curves.'+ext),dpi=160)
    plt.close(fig)
    if complete:
        fig,axes=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True)
        for ax,key,title in zip(axes,KEYS,['Outer C-index (higher)','Outer annual NLL (lower)','Outer IPCW IBS (lower)']):
            means=[aggregate[m][key]['mean'] for m in MODES]
            sd=[aggregate[m][key]['sd'] or 0 for m in MODES]
            ax.bar(range(3),means,yerr=sd,capsize=4,color=['#8ba6bf','#6495b8','#386f96'])
            for r in complete:ax.scatter(range(3),[r['outer'][m][key] for m in MODES],s=13,color='black',zorder=3)
            ax.set(xticks=range(3),xticklabels=['Coarse only','20% fine','Full fine'],title=title)
        fig.suptitle(f'Fixed-best outer evaluation | completed {len(complete)}/5 | bars mean +/- fold SD')
        for ext in ['png','svg','pdf']:fig.savefig(public/('outer_results.'+ext),dpi=160)
        plt.close(fig)
    table=''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [r['fold']+1,r['state'],r.get('epoch','—'),r.get('optimizer_steps',0),r.get('best_epoch','—')])+'</tr>' for r in records)
    content='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>改进R3五折实验</title>
<style>body{font:16px system-ui;max-width:1180px;margin:32px auto;color:#182938;padding:0 20px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #dde5ee;text-align:left}img{width:100%}.note{background:#eef4fa;padding:16px}</style>'''
    content+=f'<h1>改进R3：CRC五折实验</h1><p>已完成 {len(complete)}/5 折；状态 {state}；更新时间 '+time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())+'</p>'
    content+='<p class="note">恒等粗门控＋真实随访时长似然。550位患者，原外层五折；每折内部验证选模型，固定后才评价该折留出患者。历史已接触这些数据，因此属于重复内部评估。实线为内部验证，虚线为固定训练子集；C-index越高越好，NLL和IBS越低越好。</p>'
    content+='<table><tr><th>折</th><th>状态</th><th>当前轮</th><th>实际更新</th><th>最佳轮</th></tr>'+table+'</table><h2>训练与内部验证曲线</h2><img src="learning_curves.png">'
    if complete:content+='<h2>已完成折的固定模型留出结果</h2><p>柱为均值、误差条为折间标准差；未完成五折时为部分结果。</p><img src="outer_results.png">'
    content+='<p><a href="summary.json">匿名指标JSON</a> · <a href="learning_curves.pdf">曲线PDF</a></p></html>'
    temporary=public/'report.partial.html';temporary.write_text(content,encoding='utf-8');temporary.replace(public/'report.html')
    if state=='completed':
        for name,digest in json.loads((root/'provenance.json').read_text())['files'].items():assert sha256_file(root/name)==digest,name
        write(root/'suite_completed.json',dict(state='completed',fits=5,provenance_sha256=signature,
              best_sha256=[r['best_sha256'] for r in records],at=time.time(),source_verified=True))
    with tarfile.open(root/'public.partial.tar.gz','w:gz') as archive:
        for p in public.iterdir():
            if p.suffix in ['.json','.html','.png','.svg','.pdf']:archive.add(p,arcname=p.name)
    (root/'public.partial.tar.gz').replace(root/'public.tar.gz')
    return dict(state=state,completed_folds=len(complete),optimizer_steps=[r.get('optimizer_steps',0) for r in records])


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--watch',action='store_true')
    args=p.parse_args();end=time.time()+72*3600
    while True:
        try:status=build(args.root);print(json.dumps(status),flush=True)
        except Exception:
            import traceback
            traceback.print_exc()
            if not args.watch:raise
            status={}
        if not args.watch or status.get('state') in ['completed','failed'] or time.time()>end:break
        time.sleep(45)


if __name__=='__main__':main()
