"""Frozen-input window scan: two videos, TextVQA and GSM8K. No training."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent


def run_logged(command,env,destination):
    if destination.exists():
        raise FileExistsError(f'choose a fresh cell directory: {destination}')
    print(f'Running: {destination}',flush=True)
    with destination.with_suffix('.log').open('w') as log:
        with subprocess.Popen(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as child:
            for line in child.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
            status=child.wait()
            if status:raise subprocess.CalledProcessError(status,command)


def checkpoints(args):
    if args.checkpoint:
        result={label:Path(p).expanduser().resolve(strict=True) for label,p in args.checkpoint}
        if len(result)!=len(args.checkpoint): raise ValueError('duplicate checkpoint labels')
    else:
        result={
            'dlite_20k':args.workspace/'runs/vlm_three_3ep_20261021_045040_2983971/dlite/last',
            'dlite_100k':args.workspace/'runs/vlm_dlite_dflash_3ep_mb2_b8_20261022_060311_3121049/dlite/last'}
        result={k:p.resolve(strict=True) for k,p in result.items()}
    for label,path in result.items():
        if label=='ar' or not label or any(not(c.isalnum() or c in '-_') for c in label):
            raise ValueError('safe unique model labels, excluding ar, required')
        cfg=json.loads((path/'config.json').read_text())
        d=cfg.get('dlite_config',{})
        if cfg.get('architectures')!=['DLiteDraftModel'] or d.get('model_role')!='student':
            raise ValueError(f'{label}: original DLite student required')
        if not list(path.glob('*.safetensors')) and not list(path.glob('*.bin')):
            raise ValueError(f'{label}: missing weights')
    return result


def report(root,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (root/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    lines=['# DLite recent-position window', '',
        'K includes the current pivot; each time position retains all selected Target layers.',
        'B is replicated batch size, not online concurrency. Greedy, per_token, decode-only raw ratios.',
        'Inference overrides on old checkpoints are zero-shot interventions, not trained-window models.', '',
        '| Dataset | Model | K | B | Accept | First hit | tok/s | Speedup | Prefill s | Peak GiB | Match |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        tau=r['accept_length']
        hit=r.get('first_candidate_hit_rate')
        lines.append(f"| {r['dataset']} | {r['model']} | {r['window'] if r['window'] else '—'} | {r['batch_size']} | {f'{tau:.3f}' if tau is not None else '—'} | {f'{hit:.1%}' if hit is not None else '—'} | {r['tok_s']:.2f} | {r['speedup']:.3f}× | {r['prefill_seconds']:.2f} | {r['peak_gib']:.3f} | {r['matched']}/{r['requests']} |")
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--target-model',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--checkpoint',nargs=2,action='append',default=[])
    p.add_argument('--windows',nargs='+',type=int,default=[1,2,5,10])
    p.add_argument('--trained-window',action='store_true',help='use each checkpoint saved K; do not override')
    p.add_argument('--tests',nargs='+',default=['Video-MME','VideoDetailCaption','textvqa','text-only'])
    p.add_argument('--samples',type=int,default=16)
    p.add_argument('--batch-sizes',nargs='+',type=int,default=[1])
    p.add_argument('--max-new-tokens',type=int,default=256)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--frames',type=int,default=16)
    p.add_argument('--max-pixels-per-frame',type=int,default=448*448)
    p.add_argument('--image-max-tokens',type=int,default=1024)
    p.add_argument('--video-data-root',type=Path,default=Path.home()/'LMM/liyt/datasets/eval_bench')
    args=p.parse_args()
    if min(args.samples,args.max_new_tokens,args.image_max_tokens,*args.windows,*args.batch_sizes)<1 or len(set(args.windows))!=len(args.windows):
        p.error('unique positive windows and positive budgets required')
    if len(set(args.tests))!=len(args.tests): p.error('duplicate tests')
    models=checkpoints(args)
    from video_benchmark_data import DATASETS,select_data as video_select
    from run_vlm_matrix import select_data as image_select
    videos=[x for x in args.tests if x in DATASETS]
    still=[x for x in args.tests if x not in DATASETS]
    video_records=video_select(args.video_data_root,videos,args.samples,args.seed) if videos else {}
    packets=image_select(SimpleNamespace(tests=still,samples=args.samples,seed=args.seed)) if still else {}
    root=args.output_dir.expanduser().resolve();root.mkdir(parents=True,exist_ok=False)
    settings=dict(arguments=vars(args),checkpoints={label:dict(path=str(path),config_sha256=hashlib.sha256((path/'config.json').read_bytes()).hexdigest(),saved_window=json.loads((path/'config.json').read_text()).get('dlite_config',{}).get('context_window_size',1)) for label,path in models.items()},snapshots={})
    selected_files={}
    for case,records in video_records.items():
        selected_files[case]=root/f'{case}_selection.json'
        selected_files[case].write_text(json.dumps({case:records},indent=2))
    for case,packet in packets.items():
        selected_files[case]=root/f'{case}.jsonl'
        selected_files[case].write_text(packet['serialized'])
    settings['snapshots']={case:dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for case,path in selected_files.items()}
    settings['pivot_augmentation_off'] = os.environ.get('PIVOT_AUGMENTATION_OFF') == '1'
    for label,path in models.items():
        settings['checkpoints'][label]['pivot_augmentation'] = json.loads((path/'config.json').read_text()).get('dlite_config',{}).get('pivot_augmentation')
    (root/'settings.json').write_text(json.dumps(settings,default=str,indent=2))
    target_cfg=json.loads((args.target_model.expanduser()/'config.json').read_text())['vision_config']
    pixels=args.image_max_tokens*(target_cfg['patch_size']*target_cfg['spatial_merge_size'])**2
    rows=[]
    for window in ([None] if args.trained_window else args.windows):
        for case,snapshot in selected_files.items():
            dest=root/f'{case}_{"trained" if window is None else "k"+str(window)}'
            if case in DATASETS:
                command=[sys.executable,str(HERE/'benchmark_video.py'),'--workspace',str(args.workspace),
                    '--target-model',str(args.target_model),'--datasets',case,'--selection-file',str(snapshot),
                    '--samples',str(args.samples),'--frames',str(args.frames),'--max-pixels-per-frame',str(args.max_pixels_per_frame)]
            else:
                command=[sys.executable,str(HERE/'benchmark_vlm_static_batch.py'),'--target-model',str(args.target_model),
                    '--data',str(snapshot),'--dataset-name',case,'--max-samples',str(args.samples),
                    '--image-max-pixels',str(pixels),'--cpu-inputs','--attention-backend','sdpa','--state-mode','per_token']
            command += ['--output-dir',str(dest),'--batch-sizes',*map(str,args.batch_sizes),'--max-new-tokens',str(args.max_new_tokens),'--seed',str(args.seed)]
            if window is not None: command += ['--context-window-size',str(window)]
            for label,path in models.items(): command += ['--checkpoint',label,str(path)]
            run_logged(command,dict(os.environ),dest)
            for row in json.loads((dest/'summary.json').read_text()):
                video=case in DATASETS
                label=row['model']
                rows.append(dict(dataset=case,model=label,window=0 if label=='ar' else (row.get('context_window_size') or window or 1),
                    batch_size=row['batch_size'],accept_length=row['mean_committed_length'],
                    first_candidate_hit_rate=row.get('first_candidate_hit_rate'),
                    tok_s=row['decode_tok_s'] if video else row['output_tokens_per_second'],
                    speedup=row['raw_decode_ratio_vs_ar'] if video else row['raw_throughput_speedup_vs_ar'],
                    prefill_seconds=row['prefill_seconds'],peak_gib=row['peak_allocated_gib'],
                    matched=row['matched'],requests=row['requests'],
                    interpretation='trained_window' if args.trained_window else 'zero_shot_override',
                    checkpoint=str(models[label]) if label in models else None))
            report(root,rows)
    print(f'Report: {root}/comparison.md',flush=True)


if __name__=='__main__':main()

