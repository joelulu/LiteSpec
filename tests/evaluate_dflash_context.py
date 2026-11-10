"""Matched B=1 context benchmark and paired visual-slot diagnosis."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from run_vlm_matrix import select_data
from litespec.vlm.dflash_context import CONTEXTS

HERE=Path(__file__).resolve().parent


def execute(cmd):
    print(' '.join(map(str,cmd)),flush=True)
    subprocess.run(list(map(str,cmd)),check=True)


def main():
    workspace=Path(os.environ['DLITE_WORKSPACE']).expanduser()
    group=Path(os.environ['RUN_GROUP']).expanduser().resolve(strict=True)
    root=Path(os.environ['RESULT_DIR']).expanduser().resolve()
    root.mkdir(parents=True,exist_ok=False)
    target=Path(os.getenv('TARGET_MODEL',str(Path.home()/'LMM/lmm_model/Qwen3.5-4B'))).expanduser()
    old=Path(os.getenv('DFLASH_100K',str(workspace/'runs/vlm_dlite_dflash_3ep_mb2_b8_20261022_060311_3121049/dflash/last'))).expanduser().resolve(strict=True)
    models={mode:group/mode/'last' for mode in CONTEXTS}
    models['dflash_100k']=old
    for path in models.values():
        if not (path/'config.json').is_file():raise FileNotFoundError(path)
    tests=os.getenv('TESTS','charxiv_val textvqa').split()
    samples=int(os.getenv('EVAL_SAMPLES','64')); seed=int(os.getenv('SAMPLE_SEED','42'))
    cap=int(os.getenv('EVAL_MAX_NEW_TOKENS','256')); image_tokens=int(os.getenv('IMAGE_MAX_TOKENS','1024'))
    vision=json.loads((target/'config.json').read_text())['vision_config']
    pixels=image_tokens*(vision['patch_size']*vision['spatial_merge_size'])**2
    packets=select_data(SimpleNamespace(tests=tests,samples=samples,seed=seed))
    all_rows=[]
    for dataset,packet in packets.items():
        snapshot=root/f'{dataset}_inputs.jsonl';snapshot.write_text(packet['serialized'])
        correct=root/f'{dataset}_correct'
        for condition in ('correct','wrong','mask'):
            dest=root/f'{dataset}_{condition}'
            command=[sys.executable,HERE/'benchmark_vlm_static_batch.py','--target-model',target,
                '--data',snapshot,'--dataset-name',dataset,'--max-samples',samples,'--batch-sizes',1,
                '--max-new-tokens',cap,'--seed',seed,'--image-max-pixels',pixels,'--cpu-inputs',
                '--attention-backend','sdpa','--state-mode','per_token','--visual-condition',condition,'--output-dir',dest]
            if condition!='correct':command+=['--ar-reference',correct]
            for label,path in models.items():
                if condition=='correct' or label in ('full','dflash_100k'):command+=['--checkpoint',label,path]
            execute(command)
            all_rows.extend(dict(r,dataset=dataset,condition=condition) for r in json.loads((dest/'summary.json').read_text()) if condition=='correct' or r['model']!='ar')
            (root/'comparison.json').write_text(json.dumps(all_rows,indent=2))
    lines=['# DFlash context ablation','',
        'B=1, per_token; decode-only raw speedups. Match is recorded, not a rate gate.',
        'Wrong-image donor prefill is excluded from reported times; Wrong is diagnostic only. Target inputs stay correct.','',
        '| Dataset | Model | Condition | Accept | First hit | tok/s | Speedup | Peak GiB | AR match |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in all_rows:
        accept=r['mean_committed_length'];hit=r.get('first_candidate_hit_rate')
        lines.append(f"| {r['dataset']} | {r['model']} | {r['condition']} | {f'{accept:.3f}' if accept is not None else '—'} | {f'{hit:.1%}' if hit is not None else '—'} | {r['output_tokens_per_second']:.2f} | {r['raw_throughput_speedup_vs_ar']:.3f}× | {r['peak_allocated_gib']:.3f} | {r['matched']}/{r['requests']} |")
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    execute([sys.executable,HERE/'probe_dflash_visual.py','--target-model',target,
        '--checkpoint','full',models['full'],'--checkpoint','dflash_100k',old,
        '--tests',*tests,'--samples',samples,'--max-new-tokens',cap,'--seed',seed,
        '--image-max-tokens',image_tokens,'--output-dir',root/'visual_probe'])
    print((root/'comparison.md').read_text())
    print(f'Results: {root}',flush=True)


if __name__=='__main__':main()
