"""Full-only correct/wrong/mask image benchmarks using prior frozen inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys

from evaluate_dflash_all import execute, fingerprint, write_json

HERE = Path(__file__).resolve().parent
DATASETS = ('charxiv_val', 'textvqa', 'flickr30k', 'hr_bench_4k', 'hr_bench_8k')
CONDITIONS = ('correct', 'wrong', 'mask')


def source_directory(workspace, explicit):
    if explicit is not None:
        return explicit.expanduser().resolve(strict=True)
    manifests = list((workspace.expanduser()/'runs').glob(
        'dflash_exp1_10w_5ep_*/eval_all_b1_n32/manifest.json'))
    if not manifests:
        raise FileNotFoundError('No experiment-1 evaluation manifest found; set SOURCE_EVAL_DIR explicitly')
    return max(manifests, key=lambda path: path.stat().st_mtime_ns).parent.resolve()


def prepare(args):
    source = source_directory(args.workspace, args.source_eval_dir)
    manifest = json.loads((source/'manifest.json').read_text())
    settings = manifest['settings']
    if (settings['samples'], settings['batch_size'], settings['max_new_tokens']) != (32, 1, 256):
        raise ValueError('Source evaluation must use 32 samples, B=1 and max_new_tokens=256')
    checkpoint = fingerprint(manifest['checkpoints']['full']['path'])
    if checkpoint != manifest['checkpoints']['full']:
        raise ValueError('Full checkpoint files changed after the source evaluation')
    config = json.loads((Path(checkpoint['path'])/'config.json').read_text())
    if config.get('architectures') != ['DFlashDraftModel'] or config.get('vlm_config', {}).get('draft_context','full') != 'full':
        raise ValueError('Expected the experiment-1 Full DFlash checkpoint')
    target = fingerprint(manifest['target']['path'])
    if target != manifest['target']:
        raise ValueError('Target files changed after the source evaluation')
    selected = {job['dataset']: job for job in manifest['jobs'] if job['kind'] == 'image'}
    packets = {}
    for name in DATASETS:
        original = selected[name]
        if original['samples'] != 32:
            raise ValueError(f'{name}: expected 32 frozen samples')
        snapshot = Path(original['snapshot']).read_bytes()
        if hashlib.sha256(snapshot).hexdigest() != original['input_sha256']:
            raise ValueError(f'{name}: source inputs changed')
        records = [json.loads(line) for line in snapshot.splitlines() if line.strip()]
        if len(records) != 32 or len({row['id'] for row in records}) != 32:
            raise ValueError(f'{name}: expected exactly 32 unique frozen samples')
        packets[name] = snapshot

    root = args.output_dir.expanduser().resolve()
    if root.exists():
        raise FileExistsError(f'Choose a fresh output directory: {root}')
    for directory in ('inputs', 'logs', 'results'):
        (root/directory).mkdir(parents=True, exist_ok=True)
    jobs = []
    for index, name in enumerate(DATASETS):
        gpu = args.gpus[index % len(args.gpus)]
        snapshot = root/'inputs'/f'{name}.jsonl'
        snapshot.write_bytes(packets[name])
        for condition in CONDITIONS:
            output = root/'results'/name/condition
            output.parent.mkdir(exist_ok=True)
            cmd = [sys.executable, str(HERE/'benchmark_vlm_static_batch.py'),
                   '--target-model', target['path'], '--checkpoint', 'full', checkpoint['path'],
                   '--data', str(snapshot), '--dataset-name', name,
                   '--max-samples', '32', '--batch-sizes', '1', '--max-new-tokens', '256',
                   '--seed', str(settings['seed']), '--warmup', '1',
                   '--image-max-pixels', str(manifest['image_max_pixels']), '--cpu-inputs',
                   '--attention-backend', 'sdpa', '--state-mode', 'per_token',
                   '--visual-condition', condition, '--output-dir', str(output)]
            if condition != 'correct':
                cmd += ['--ar-reference', str(output.parent/'correct')]
            jobs.append(dict(dataset=f'{name}/{condition}', kind='image', gpu=gpu,
                samples=32, models=['ar','full'], snapshot=str(snapshot), output=str(output),
                log=str(root/'logs'/f'{name}_{condition}.log'), command=cmd))
    write_json(root/'manifest.json', dict(source_evaluation=str(source), target=target, checkpoint=checkpoint,
        settings=dict(samples=32, batch_size=1, max_new_tokens=256, seed=settings['seed'],
                      gpus=args.gpus, image_max_pixels=manifest['image_max_pixels']),
        intervention='Draft visual-position Target hidden only; Target images and text hidden remain correct',
        wrong_donor_prefill_excluded=True, jobs=jobs))
    return root, jobs


def visual_report(root):
    rows = json.loads((root/'comparison.json').read_text())
    lines = ['# Full visual-context intervention', '',
        'Five image datasets; identical 32 prompts per dataset, B=1, max_new_tokens=256.',
        'Target inputs stay correct; only Draft visual-position features/attention are intervened on.',
        'Wrong replaces visual features with a different-image donor; its extra donor prefill is excluded from timing.',
        'Mask removes Draft attention to visual positions. Text hidden and positions remain unchanged.',
        'Wrong/Mask reuse Correct\'s same-GPU AR baseline. All samples contribute, including AR mismatches.', '',
        '| Dataset | GPU | Condition | Committed | First hit | tok/s | Speedup | AR match | Peak GiB |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['model'] != 'full':
            continue
        name, condition = row['dataset'].split('/')
        hit = '—' if row['first_hit_rate'] is None else f"{row['first_hit_rate']:.1%}"
        lines.append(f"| {name} | {row['gpu']} | {condition} | {row['committed_length']:.3f} | {hit} | "
                     f"{row['tok_s']:.2f} | {row['speedup']:.3f}x | {row['matched']}/{row['requests']} | {row['peak_gib']:.3f} |")
    statuses = json.loads((root/'status.json').read_text())
    for name, status in statuses.items():
        if status['status'] != 'success':
            lines += ['', f"{name}: {status['status']}; see {status.get('log', 'logs/')}"]
    path = root/'visual_comparison.md'
    path.write_text('\n'.join(lines)+'\n')
    print(path.read_text(), flush=True)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--source-eval-dir', type=Path, help='prior experiment-1 eval_all_b1_n32 directory')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpus', nargs='+', default=['4','5','6','7'])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= len(args.gpus) <= 8 or len(set(args.gpus)) != len(args.gpus):
        parser.error('one to eight distinct GPU selectors required')
    root, jobs = prepare(args)
    frozen = json.loads((root/'manifest.json').read_text())
    print(f"Source evaluation: {frozen['source_evaluation']}\nFull checkpoint: {frozen['checkpoint']['path']}", flush=True)
    for job in jobs:
        print(f"GPU {job['gpu']}: {job['dataset']}\n{shlex.join(job['command'])}", flush=True)
    print(f'Results: {root}\nMonitor: tail -f {root}/logs/*.log', flush=True)
    if args.dry_run:
        return 0
    result = execute(root, jobs, 1)
    print(f'Report: {visual_report(root)}', flush=True)
    return result


if __name__ == '__main__': raise SystemExit(main())
