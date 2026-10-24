"""Evaluate three checkpoints plus wrong-image diagnosis on fixed sample snapshots."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from run_vlm_matrix import select_data


def run_logged(command, env, directory):
    if directory.exists():
        raise FileExistsError(f'choose a fresh evaluation directory: {directory}')
    print(f'Running evaluation: {directory}', flush=True)
    # Benchmark itself creates this directory; put the log beside it.
    with directory.with_suffix('.log').open('w') as log:
        with subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as child:
            for line in child.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            status = child.wait()
            if status:
                raise subprocess.CalledProcessError(status, command)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-group', required=True, type=Path)
    p.add_argument('--output-dir', type=Path)
    args = p.parse_args()
    group = args.run_group.expanduser().resolve()
    root = args.output_dir or group/'evaluation'
    root.mkdir(parents=True, exist_ok=False)
    target = Path(os.environ.get('TARGET_MODEL', str(Path.home()/'LMM/lmm_model/Qwen3.5-4B'))).expanduser()
    models = [(mode, group/mode/'last') for mode in ('none', 'vision', 'text')]
    configs = {}
    for mode, path in models:
        configs[mode] = json.loads((path/'config.json').read_text())
        if configs[mode]['dlite_config'].get('memory_mode', 'none') != mode:
            raise ValueError(f'checkpoint mode mismatch: {path}')
    tests = os.environ.get('TESTS', 'text-only charxiv_val flickr30k textvqa hr_bench_4k hr_bench_8k').split()
    data = select_data(SimpleNamespace(tests=tests, samples=int(os.environ.get('EVAL_SAMPLES', '64')),
                                     seed=int(os.environ.get('SAMPLE_SEED', '42'))))
    cfg = json.loads((target/'config.json').read_text())['vision_config']
    unit = cfg['patch_size'] * cfg['spatial_merge_size']
    image_tokens = int(os.environ.get('IMAGE_MAX_TOKENS', '1024'))
    training_budgets = {c.get('vlm_config', {}).get('image_max_tokens') for c in configs.values()}
    if training_budgets != {image_tokens}:
        raise ValueError(f'evaluation image budget differs from training: {training_budgets} vs {image_tokens}')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': os.environ.get('EVAL_GPU', '0')}
    rows = []
    for case, packet in data.items():
        snapshot = root/f'{case}.jsonl'
        snapshot.write_text(packet['serialized'])
        common = [sys.executable, str(Path(__file__).with_name('benchmark_vlm_static_batch.py')),
            '--target-model', str(target), '--data', str(snapshot), '--dataset-name', case,
            '--max-samples', str(packet['samples']), '--max-new-tokens', os.environ.get('EVAL_MAX_NEW_TOKENS', '256'),
            '--batch-sizes', *os.environ.get('BATCH_SIZES', '1').split(),
            '--image-max-pixels', str(image_tokens*unit*unit), '--cpu-inputs',
            '--state-mode', os.environ.get('STATE_MODE', 'per_token')]
        normal = root/f'{case}_correct'
        cmd = [*common, '--output-dir', str(normal)]
        for mode, path in models:
            cmd.extend(['--checkpoint', mode, str(path)])
        if os.environ.get('PROFILE_DECODE', '0') == '1':
            cmd.extend(['--profile-decode', '--profile-samples', os.environ.get('PROFILE_SAMPLES', '4')])
        run_logged(cmd, env, normal)
        rows.extend(json.loads((normal/'summary.json').read_text()))
        if case != 'gsm8k':
            conditions = ['wrong_image']
            if os.environ.get('EVAL_DISABLED', '0') == '1':
                conditions.append('disabled')
            for condition in conditions:
                dest = root/f'{case}_{condition}'
                run_logged([*common, '--output-dir', str(dest), '--ar-reference', str(normal),
                    '--checkpoint', f'vision_{condition}', str(group/'vision/last'),
                    '--memory-condition', condition], env, dest)
                rows.extend(r for r in json.loads((dest/'summary.json').read_text()) if r['model'] != 'ar')
        report(root, rows)
    print(f'Tables: {root / "memory_tables.md"}')


def report(root, rows):
    (root/'summary.json').write_text(json.dumps(rows, indent=2))
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (root/'comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ['Static replicated batch; decode-only speedup; target match is recorded but does not gate rates.',
             'Wrong-image end-to-end time includes an extra donor prefill; use it for diagnosis.', '',
             '| Dataset | B | Model | Accept length | Speedup | Decode tok/s | Memory tokens |',
             '|---|---:|---|---:|---:|---:|---:|']
    for r in rows:
        tau = r['mean_committed_length']
        lines.append(f"| {r['dataset_name']} | {r['batch_size']} | {r['model']} | "
            f"{f'{tau:.3f}' if tau is not None else '—'} | {r['raw_throughput_speedup_vs_ar']:.3f}× | "
            f"{r['output_tokens_per_second']:.1f} | {r.get('mean_memory_tokens', 0):.1f} |")
    (root/'memory_tables.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
