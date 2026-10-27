"""Same-checkpoint closed-loop controls and untimed fixed-AR-prefix reader probes."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def run_logged(command, directory):
    print(f'Closed-loop condition: {directory}', flush=True)
    with directory.with_suffix('.log').open('w') as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as child:
            for line in child.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = child.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)


def summarize(rows):
    from memory_attention import ATTENTION_FIELDS
    groups = defaultdict(list)
    for row in rows:
        groups[row['dataset'], row['condition']].append(row)
    fields = ('gate_mean', 'gate_min', 'gate_max', 'gate_below_001_fraction',
        'residual_query_norm_ratio', 'applied_query_norm_ratio', 'query_changed_fraction',
        'logits_mean_abs_delta', 'logits_max_abs_delta', 'logits_relative_rms_delta',
        'probability_l1', 'kl_correct_to_condition', 'proposal_token_agreement',
        'proposal_block_exact', 'residual_cosine_vs_correct', 'residual_difference_ratio_vs_correct',
        'memory_tokens', 'target_next_matches_reference', *ATTENTION_FIELDS)
    result = []
    for (dataset, condition), group in groups.items():
        entry = dict(dataset=dataset, condition=condition, probes=len(group),
                     samples=len({r['sample_id'] for r in group}))
        for field in fields:
            values = [r[field] for r in group if r.get(field) is not None]
            entry[field] = sum(values)/len(values) if values else None
        entry['gate_min'] = min((r['gate_min'] for r in group if r.get('gate_min') is not None), default=None)
        entry['gate_max'] = max((r['gate_max'] for r in group if r.get('gate_max') is not None), default=None)
        entry['logits_max_abs_delta'] = max(r['logits_max_abs_delta'] for r in group)
        result.append(entry)
    return result


def report(root, paired, closed):
    summaries = summarize(paired)
    write_json(root/'paired_summary.json', summaries)
    if summaries:
        with (root/'paired_summary.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    lines = ['# Memory diagnosis', '',
        'One vision checkpoint; no parameter updates. Fixed-prefix probes are untimed.',
        'All conditions share each target prefix/pivot; their proposals never enter the probe target cache.', '',
        '## Closed-loop performance', '',
        'Decode-only raw speedup; correctness matches remain in the per-cell outputs.',
        'Wrong-image end-to-end time includes donor prefill and is diagnostic only.', '',
        '| Dataset | Condition | Accept length | Speedup | Decode tok/s |',
        '|---|---|---:|---:|---:|']
    for r in closed:
        tau = r['mean_committed_length']
        lines.append(f"| {r['dataset_name']} | {r['model']} | {f'{tau:.4f}' if tau else '—'} | "
            f"{r['raw_throughput_speedup_vs_ar']:.3f}× | {r['decode_tokens_per_second']:.2f} |")
    lines += ['', '## Fixed-prefix sensitivity', '',
        'Logits/proposals compared with correct memory at the identical prefix.',
        'Applied Δ/query measures actual model-dtype addition, including BF16 rounding.', '',
        '| Dataset | Condition | Probes | Gate mean | Residual/query | Applied Δ/query | Query changed | Proposal agreement | Mean abs Δlogit |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        gate = f"{r['gate_mean']:.5f}" if r['gate_mean'] is not None else '—'
        lines.append(f"| {r['dataset']} | {r['condition']} | {r['probes']} | {gate} | "
            f"{r['residual_query_norm_ratio']:.3e} | {r['applied_query_norm_ratio']:.3e} | "
            f"{100*r['query_changed_fraction']:.2f}% | {100*r['proposal_token_agreement']:.2f}% | {r['logits_mean_abs_delta']:.3e} |")
    lines += ['', '## Visual-memory attention (FP32 reconstruction)', '',
        'Attention is normalized only over visual memory: its total mass is always 1.',
        'Uniform attention has normalized entropy 1, effective fraction 1 and peak/uniform 1.',
        'Top-K mass depends on memory size. Sharp attention is not proof of useful evidence.', '',
        '| Dataset | Condition | Memory N | Score std | Peak/uniform | Top16 mass | Norm entropy | Effective N | Readout/value | Δ from uniform/value |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        if r.get('attention_valid_tokens') is None:
            continue
        lines.append(f"| {r['dataset']} | {r['condition']} | {r['attention_valid_tokens']:.1f} | "
            f"{r['attention_logit_std']:.4f} | {r['attention_peak_vs_uniform']:.2f} | "
            f"{r['attention_top16_mass']:.3f} | {r['attention_entropy_normalized']:.3f} | "
            f"{r['attention_effective_tokens']:.1f} | {r['attention_readout_value_ratio']:.3f} | "
            f"{r['attention_readout_uniform_delta_ratio']:.3f} |")
    lines += ['', '## Gradient probe', '',
        'See gradient_probe.json: held-out-prefix teacher-forced CE, not historical training gradients.',
        'No optimizer is constructed. Weight values are never updated; gradients are cleared.',
        'Gate depends only on pivot in this checkpoint, so correct/wrong gate equality is expected.',
        'Small effect alone does not distinguish redundant visual evidence from an ineffective reader.']
    (root/'diagnostics.md').write_text('\n'.join(lines)+'\n')
    (root/'attention.md').write_text('\n'.join(lines[lines.index('## Visual-memory attention (FP32 reconstruction)'):lines.index('## Gradient probe')])+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-group', type=Path, required=True)
    parser.add_argument('--evaluation-dir', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--datasets', nargs='+', default=['charxiv_val', 'textvqa'])
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--probe-stride', type=int, default=8)
    parser.add_argument('--gradient-samples', type=int, default=1)
    parser.add_argument('--probe-only', action='store_true', help='use original AR traces/donor mappings; omit fresh timing')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if min(args.samples, args.max_new_tokens, args.probe_stride) < 1 or args.gradient_samples < 0:
        parser.error('positive sample/token/stride limits and nonnegative gradient sample count required')
    if not args.datasets or len(args.datasets) != len(set(args.datasets)) or any(
            name not in {'charxiv_val', 'textvqa', 'flickr30k', 'hr_bench_4k', 'hr_bench_8k'} for name in args.datasets):
        parser.error('select unique named vision datasets')
    group = args.run_group.expanduser().resolve()
    evaluation = (args.evaluation_dir or group/'evaluation').expanduser().resolve()
    root = args.output_dir.expanduser().resolve()
    cases, checkpoint = [], args.checkpoint.expanduser().resolve() if args.checkpoint else None
    for name in args.datasets:
        prior = evaluation/f'{name}_correct'
        settings = load_json(prior/'settings.json')
        prior_vision = next(r for r in load_json(prior/'summary.json') if r['model'] == 'vision')
        candidate = Path(prior_vision['checkpoint']).expanduser().resolve()
        if checkpoint is None:
            checkpoint = candidate
        if args.checkpoint is None and candidate != checkpoint:
            raise ValueError('datasets were evaluated with different vision checkpoints')
        records = [json.loads(line) for line in (evaluation/f'{name}.jsonl').read_text().splitlines() if line.strip()]
        if len(records) < args.samples:
            raise ValueError(f'need {args.samples} saved samples for {name}')
        records = records[:args.samples]
        ids = [r['id'] for r in records]
        if ids != settings['sample_ids'][:args.samples] or len(set(ids)) != len(ids):
            raise ValueError('sample snapshot no longer matches original evaluation')
        cases.append((name, settings, records))
    config = load_json(checkpoint/'config.json')
    if config.get('dlite_config', {}).get('memory_mode') != 'vision':
        raise ValueError('requires the trained DLite vision-memory checkpoint')
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/'settings.json', {
            'arguments': {k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
            'checkpoint':str(checkpoint), 'checkpoint_config_sha256':hashlib.sha256((checkpoint/'config.json').read_bytes()).hexdigest(),
            'device_visibility':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'datasets': {name:dict(source=str(evaluation/f'{name}.jsonl'),
                selected_ids=[r['id'] for r in records],
                selected_records_sha256=hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest())
                for name, settings, records in cases},
            'fixed_prefix_backend':'native_HF_single_token_teacher_forcing',
            'note':'probe timings are invalid; performance is measured in isolated closed-loop subprocesses'})
    import torch
    from litespec.vlm.runtime import load_target, prepare_example
    from litespec.vlm.baselines import load_draft
    from memory_probe import probe_reference
    paired, closed, gradients = [], [], []
    for name, settings, records in cases:
        snapshot = root/f'{name}.jsonl'
        snapshot.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records))
        size = settings.get('effective_image_processor_size')
        if not size:
            raise ValueError('original image processor size is missing')
        if args.probe_only:
            source = evaluation/f'{name}_correct'
            wrong_source = evaluation/f'{name}_wrong_image'
            if args.max_new_tokens > settings['max_new_tokens']:
                raise ValueError('probe-only cannot extend original AR traces')
        else:
            common = [sys.executable, str(Path(__file__).with_name('benchmark_vlm_static_batch.py')),
                '--target-model', settings['target_model'], '--data', str(snapshot), '--dataset-name', name,
                '--max-samples', str(args.samples), '--max-new-tokens', str(args.max_new_tokens),
                '--batch-sizes', '1', '--cpu-inputs', '--attention-backend', settings['actual_attention_backend'],
                '--image-min-pixels', str(size['shortest_edge']), '--image-max-pixels', str(size['longest_edge']),
                '--seed', str(settings['seed']), '--state-mode', settings.get('state_mode', 'per_token'), '--device', args.device]
            source, wrong_source = root/f'{name}_correct', root/f'{name}_wrong_image'
            for condition, dest in [('correct', source), ('wrong_image', wrong_source), ('disabled', root/f'{name}_disabled')]:
                cmd = [*common, '--output-dir', str(dest), '--checkpoint', f'vision_{condition}', str(checkpoint),
                       '--memory-condition', condition]
                if condition != 'correct':
                    cmd += ['--ar-reference', str(source)]
                run_logged(cmd, dest)
                closed.extend(r for r in load_json(dest/'summary.json') if condition == 'correct' or r['model'] != 'ar')
        traces = {r['sample_id']:r['result']['token_ids'] for r in (
            json.loads(line) for line in (source/'ar_b1.jsonl').read_text().splitlines() if line.strip())}
        mapping = load_json(wrong_source/'settings.json')['wrong_image_mapping']
        dtype = torch.float32 if args.device == 'cpu' else torch.bfloat16
        target, processor = load_target(settings['target_model'], device=args.device, dtype=dtype,
                                         attention=settings['actual_attention_backend'])
        processor.image_processor.size = dict(size)
        draft = load_draft(checkpoint, attn_implementation=settings['actual_attention_backend']).to(args.device, dtype).eval()
        weights = {n:dict(norm=float(p.detach().float().norm()), finite=bool(torch.isfinite(p).all()))
                   for n,p in draft.memory_reader.named_parameters()}
        write_json(root/f'{name}_reader_parameters.json', weights)
        with (root/f'{name}_paired.jsonl').open('w') as handle:
            for index, record in enumerate(records):
                if record['id'] not in traces or record['id'] not in mapping:
                    raise ValueError('missing AR trace or saved donor assignment')
                donor_image = mapping[record['id']]['donor_image']
                if Path(donor_image).resolve() == Path(record['image']).resolve():
                    raise ValueError('donor must use a different image')
                example = prepare_example(processor, target.config, record, device=args.device, base_dir=snapshot.parent)
                donor = prepare_example(processor, target.config, {**record, 'image':donor_image},
                                        device=args.device, base_dir=snapshot.parent)
                rows, grad = probe_reference(target, draft, example, donor,
                    traces[record['id']][:args.max_new_tokens], stride=args.probe_stride,
                    gradients=index < args.gradient_samples)
                for row in rows:
                    row.update(dataset=name, sample_id=record['id'], donor_image=donor_image)
                    handle.write(json.dumps(row)+'\n')
                handle.flush()
                paired.extend(rows)
                if grad is not None:
                    gradients.append(dict(dataset=name, sample_id=record['id'], **grad))
                print(f'Fixed-prefix probe {name}: {index+1}/{len(records)} samples', flush=True)
        del target, draft, processor
        if args.device != 'cpu':
            torch.cuda.empty_cache()
        write_json(root/'gradient_probe.json', gradients)
        write_json(root/'closed_loop_summary.json', closed)
        report(root, paired, closed)
        for r in summarize(paired):
            if r['dataset'] == name and r.get('attention_valid_tokens') is not None:
                print(f"ATTN {name} {r['condition']}: N={r['attention_valid_tokens']:.1f}, "
                    f"score_std={r['attention_logit_std']:.4f}, peak/uniform={r['attention_peak_vs_uniform']:.2f}, "
                    f"top16={r['attention_top16_mass']:.3f}, Hnorm={r['attention_entropy_normalized']:.3f}, "
                    f"effective_N={r['attention_effective_tokens']:.1f}, "
                    f"readout/value={r['attention_readout_value_ratio']:.3f}", flush=True)
    print(f'Report: {root / "diagnostics.md"}', flush=True)


if __name__ == '__main__':
    main()
