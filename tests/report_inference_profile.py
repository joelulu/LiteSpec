"""Aggregate optional diagnostic passes; never substitute these for benchmark rates."""
import argparse
from collections import Counter
import json
from pathlib import Path


def summarize_profile(results, reference_results):
    stages, host, counts = Counter(), Counter(), Counter()
    for row in results:
        stages.update(row['profile']['stage_seconds'])
        host.update(row['profile']['host_stage_seconds'])
        counts.update(row['profile']['stage_calls'])
    seconds = sum(row['decode_seconds'] for row in results)
    normal = sum(row['decode_seconds'] for row in reference_results)
    rounds = sum(len(row['accept_lengths']) for row in results)
    calls = sum(row['target_decode_calls'] for row in results)
    replays = sum(row['recovery_replays'] for row in results)
    tokens = sum(row['output_tokens'] for row in results)
    return dict(profile_samples=len(results), profile_decode_seconds=seconds,
        matching_unprofiled_decode_seconds=normal, instrumentation_ratio=seconds/normal if normal else None,
        output_tokens=tokens, speculative_rounds=rounds, recovery_replays=replays,
        replay_fraction=replays/rounds if rounds else None, target_decode_calls=calls,
        target_calls_per_round=calls/rounds if rounds else None,
        ms_per_round=1000*seconds/rounds if rounds else None,
        ms_per_target_call=1000*seconds/calls if calls else None,
        mean_committed_length=tokens/results[0]['batch_size']/rounds if rounds else None,
        stage_seconds=dict(stages), host_stage_seconds=dict(host), stage_calls=dict(counts),
        stage_percent={k:100*v/seconds for k,v in stages.items()},
        unassigned_seconds=seconds-sum(stages.values()),
        tokens_match_unprofiled=all(a['token_ids']==b['token_ids'] for a,b in zip(results,reference_results)),
        clock=results[0]['profile']['clock'],
        note='Separate instrumented pass. CUDA stream intervals include launch gaps; host times overlap CUDA times. Not benchmark throughput.')


def report(root):
    root = Path(root)
    lines = ['# Decode cost diagnostics', '',
        'Separate profiling passes; use comparison.csv for uninstrumented speedup. ',
        'CUDA stream stage intervals include host launch gaps; they are not pure kernel sums. ',
        'Host times overlap stream intervals and must not be added to them.', '']
    for path in sorted(root.rglob('decode_profile.json')):
        rows = json.loads(path.read_text())
        lines += [f'## {path.parent.relative_to(root)}', '',
            '| Model | B | Samples | ms/round | Replay % | Target calls/round | Profile/normal time |',
            '|---|---:|---:|---:|---:|---:|---:|']
        def number(value):
            return '—' if value is None else f'{value:.3f}'
        for row in rows:
            lines.append(f"| {row['model']} | {row['batch_size']} | {row['profile_samples']} | "
                f"{number(row['ms_per_round'])} | {number(None if row['replay_fraction'] is None else row['replay_fraction']*100)} | "
                f"{number(row['target_calls_per_round'])} | {number(row['instrumentation_ratio'])} |")
        for row in rows:
            lines += ['', f"### {row['model']} B={row['batch_size']}", '',
                '| Stage | Stream/CPU ms | ms/call | Decode % | Host ms (overlapping) | Calls |',
                '|---|---:|---:|---:|---:|---:|']
            for stage, seconds in sorted(row['stage_seconds'].items(), key=lambda item:-item[1]):
                lines.append(f"| {stage} | {1000*seconds:.3f} | {1000*seconds/row['stage_calls'][stage]:.3f} | {row['stage_percent'][stage]:.2f} | "
                    f"{1000*row['host_stage_seconds'].get(stage,0):.3f} | {row['stage_calls'][stage]} |")
            lines += ['', f"Tokens equal to unprofiled pass: {row['tokens_match_unprofiled']}. "
                f"Unassigned time: {1000*row['unassigned_seconds']:.3f} ms."]
            probe = row.get('kernel_probe')
            if probe:
                lines += ['', 'Actual HF Python fallback entries (includes prefill, identified by sequence length):',
                          '```json', json.dumps(probe['python_fallback_calls'], indent=2), '```',
                          'GDN/conv-related GPU kernel names (see JSON for the full list):',
                          '```text', '\n'.join(name for name in probe['cuda_kernel_names']
                              if any(key in name.lower() for key in ('gdn','delta','conv','recurrent','chunk'))) or '(none matched)', '```',
                          f"CUDA trace: `{probe['trace_file']}`.",
                          'Package availability alone does not establish which kernel ran.']
    output = root / 'inference_profile.md'
    output.write_text('\n'.join(lines)+'\n')
    print(f'Decode cost report: {output}', flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    report(parser.parse_args().root)
