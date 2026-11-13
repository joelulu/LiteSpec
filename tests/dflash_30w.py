"""Matched Full/Pivot1 x block8/16: common data, cache provenance and exports."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
from types import SimpleNamespace

import yaml

HERE = Path(__file__).resolve().parent
VARIANTS = {'full_b8': ('full', 8), 'pivot1_b8': ('pivot1', 8),
            'full_b16': ('full', 16), 'pivot1_b16': ('pivot1', 16)}
RECIPE = dict(algorithm='dflash', num_epochs=5, batch_size=2, accumulation_steps=32,
              num_draft_layers=5, num_anchors=512, max_length=4096,
              image_max_tokens=1024, learning_rate=0.0006, warmup_ratio=0.04,
              weight_decay=0., max_grad_norm=1., use_muon=False, seed=42,
              loss_decay_gamma=4., context_projection='shared', position_layer_bias=False,
              save_every_epoch=True, final_only=False, final_save=True,
              save_interval=10**12, log_interval=1)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def config(mode, block):
    # Independent of stale INIT_CHECKPOINT/BLOCK_SIZE/debug environment values.
    cfg = yaml.safe_load((HERE.parent / 'scripts/config/vlm/sft_dflash.yaml').read_text())
    cfg.update(RECIPE, dflash_context_experiment=True, draft_context=mode,
               block_size=block, context_experiment_variant=mode)
    return cfg


def verify_generation(generated, candidates=300000):
    """Every manifest ID must finish exactly once in train, val or skipped."""
    generated = Path(generated)
    prompts = generated / 'manifest/prompts.jsonl'
    meta = read(generated / 'manifest/manifest_config.json')
    generation = read(generated / 'generation_config.json')
    if sha(prompts) != meta['manifest_sha256'] or sha(prompts) != generation['manifest_sha256']:
        raise ValueError('Generation prompt manifest changed')
    expected = set()
    with prompts.open() as handle:
        for line in handle:
            if line.strip():
                key = json.loads(line)['id']
                if key in expected: raise ValueError('Duplicate candidate ID')
                expected.add(key)
    if len(expected) != candidates: raise ValueError('Candidate manifest count differs')
    completed, counts = set(), {}
    for name in ('train', 'val', 'skipped'):
        count = 0
        with (generated / f'{name}.jsonl').open() as handle:
            for line in handle:
                if not line.strip(): continue
                row = json.loads(line); key = row['id']
                if key not in expected or key in completed:
                    raise ValueError('Unknown/duplicate generation output ID')
                if name != 'skipped' and (row.get('split') != name or
                        row.get('answer_target') != generation['target_model'] or
                        not isinstance(row.get('answer'), str) or not row['answer'].strip()):
                    raise ValueError('Wrong generation split/Target or empty answer')
                completed.add(key); count += 1
        counts[name] = count
    if completed != expected: raise ValueError('Incomplete regeneration: rerun the data stage')
    return counts


def verify_common(root, target=None):
    root = Path(root).expanduser().resolve(strict=True)
    saved = read(root / 'manifest.json')
    if saved.get('schema') != 'dflash_common_b8_b16_v1':
        raise ValueError('Wrong common-data schema')
    if target is not None and str(Path(target).expanduser().resolve()) != saved['target_model']:
        raise ValueError('Target differs from common preprocessing')
    for path, digest in saved['files'].items():
        if sha(path) != digest:
            raise ValueError(f'Prepared data/provenance changed: {path}')
    from evaluate_dflash_all import fingerprint
    if fingerprint(saved['target_model']) != saved['target']:
        raise ValueError('Target model changed since preprocessing')
    return saved


def publish_common(output, source, records, validations, cfgs, metadata, reuse):
    """Only publish the intersection actually checked by BOTH original rules."""
    output, source = Path(output).resolve(), Path(source).resolve()
    if any(cfgs[b]['data_sha256'] != sha(source) for b in (8, 16)):
        raise ValueError('Generated data changed during preflight')
    valid = sorted(set(validations[8]['valid']) & set(validations[16]['valid']))
    if not valid:
        raise ValueError('No jointly eligible training samples')
    selected = [records[i] for i in valid]
    if len({r['id'] for r in selected}) != len(selected):
        raise ValueError('Duplicate training IDs')
    for row in selected:
        if not Path(row['image']).is_absolute():
            raise ValueError('Generated training images must have absolute paths')
    if output.exists():
        raise FileExistsError(f'Common output already exists: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.common-building-', dir=output.parent))
    try:
        data = temporary / 'train.jsonl'
        data.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in selected))
        write(temporary / 'selected_source_indices.json', valid)
        rejected = {}
        for block in (8, 16):
            rejected[str(block)] = dict(Counter(r['reason'] for r in validations[block]['skipped']))
            (temporary / f'source_skipped_b{block}.jsonl').write_text(''.join(
                json.dumps(row) + '\n' for row in validations[block]['skipped']))
            directory = temporary / f'validation_b{block}'
            directory.mkdir()
            cfg = dict(cfgs[block], train_data=str(output / 'train.jsonl'),
                       data_sha256=sha(data), output_dir=str(output / directory.name),
                       debug_max_samples=0, debug_max_steps=None, eligible_samples=len(selected),
                       eligible_indices_sha256=hashlib.sha256(json.dumps(list(range(len(selected)))).encode()).hexdigest(),
                       common_source=str(source), common_rule='intersection_of_real_b8_and_b16_preflights')
            write(directory / 'effective_config.json', cfg)
            (directory / 'skipped_training.jsonl').write_text('')
            # Uses the trainer's validator before publication. No image encoding.
            reuse(selected, cfg, directory)
            (directory / 'reuse_validation.env').write_text('export REUSE_VALIDATION_FROM=' +
                shlex.quote(str(output / directory.name)) + '\n')
        files = {str(output / p.relative_to(temporary)): sha(p)
                 for p in temporary.rglob('*') if p.is_file()}
        files.update(metadata.pop('source_files'))
        saved = dict(metadata, schema='dflash_common_b8_b16_v1', files=files,
                     source_training_records=len(records), eligible_samples=len(selected),
                     individually_eligible={str(b): len(validations[b]['valid']) for b in (8, 16)},
                     rejected_reasons=rejected, train_data=str(output / 'train.jsonl'))
        write(temporary / 'manifest.json', saved)
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return saved


def prepare_data(args):
    from evaluate_dflash_all import fingerprint
    from prepare_vlm_validation import prepare, load_dependencies
    generated = args.generated.expanduser().resolve(strict=True)
    output = args.common.expanduser().resolve()
    if output.exists():
        return verify_common(output, args.target_model)
    manifest = read(generated / 'manifest/manifest_config.json')
    counts = manifest['counts']
    candidate_count = counts.get('train', 0) + counts.get('val', 0)
    if manifest['settings']['max_samples'] != 300000 or candidate_count != 300000:
        raise ValueError(f'Expected 300000 unique candidate prompts; got {candidate_count}')
    generation = read(generated / 'generation_config.json')
    summary_path = generated / 'last_summary.json'
    summary = read(summary_path) if summary_path.exists() else {}
    expected = dict(target_model=str(args.target_model.expanduser().resolve()), temperature=0.,
                    enable_thinking=False, max_tokens=256, max_length=4096, min_answer_tokens=1,
                    answer_budget_policy='adaptive_suffix8')
    for key, value in expected.items():
        if generation.get(key) != value:
            raise ValueError(f'Generation setting differs: {key}')
    budget = generation.get('image_budget', {})
    if budget.get('max_pixels') != 1024 * budget.get('factor', 0)**2 or not budget.get('factor'):
        raise ValueError('Generation must use image_max_tokens=1024')
    # A resumed generator can finish every ID before rewriting last_summary.
    # Complete, disjoint outputs are authoritative; historical errors stay logged.
    generation_counts = verify_generation(generated)
    source = generated / 'train.jsonl'
    deps = load_dependencies()
    records = list(deps.train.read_records(source))
    packets, cfgs, cache_files = {}, {}, {}
    with tempfile.TemporaryDirectory(prefix='dflash-b8b16-') as temp:
        for block in (8, 16):
            path = Path(temp) / f'b{block}.yaml'
            path.write_text(yaml.safe_dump(config('full', block)))
            directory = prepare(SimpleNamespace(config=path, train_data=source,
                target_model=args.target_model, max_samples=0, workers=args.workers, chunk_size=64,
                reuse_validation_from=None, cache_root=output.parent / 'candidate_validation',
                search_root=[], force=False))
            cfgs[block] = read(directory / 'effective_config.json')
            packets[block] = deps.train.reuse_validation(records, cfgs[block], directory)
            for name in ('effective_config.json', 'skipped_training.jsonl'):
                cache_files[str(directory / name)] = sha(directory / name)
    for path in (source, generated / 'val.jsonl', generated / 'skipped.jsonl', generated / 'manifest/manifest_config.json',
                 generated / 'manifest/prompts.jsonl', generated / 'generation_config.json',
                 generated / 'last_summary.json'):
        if path.exists(): cache_files[str(path)] = sha(path)
    if sha(generated / 'manifest/prompts.jsonl') != generation['manifest_sha256']:
        raise ValueError('Generation prompt manifest changed')
    return publish_common(output, source, records, packets, cfgs, dict(
        target_model=str(args.target_model.expanduser().resolve()), target=fingerprint(args.target_model),
        candidates=candidate_count, generation_counts=generation_counts,
        generation_settings=generation, generation_summary=summary,
        source_files=cache_files), deps.train.reuse_validation)


def prepare_run(args):
    root = args.run_group.expanduser().resolve()
    saved = verify_common(args.common, args.target_model)
    if root.exists():
        raise FileExistsError(f'Choose a new training run group: {root}')
    root.mkdir(parents=True)
    hashes = {}
    for label, (mode, block) in VARIANTS.items():
        directory = root / label
        directory.mkdir()
        cfg = config(mode, block)
        cfg.update(target_model=saved['target_model'], train_data=saved['train_data'], output_dir=str(directory))
        path = directory / 'config.yaml'
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        hashes[label] = sha(path)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE.parent, text=True).strip()
    common = args.common.expanduser().resolve()
    write(root / 'experiment.json', dict(schema='dflash_30w_b8_b16_v1', variants=VARIANTS,
        common=str(common), common_manifest_sha256=sha(common / 'manifest.json'),
        config_sha256=hashes, target=saved['target'], recipe=RECIPE, world_size=4,
        global_batch=256, initialization='from_scratch', git_commit=revision,
        data_sha256=sha(saved['train_data']), eligible_samples=saved['eligible_samples']))
    (root / 'launch.env').write_text(''.join('export ' + k + '=' + shlex.quote(v) + '\n' for k, v in
        dict(TRAIN_DATA=saved['train_data'], TARGET_MODEL=saved['target_model'], COMMON_DATA_DIR=str(common)).items()))
    print(f'Run: {root}; eligible={saved["eligible_samples"]}; 4x2x32=256; 5 epochs', flush=True)


def finalize(root):
    from evaluate_dflash_all import fingerprint
    root = Path(root).expanduser().resolve(strict=True)
    saved = read(root / 'experiment.json')
    if saved.get('schema') != 'dflash_30w_b8_b16_v1':
        raise ValueError('Wrong experiment schema')
    common = Path(saved['common'])
    if sha(common / 'manifest.json') != saved['common_manifest_sha256']:
        raise ValueError('Common-data manifest changed')
    data = verify_common(common)
    checkpoints = {}
    for label, (mode, block) in VARIANTS.items():
        directory = root / label
        if sha(directory / 'config.yaml') != saved['config_sha256'][label]:
            raise ValueError(f'Config changed: {label}')
        effective = read(directory / 'effective_config.json')
        for key, value in dict(RECIPE, block_size=block, draft_context=mode,
                              train_data=data['train_data'], data_sha256=saved['data_sha256'],
                              target_model=data['target_model'], world_size=4,
                              eligible_samples=data['eligible_samples']).items():
            if effective.get(key) != value:
                raise ValueError(f'Training recipe differs: {label}/{key}')
        if effective.get('init_checkpoint') or effective.get('debug_max_samples', 0) or effective.get('debug_max_steps'):
            raise ValueError(f'Warm start/debug training is forbidden: {label}')
        for export in [*[f'epoch_{epoch}' for epoch in range(1, 6)], 'last']:
            item = fingerprint(directory / export)
            cfg = read(Path(item['path']) / 'config.json')
            if (cfg.get('architectures') != ['DFlashDraftModel'] or cfg.get('block_size') != block or
                    cfg.get('vlm_config', {}).get('draft_context', 'full') != mode):
                raise ValueError(f'Wrong checkpoint context/block: {label}/{export}')
        checkpoints[label] = fingerprint(directory / 'epoch_5')
    saved['checkpoints'] = checkpoints
    write(root / 'experiment.json', saved)
    return saved


def check_inputs(args):
    """Fail early on missing eval data, before a multi-hour training run."""
    from prepare_longctx_data import verify_dataset
    from evaluate_dflash_all import PLAN, frozen_selection
    rows, _, _ = verify_dataset(args.longctx_data_dir)
    if len(rows) < 32:
        raise ValueError('Long-context evaluation needs at least 32 prepared records')
    source = os.environ.get('SOURCE_EVAL_DIR')
    if not source and list((args.workspace / 'runs').glob('dflash_exp1_10w_5ep_*/eval_all_b1_n32/manifest.json')):
        source = 'latest-exp1'
    if source:
        _, provenance = frozen_selection(SimpleNamespace(reuse_selection_from=source,
            workspace=args.workspace, samples=32, video_samples=32, max_new_tokens=256,
            seed=42, image_max_tokens=1024, frames=16, max_pixels_per_frame=200704))
        source = str(Path(provenance['manifest']).parent)
    else:
        from run_vlm_matrix import select_data
        from video_benchmark_data import select_data as select_video
        os.environ['PROMPT_FIELD'] = 'reference'
        failures = []
        for name, kind in PLAN:
            try:
                if kind == 'video':
                    select_video(Path(os.environ.get('VIDEO_DATA_ROOT', str(Path.home()/'LMM/liyt/datasets/eval_bench'))), [name], 32, 42)
                else:
                    select_data(SimpleNamespace(tests=['text-only' if name == 'gsm8k' else name], samples=32, seed=42))
            except Exception as exc:
                failures.append(f'{name}: {exc}')
        if failures: raise ValueError('Evaluation inputs unavailable before training:\n'+'\n'.join(failures))
    args.result_file.write_text('export EVAL_REUSE_SELECTION_FROM=' + shlex.quote(source or '') + '\n')
    print(f'Eight datasets and long-context data checked; frozen selection: {source or "raw seed42"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    data = sub.add_parser('data')
    data.add_argument('--generated', type=Path, required=True)
    data.add_argument('--common', type=Path, required=True)
    data.add_argument('--target-model', type=Path, required=True)
    data.add_argument('--workers', type=int, default=8)
    run = sub.add_parser('run')
    run.add_argument('--common', type=Path, required=True)
    run.add_argument('--target-model', type=Path, required=True)
    run.add_argument('--run-group', type=Path, required=True)
    done = sub.add_parser('finalize')
    done.add_argument('--run-group', type=Path, required=True)
    verify = sub.add_parser('verify')
    verify.add_argument('--common', type=Path, required=True)
    verify.add_argument('--target-model', type=Path, required=True)
    inputs = sub.add_parser('inputs')
    inputs.add_argument('--workspace', type=Path, required=True)
    inputs.add_argument('--longctx-data-dir', type=Path, required=True)
    inputs.add_argument('--result-file', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'data': prepare_data(args)
    elif args.command == 'run': prepare_run(args)
    elif args.command == 'finalize': finalize(args.run_group)
    elif args.command == 'inputs': check_inputs(args)
    else: verify_common(args.common, args.target_model)


if __name__ == '__main__':
    main()
