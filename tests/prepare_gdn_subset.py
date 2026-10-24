"""Collect exactly N usable image/answer records from the EXISTING train manifest.

No prompt conversion, answer regeneration, target weights or CUDA. Validate with
native Qwen processing and the aligned trainer's own supervised-anchor rule.
"""
import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litespec.vlm.serving_data import file_digest, output_lock, read_jsonl

_PROCESSOR = _TARGET_CONFIG = _OPTIONS = None


def init_validator(target_model, max_length, block_size, image_max_tokens):
    global _PROCESSOR, _TARGET_CONFIG, _OPTIONS
    import torch
    from transformers import AutoConfig, AutoProcessor
    from litespec.vlm.runtime import configure_image_token_budget
    torch.set_num_threads(1)
    _TARGET_CONFIG = AutoConfig.from_pretrained(target_model, local_files_only=True)
    _PROCESSOR = AutoProcessor.from_pretrained(target_model, local_files_only=True)
    configure_image_token_budget(_PROCESSOR, _TARGET_CONFIG, image_max_tokens)
    _OPTIONS = dict(max_length=max_length, block_size=block_size)


def validate_record(record):
    from PIL.Image import DecompressionBombError
    from litespec.vlm.runtime import prepare_example
    from litespec.vlm.eligibility import has_anchor, loss_mask
    try:
        if not isinstance(record.get('image'), str) or not record['image'].strip():
            return dict(reason='missing_single_image')
        if record.get('split', 'train') != 'train':
            return dict(reason='non_train_split')
        example = prepare_example(_PROCESSOR, _TARGET_CONFIG, record, device='cpu', training=True,
                                  max_length=_OPTIONS['max_length'])
        if not has_anchor(example, _OPTIONS['block_size']):
            return dict(reason='insufficient_supervised_answer_or_full_block_anchor')
        if 'pixel_values' not in example.inputs or 'image_grid_thw' not in example.inputs:
            return dict(reason='missing_encoded_image')
        image_tokens = int((example.inputs['input_ids'] == _TARGET_CONFIG.image_token_id).sum())
        if image_tokens < 1:
            return dict(reason='missing_encoded_image_tokens')
        return dict(reason=None, input_tokens=example.inputs['input_ids'].shape[1],
                    answer_tokens=int(loss_mask(example).sum()), image_tokens=image_tokens)
    except (ValueError, OSError, EOFError, DecompressionBombError) as exc:
        return dict(reason=type(exc).__name__, detail=str(exc)[:1000])


def ordered_checks(records, indices, *, workers, init_args, validator=None):
    """Bounded prefetch, consumed in seeded order regardless of worker timing."""
    if workers == 1:
        if validator is None:
            init_validator(*init_args)
        for i in indices:
            yield i, (validator or validate_record)(records[i])
        return
    if validator is not None:
        raise ValueError('injected test validators require workers=1')
    pending, iterator = deque(), iter(indices)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=init_validator, initargs=init_args) as pool:
        try:
            for _ in range(min(len(indices), 2*workers)):
                index = next(iterator)
                pending.append((index, pool.submit(validate_record, records[index])))
            while pending:
                index, future = pending.popleft()
                yield index, future.result()
                try:
                    next_index = next(iterator)
                except StopIteration:
                    continue
                pending.append((next_index, pool.submit(validate_record, records[next_index])))
        finally:
            for _, future in pending:
                future.cancel()


def collect(source, directory, target_model, *, count=20000, seed=42, max_length=4096,
            block_size=8, image_max_tokens=1024, workers=4, validator=None):
    if min(count, max_length, block_size, image_max_tokens, workers) < 1 or block_size < 2:
        raise ValueError('positive counts/limits/workers and block_size >= 2 required')
    source, directory, target_model = (Path(p).expanduser().resolve() for p in (source, directory, target_model))
    if directory == source.parent:
        raise ValueError('subset output must be separate from the source manifest directory')
    import transformers
    root = Path(__file__).resolve().parents[1]
    identity = dict(schema='gdn_usable_subset_v1', source=str(source), source_sha256=file_digest(source),
        target_model=str(target_model), count=count, seed=seed, max_length=max_length,
        block_size=block_size, image_max_tokens=image_max_tokens, transformers=transformers.__version__,
        selection='seeded_permutation_first_N_valid_unique_image_question_and_id',
        validation_code_sha256={p:file_digest(root/p) for p in
            ('tests/prepare_gdn_subset.py', 'litespec/vlm/runtime.py', 'litespec/vlm/eligibility.py')},
        target_files_sha256={p.name:file_digest(p) for p in sorted(target_model.iterdir())
                             if p.suffix in {'.json', '.jinja'}})
    with output_lock(directory):
        metadata = directory/'selection.json'
        manifest = directory/'train.jsonl'
        if metadata.exists():
            prior = json.loads(metadata.read_text())
            if prior['identity'] != identity or file_digest(manifest) != prior['train_sha256']:
                raise ValueError('subset source/settings/files changed; choose a fresh SUBSET_DIR')
            # Regenerate the lightweight compatibility packet from verified output.
            write_validation(directory, prior)
            print(f'Reusing {count} validated records: {manifest}', flush=True)
            return prior
        if manifest.exists():
            raise ValueError('untracked train.jsonl exists; choose a fresh SUBSET_DIR')
        records, source_lines = [], []
        for number, row in read_jsonl(source):
            if not isinstance(row, dict):
                raise ValueError(f'source line {number} is not a record object')
            row = dict(row)
            row['id'] = str(row.get('id', number-1))
            image = row.get('image')
            if isinstance(image, str) and image.strip():
                path = Path(image).expanduser()
                row['image'] = str((source.parent/path).resolve() if not path.is_absolute() else path.resolve())
            records.append(row)
            source_lines.append(number)
        indices = list(range(len(records)))
        random.Random(seed).shuffle(indices)
        selected_ids, seen_ids, seen_prompts, skipped = [], set(), set(), Counter()
        scanned = 0
        temporary = directory/'train.jsonl.tmp'
        init_args = (str(target_model), max_length, block_size, image_max_tokens)
        checks = ordered_checks(records, indices, workers=workers, init_args=init_args, validator=validator)
        from tqdm import tqdm
        progress = tqdm(total=count, desc='Collect usable image records', unit='record')
        try:
            with temporary.open('w', encoding='utf-8') as handle, (directory/'rejected.jsonl').open('w', encoding='utf-8') as rejects, (directory/'selected.jsonl').open('w', encoding='utf-8') as audit:
                for index, result in checks:
                    scanned += 1
                    row = records[index]
                    reason = result.get('reason')
                    prompt_key = (row.get('image'), row.get('question'))
                    if reason is None:
                        if row['id'] in seen_ids:
                            reason = 'duplicate_id'
                        elif prompt_key in seen_prompts:
                            reason = 'duplicate_image_question'
                    if reason is not None:
                        skipped[reason] += 1
                        rejects.write(json.dumps(dict(source_line=source_lines[index], id=row['id'],
                            **{**result, 'reason':reason}), ensure_ascii=False)+'\n')
                    else:
                        seen_ids.add(row['id'])
                        seen_prompts.add(prompt_key)
                        selected_ids.append(row['id'])
                        handle.write(json.dumps(row, ensure_ascii=False)+'\n')
                        audit.write(json.dumps(dict(source_line=source_lines[index], id=row['id'],
                            **result), ensure_ascii=False)+'\n')
                        progress.update()
                    progress.set_postfix(scanned=scanned, rejected=sum(skipped.values()), refresh=False)
                    if len(selected_ids) == count:
                        break
        finally:
            checks.close()
            progress.close()
        stats = dict(source_rows=len(records), scanned=scanned, selected=len(selected_ids),
                     rejected=sum(skipped.values()), rejection_reasons=dict(skipped))
        if len(selected_ids) != count:
            temporary.unlink(missing_ok=True)
            (directory/'incomplete.json').write_text(json.dumps(stats, indent=2)+'\n')
            raise ValueError(f'only {len(selected_ids)} usable records found; need exactly {count}; training will not start')
        if file_digest(source) != identity['source_sha256']:
            temporary.unlink(missing_ok=True)
            raise ValueError('source changed while collecting; training will not start')
        os.replace(temporary, manifest)
        result = dict(identity=identity, stats=stats, train_sha256=file_digest(manifest),
                      selected_ids_sha256=hashlib.sha256(json.dumps(selected_ids).encode()).hexdigest())
        write_validation(directory, result)
        metadata.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps(dict(train_data=str(manifest), **stats), ensure_ascii=False), flush=True)
        return result


def write_validation(directory, result):
    """Compatibility packet consumed by the existing strict reuse_validation."""
    identity = result['identity']
    count = identity['count']
    cfg = dict(train_data=str(directory/'train.jsonl'), data_sha256=result['train_sha256'],
        target_model=identity['target_model'], max_length=identity['max_length'],
        block_size=identity['block_size'], image_max_tokens=identity['image_max_tokens'], debug_max_samples=0,
        eligible_samples=count, eligible_indices_sha256=hashlib.sha256(json.dumps(list(range(count))).encode()).hexdigest())
    (directory/'effective_config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    # Source rejections do not belong here: every row of the OUTPUT is valid.
    (directory/'skipped_training.jsonl').write_text('')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--target-model', type=Path, required=True)
    parser.add_argument('--count', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-length', type=int, default=4096)
    parser.add_argument('--block-size', type=int, default=8)
    parser.add_argument('--image-max-tokens', type=int, default=1024)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(argv)
    collect(args.source, args.output_dir, args.target_model, count=args.count, seed=args.seed,
            max_length=args.max_length, block_size=args.block_size,
            image_max_tokens=args.image_max_tokens, workers=args.workers)


if __name__ == '__main__':
    main()
