"""Freeze <=100 evaluation questions per suite; optionally encode Qwen3.5 inputs.

No Target weights, answers, generated histories, or training are used as inputs.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import shutil
from types import SimpleNamespace
import uuid

from prepare_longctx_data import prepare as prepare_mathvista, sha256, jsonl
from video_benchmark_data import select_rows

SETS = ('mathvista-govreport', 'Video-MME', 'VideoDetailCaption', 'MMLongBench-Doc')
DOC_REPO = 'yubo2333/MMLongBench-Doc'


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def lines(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')


def capped_rows(rows, limit, seed):
    """Spread document questions across PDFs before taking second questions."""
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row['doc_id'])].append((index, row))
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    for values in groups.values(): rng.shuffle(values)
    selected = []
    while len(selected) < limit and any(groups.values()):
        for key in keys:
            if groups[key]: selected.append(groups[key].pop())
            if len(selected) == limit: break
    return selected


def verify(root, settings):
    m = json.loads((root/'manifest.json').read_text())
    if m['settings'] != settings: raise ValueError(f'{root}: settings changed; choose a new output directory')
    for name, digest in m['files'].items():
        if sha256(root/name) != digest: raise ValueError(f'{root/name}: contents changed')
    for external in m.get('external_files', []):
        p = Path(external['path']); stat = p.stat()
        if stat.st_size != external['bytes'] or stat.st_mtime_ns != external['mtime_ns']:
            raise ValueError(f'External media changed: {p}')
    return m


def atomic_set(root, settings, writer):
    if root.exists():
        m = verify(root, settings)
        print(f'Reusing verified {root.name}: {m["samples"]} samples', flush=True)
        return m
    root.parent.mkdir(parents=True, exist_ok=True)
    stage = root.parent/('.prepare-'+uuid.uuid4().hex)
    stage.mkdir()
    try:
        records, extra = writer(stage)
        if not records: raise ValueError(f'{root.name}: no evaluation samples')
        if len(records) > 100 or len({r['id'] for r in records}) != len(records):
            raise ValueError('Expected <=100 unique sample IDs')
        lines(stage/'base.jsonl', records)
        dump(stage/'inspection_first5.json', records[:5])
        files = {str(p.relative_to(stage)): sha256(p) for p in sorted(stage.rglob('*')) if p.is_file()}
        manifest = dict(schema='multimodal_longctx_v1', settings=settings, samples=len(records),
                        sample_ids=[r['id'] for r in records], files=files, **extra)
        dump(stage/'manifest.json', manifest)
        stage.rename(root)
        return manifest
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def prepare_video(args, name):
    import pandas as pd
    folder = args.video_root/name
    annotation = folder/('videomme/test-00000-of-00001.parquet' if name == 'Video-MME'
                         else 'data/test-00000-of-00001.parquet')
    raw = pd.read_parquet(annotation).to_dict('records')
    key = 'videoID' if name == 'Video-MME' else 'video_name'
    count = min(args.samples, len({str(r[key]) for r in raw}))
    settings = dict(seed=args.seed, limit=args.samples, annotation=str(annotation.resolve()),
                    annotation_sha256=sha256(annotation), frames=args.frames,
                    max_pixels_per_frame=args.video_max_pixels, protocol='description')
    def writer(stage):
        selected = select_rows(name, raw, folder, count, args.seed)
        for frames in args.frames:
            lines(stage/f'frames_{frames}.jsonl', [dict(r, num_frames=frames,
                max_pixels_per_frame=args.video_max_pixels) for r in selected])
        external = [dict(path=r['video'], bytes=r['video_bytes'], mtime_ns=r['video_mtime_ns']) for r in selected]
        return selected, dict(external_files=external,
            prompt_field='question' if name == 'VideoDetailCaption' else 'fixed CAPTION_PROMPT',
            turn_mode='first', protocol_note='Video-MME description speed benchmark, not official QA accuracy',
            frame_sampling='video_runtime.prepare_video: uniform indices independently for each frame count')
    return atomic_set(args.output_dir/name, settings, writer)


def document_records(selected, root, stage, pdf_lookup, page_edge):
    import fitz
    rendered, sources, records = {}, {}, []
    for index, row in selected:
        question = row.get('question')
        if not isinstance(question, str) or not question.strip(): raise ValueError(f'Row {index}: empty question')
        doc_id = row['doc_id']
        if not isinstance(doc_id, str) or Path(doc_id).name != doc_id: raise ValueError(f'Unsafe doc_id: {doc_id}')
        if doc_id not in rendered:
            pdf = Path(pdf_lookup(doc_id)).resolve(strict=True)
            import hashlib
            key = hashlib.sha256(doc_id.encode()).hexdigest()[:20]
            pages = []
            with fitz.open(pdf) as doc:
                if not len(doc): raise ValueError(f'Empty PDF: {pdf}')
                for page_index, page in enumerate(doc):
                    rel = Path('pages')/key/f'{page_index+1:04d}.jpg'
                    (stage/rel).parent.mkdir(parents=True, exist_ok=True)
                    scale = page_edge/max(page.rect.width, page.rect.height)
                    page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).save(str(stage/rel))
                    pages.append(str(root/rel))
            rendered[doc_id] = pages
            sources[doc_id] = dict(path=str(pdf), sha256=sha256(pdf), pages=len(pages))
        records.append(dict(id=f'mmlongbench-doc:{index}', source_row=index, doc_id=doc_id,
            question=question, images=rendered[doc_id], page_numbers=list(range(1,len(rendered[doc_id])+1)),
            answer=row.get('answer'), evidence_pages=row.get('evidence_pages'),
            doc_type=row.get('doc_type'), source=DOC_REPO))
        print(f'Document {len(records)}/{len(selected)}: {doc_id}, {len(rendered[doc_id])} pages', flush=True)
    return records, dict(documents=sources, prompt_field='question', turn_mode='first',
        page_policy='all pages in original order; no evidence-based selection',
        page_longest_edge=page_edge, external_files=[])


def prepare_documents(args):
    import pandas as pd
    revision = args.doc_revision
    def fetch(filename):
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(DOC_REPO, filename, repo_type='dataset', revision=revision,
                                    cache_dir=str(args.cache_dir)))
    if args.doc_parquet is None or args.doc_root is None:
        from huggingface_hub import HfApi
        revision = HfApi().dataset_info(DOC_REPO, revision=revision).sha
    annotation = args.doc_parquet or fetch('data/train-00000-of-00001.parquet')
    raw = pd.read_parquet(annotation).to_dict('records')
    # Inspect source schema before converting: source question, exactly one user turn.
    print(json.dumps([dict(source_row=i, **r) for i,r in enumerate(raw[:5])], ensure_ascii=False, default=str, indent=2), flush=True)
    settings = dict(seed=args.seed, limit=args.samples, annotation_sha256=sha256(annotation),
                    prompt_field='question', turn_mode='first', page_edge=args.doc_page_edge,
                    revision=revision, doc_root=str(args.doc_root.resolve()) if args.doc_root else None)
    root = args.output_dir/'MMLongBench-Doc'
    def lookup(doc_id):
        return args.doc_root/doc_id if args.doc_root else fetch('documents/'+doc_id)
    def writer(stage):
        records, extra = document_records(capped_rows(raw,args.samples,args.seed), root,stage,lookup,args.doc_page_edge)
        # Local PDFs must remain unchanged on reuse too.
        extra['external_files'] = [dict(path=v['path'], bytes=Path(v['path']).stat().st_size,
            mtime_ns=Path(v['path']).stat().st_mtime_ns) for v in extra['documents'].values()]
        return records, extra
    return atomic_set(root, settings, writer)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--video-root', type=Path, default=Path.home()/'LMM/liyt/datasets/eval_bench')
    p.add_argument('--tests', nargs='+', choices=SETS, default=list(SETS))
    p.add_argument('--samples', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--frames', type=int, nargs='+', default=[32,64,128])
    p.add_argument('--video-max-pixels', type=int, default=448*448)
    p.add_argument('--doc-page-edge', type=int, default=896)
    p.add_argument('--doc-parquet', type=Path)
    p.add_argument('--doc-root', type=Path, help='Local documents/ directory, containing PDFs')
    p.add_argument('--doc-revision', default='main')
    p.add_argument('--mathvista-parquet', type=Path)
    p.add_argument('--govreport-parquet', type=Path)
    a = p.parse_args(argv)
    if not 1 <= a.samples <= 100: p.error('samples must be 1..100')
    if any(f < 2 or f % 2 for f in a.frames) or len(set(a.frames)) != len(a.frames): p.error('frames must be unique positive even counts')
    if a.doc_page_edge < 32 or a.video_max_pixels < 1024: p.error('invalid image size')
    for key in ('output_dir','cache_dir','video_root','doc_root','doc_parquet','mathvista_parquet','govreport_parquet'):
        value=getattr(a,key)
        if value is not None: setattr(a,key,value.expanduser().resolve())
    a.output_dir.mkdir(parents=True,exist_ok=True)
    summary = {}
    for name in a.tests:
        if name == 'mathvista-govreport':
            dest = a.output_dir/name
            prepare_mathvista(SimpleNamespace(output_dir=dest, cache_dir=a.cache_dir, samples=a.samples,
                mathvista_parquet=a.mathvista_parquet, govreport_parquet=a.govreport_parquet))
            summary[name]=dict(path=str(dest), samples=a.samples, lengths=[0,8192,32768,65536],
                note='Base + filler pool. Exact length variants require encode_multimodal_longctx.py.')
        else:
            m=prepare_documents(a) if name == 'MMLongBench-Doc' else prepare_video(a,name)
            summary[name]=dict(path=str(a.output_dir/name), samples=m['samples'])
    # Merge independently prepared subsets rather than losing previous entries.
    path=a.output_dir/'suite.json'
    previous=json.loads(path.read_text()) if path.exists() else {}
    previous.update(summary);dump(path,previous)
    print(json.dumps(previous,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__': main()
