"""Download MathVista/GovReport shards and save 100 fixed single-image records."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import uuid

HERE = Path(__file__).resolve().parent
SOURCES = {
    'mathvista': ('AI4Math/MathVista', 'data/testmini-00000-of-00001-725687bf7a18d64b.parquet'),
    'govreport': ('ccdv/govreport-summarization', 'document/test-00000-of-00001.parquet'),
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def verify_dataset(root, samples=None):
    root = Path(root).expanduser().resolve(strict=True)
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest.get('schema') != 'mathvista_govreport_longctx_v1':
        raise ValueError('Not a prepared long-context dataset')
    if samples is not None and manifest['samples'] != samples:
        raise ValueError('Existing data count differs; choose a fresh output directory')
    for name, digest in manifest['files'].items():
        if sha256(root/name) != digest: raise ValueError(f'Prepared data changed: {name}')
    rows, reports = jsonl(root/'base.jsonl'), jsonl(root/'reports.jsonl')
    if len(rows) != manifest['samples'] or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Incorrect or duplicate base sample IDs')
    if not reports or any(not row['text'].strip() for row in reports): raise ValueError('Empty report pool')
    return rows, reports, manifest


def write_dataset(output, math_rows, report_rows, reference, samples, sources):
    """Inspect five raw records first, then apply the agreed query/single-turn mapping."""
    from PIL import Image
    output = Path(output).expanduser().resolve()
    wanted = reference['mathvista_ids'][:samples]
    by_id = {str(row['pid']): row for row in math_rows}
    if len(wanted) != samples or len(set(wanted)) != samples: raise ValueError('Need unique reference IDs')
    inspection = []
    for pid in wanted[:5]:
        row = by_id[pid]
        prompt = row.get('query')
        if not isinstance(prompt, str) or not prompt.strip(): raise ValueError(f'{pid}: missing query string')
        inspection.append(dict(pid=pid, available_keys=sorted(row), prompt_field='query', turn_mode='first',
                               prompt_preview=prompt[:240], image_field='decoded_image.bytes'))
    print('Inspecting first five selected records; query -> question, one user turn:', flush=True)
    print(json.dumps(inspection, ensure_ascii=False, indent=2), flush=True)
    if output.exists(): raise FileExistsError(f'Choose a fresh output directory: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent/('.longctx-'+uuid.uuid4().hex)
    (staging/'images').mkdir(parents=True)
    try:
        records = []
        for pid in wanted:
            row = by_id[pid]
            prompt = row.get('query')
            if not isinstance(prompt,str) or not prompt.strip(): raise ValueError(f'{pid}: missing query string')
            data = row['decoded_image']['bytes']
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            path = staging/'images'/f'{pid}.png'
            path.write_bytes(data)
            records.append(dict(id=f'mathvista:{pid}', source_id=pid, question=prompt,
                image=str(output/'images'/path.name), answer=str(row.get('answer','')), source='AI4Math/MathVista:testmini'))
        reports = []
        for index in reference['govreport_indices']:
            text = report_rows[index]['report']
            if not isinstance(text,str) or not text.strip(): raise ValueError(f'Empty GovReport row {index}')
            reports.append(dict(id=f'govreport:{index}', text=text))
        for name, rows in [('base.jsonl',records), ('reports.jsonl',reports)]:
            (staging/name).write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
        (staging/'inspection_first5.json').write_text(json.dumps(inspection,ensure_ascii=False,indent=2)+'\n')
        files = {str(p.relative_to(staging)):sha256(p) for p in sorted(staging.rglob('*')) if p.is_file()}
        manifest = dict(schema='mathvista_govreport_longctx_v1', samples=samples, report_count=len(reports),
            prompt_field='query', turn_mode='first', sources=sources, reference=reference,
            sample_ids=[row['id'] for row in records], files=files)
        (staging/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging,ignore_errors=True)
        raise
    return output


def prepare(args):
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        verify_dataset(output,args.samples)
        print(f'Reusing verified data: {output}',flush=True)
        return output
    from huggingface_hub import hf_hub_download
    import pandas as pd
    reference = json.loads((HERE/'longctx_reference.json').read_text())
    tables, sources = {}, {}
    for key,(repo,filename) in SOURCES.items():
        print(f'Downloading/caching {repo}/{filename}',flush=True)
        path = Path(hf_hub_download(repo_id=repo,repo_type='dataset',filename=filename,cache_dir=str(args.cache_dir)))
        parts=path.parts
        revision=parts[parts.index('snapshots')+1] if 'snapshots' in parts else 'unknown'
        sources[key]=dict(repo=repo,filename=filename,resolved_revision=revision,sha256=sha256(path))
        columns=['pid','query','answer','decoded_image'] if key=='mathvista' else ['report']
        tables[key]=pd.read_parquet(path,columns=columns).to_dict('records')
    return write_dataset(output,tables['mathvista'],tables['govreport'],reference,args.samples,sources)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--cache-dir',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=100)
    args=parser.parse_args(argv)
    if not 5 <= args.samples <= 100: parser.error('samples must be between 5 and 100')
    out=prepare(args)
    rows,reports,_=verify_dataset(out,args.samples)
    print(f'Prepared {len(rows)} image/question records and {len(reports)} report texts: {out}',flush=True)


if __name__=='__main__': main()
