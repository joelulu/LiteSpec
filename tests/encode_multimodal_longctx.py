"""Materialize exact MathVista lengths and audit video/document input lengths.

Loads only local Processor/config (no model weights). Does not generate answers.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prepare_multimodal_longctx import SETS, atomic_set, dump, lines, verify
from prepare_longctx_data import jsonl, verify_dataset, sha256


def document_example(processor, config, record):
    """Evaluation-only multi-page input; preserve page order and exclude answer."""
    from PIL import Image
    import torch
    from litespec.vlm.context import visual_positions
    from litespec.vlm.runtime import PreparedExample
    if not record.get('images') or not record.get('question', '').strip():
        raise ValueError('Document requires all page images and a question')
    content=[]
    for path in record['images']:
        with Image.open(path) as image:
            content.append(dict(type='image', image=image.convert('RGB')))
    content.append(dict(type='text',text=record['question']))
    inputs=processor.apply_chat_template([dict(role='user',content=content)],tokenize=True,
        add_generation_prompt=True,enable_thinking=False,return_dict=True,return_tensors='pt')
    inputs={k:v.cpu() if torch.is_tensor(v) else v for k,v in inputs.items()}
    if inputs['image_grid_thw'].shape[0] != len(record['images']): raise ValueError('Processor dropped document pages')
    length=inputs['input_ids'].shape[1]
    mask=visual_positions(inputs['input_ids'][0], config)
    return PreparedExample(inputs,length,mask,record['id'])


def length_bin(tokens):
    for bound in (8192,32768,65536):
        if tokens <= bound: return f'le_{bound}'
    return 'gt_65536'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--target-model',type=Path,required=True)
    p.add_argument('--tests',nargs='+',choices=SETS,default=list(SETS))
    p.add_argument('--image-max-tokens',type=int,default=1024,help='Per image/page cap; document has multiple pages')
    p.add_argument('--max-new-tokens',type=int,default=256)
    a=p.parse_args(argv)
    if min(a.image_max_tokens,a.max_new_tokens)<1:p.error('token budgets must be positive')
    a.data_dir=a.data_dir.expanduser().resolve(strict=True)
    target=a.target_model.expanduser().resolve(strict=True)
    from transformers import AutoConfig, AutoProcessor
    from reference_vlm_image_budget import configure_image_budget
    config=AutoConfig.from_pretrained(target,local_files_only=True)
    processor=AutoProcessor.from_pretrained(target,local_files_only=True)
    vision=config.vision_config
    pixels=a.image_max_tokens*(vision.patch_size*vision.spatial_merge_size)**2
    configure_image_budget(processor,max_pixels=pixels)
    limit=config.text_config.max_position_embeddings
    processor_files={str(f.relative_to(target)):sha256(f) for f in target.iterdir()
        if f.is_file() and f.suffix in ('.json','.txt','.jinja','.model')}
    for name in a.tests:
        source=a.data_dir/name
        if name == 'mathvista-govreport':
            verify_dataset(source)
        else:
            source_manifest=json.loads((source/'manifest.json').read_text())
            verify(source,source_manifest['settings'])
        settings=dict(source_manifest_sha256=sha256(source/'manifest.json'),target=str(target),
            processor_files=processor_files,image_max_tokens=a.image_max_tokens,max_new_tokens=a.max_new_tokens)
        destination=a.data_dir/'encoded'/name
        def writer(stage):
            if name=='mathvista-govreport':
                from evaluate_dflash_longctx import build_inputs
                from litespec.vlm.runtime import prepare_example
                rows,reports,_=verify_dataset(source)
                def encode(row):
                    e=prepare_example(processor,config,row,device='cpu')
                    return e.prefill_length,int(e.vision_mask.sum())
                args=SimpleNamespace(samples=len(rows),lengths=[0,8192,32768,65536],seed=42,
                    max_new_tokens=a.max_new_tokens,context_limit=limit)
                build_inputs(args,stage,rows,reports,processor.tokenizer,encode)
                return rows,dict(context_limit=limit,lengths=args.lengths)
            m=json.loads((source/'manifest.json').read_text())
            verify(source,m['settings'])
            records=jsonl(source/'base.jsonl');audits=[]
            variants=[(None,records)] if name=='MMLongBench-Doc' else [
                (f,jsonl(source/f'frames_{f}.jsonl')) for f in m['settings']['frames']]
            for frames,variant in variants:
                for row in variant:
                    if frames is None:
                        e=document_example(processor,config,row)
                        info=dict(pages=len(row['images']))
                    else:
                        from video_runtime import prepare_video
                        e,info=prepare_video(processor,config,row,frames,row['max_pixels_per_frame'])
                    info.update(id=row['id'],frames=frames,input_tokens=e.prefill_length,
                        visual_tokens=int(e.vision_mask.sum()),length_bin=length_bin(e.prefill_length),
                        within_model_limit=e.prefill_length+a.max_new_tokens<=limit)
                    audits.append(info)
                    print(f'{name}: {row["id"]}, frames={frames}, input_tokens={e.prefill_length}',flush=True)
                    del e
            lines(stage/'input_lengths.jsonl',audits)
            return records,dict(context_limit=limit,over_limit=sum(not r['within_model_limit'] for r in audits),
                note='Over-limit records retained and flagged, not truncated. These are processor counts, not GPU peak estimates.')
        atomic_set(destination,settings,writer)


if __name__=='__main__':main()
