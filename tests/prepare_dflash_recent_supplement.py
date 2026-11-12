"""Clone the completed experiment-1 recipe; never modify its old exports."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace
import yaml

from evaluate_dflash_all import fingerprint,frozen_selection,write_json

OLD=('full','recent128','pivot1','pivot5','pivot12')
NEW=('recent256','recent64','recent0')
ALL=('full','recent256','recent128','recent64','pivot1','pivot5','pivot12','recent0')
MATCH_KEYS=('algorithm','num_epochs','world_size','batch_size','accumulation_steps','block_size',
    'num_draft_layers','num_anchors','image_max_tokens','max_length','seed','sampler_seed',
    'learning_rate','weight_decay','warmup_ratio','max_grad_norm','use_muon','muon_lr',
    'muon_momentum','muon_weight_decay','muon_adamw_lr','muon_adamw_weight_decay',
    'loss_decay_gamma','objective_chunk_blocks','train_data','target_model','data_sha256',
    'eligible_samples','eligible_indices_sha256','samples_per_rank','total_optimizer_steps','debug_max_samples','debug_max_steps')
REUSE_KEYS=('data_sha256','train_data','target_model','max_length','block_size','debug_max_samples',
            'image_max_tokens','pivot_two_rounds')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda:handle.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def training_layout(base,world_size):
    if world_size not in (4,8):raise ValueError('Supplement requires four or eight training ranks')
    micro=base['batch_size'];accumulation=256//(world_size*micro)
    per_rank=(base['eligible_samples']+world_size-1)//world_size
    batches=per_rank//micro
    if batches<1:raise ValueError('Not enough eligible samples for one full batch per rank')
    return dict(world_size=world_size,batch_size=micro,accumulation_steps=accumulation,
        samples_per_rank=per_rank,total_optimizer_steps=(batches*base['num_epochs']+accumulation-1)//accumulation)


def verify_baseline(manifest):
    if digest(Path(manifest['source_evaluation'])/'manifest.json')!=manifest['source_evaluation_manifest_sha256']:
        raise ValueError('Frozen baseline evaluation manifest changed')
    for name,item in manifest['baseline_checkpoints'].items():
        if fingerprint(item['path'])!=item:raise ValueError(f'Old {name} checkpoint changed; refusing mixed comparison')
    for path,sha in manifest['baseline_effective_config_sha256'].items():
        if digest(path)!=sha:raise ValueError(f'Old training config changed: {path}')
    if fingerprint(manifest['target']['path'])!=manifest['target']:raise ValueError('Target changed')


def prepare(args):
    baseline=args.baseline_run_group.expanduser().resolve(strict=True)
    root=args.output_dir.expanduser().resolve()
    if root.exists() or root==baseline:
        raise FileExistsError('Choose a fresh supplement subdirectory; the original group itself is reserved')
    for name in OLD:
        control=(baseline/name).resolve()
        if root==control or control in root.parents:
            raise FileExistsError(f'Cannot place a supplement inside an old model directory: {name}')
    recorded=read(baseline/'experiment.json')
    if recorded.get('suite')!='exp1_10w_5ep' or recorded.get('global_batch')!=256:
        raise ValueError('Expected the original experiment-1 10w/5-epoch/global-batch-256 run')
    configs={name:read(baseline/name/'effective_config.json') for name in OLD}
    base=configs['full']
    reference=Path(__file__).resolve().parents[1]/'scripts/config/dflash.yaml'
    if base.get('upstream_config_sha256') and digest(reference)!=base['upstream_config_sha256']:
        raise ValueError('Upstream DFlash recipe changed since the original experiment')
    required=dict(algorithm='dflash',draft_context='full',world_size=8,batch_size=4,
        accumulation_steps=8,num_epochs=5,block_size=8,num_draft_layers=5,num_anchors=512,image_max_tokens=1024)
    for key,value in required.items():
        if base.get(key)!=value:raise ValueError(f'Original Full recipe differs: {key}={base.get(key)!r}; expected {value!r}')
    if base.get('debug_max_samples',0)!=0 or base.get('debug_max_steps') is not None or base.get('init_checkpoint'):
        raise ValueError('Original run must be full-data training from scratch')
    for name,cfg in configs.items():
        if cfg.get('draft_context','full')!=name:raise ValueError(f'Wrong old context: {name}')
        for key in MATCH_KEYS:
            if cfg.get(key)!=base.get(key):raise ValueError(f'Old recipe mismatch: {name}/{key}')
    layout=training_layout(base,getattr(args,'world_size',8))
    data=Path(base['train_data']).expanduser().resolve(strict=True)
    if digest(data)!=base['data_sha256']:raise ValueError('Training JSONL differs from original Full run')
    target=fingerprint(base['target_model'])
    reuse=args.reuse_validation_from.expanduser().resolve(strict=True)
    previous=read(reuse/'effective_config.json')
    if not (reuse/'skipped_training.jsonl').is_file():raise FileNotFoundError('Preflight has no skipped_training.jsonl')
    for key in REUSE_KEYS:
        default=False if key=='pivot_two_rounds' else None
        if base.get(key,default)!=previous.get(key,default):raise ValueError(f'Preflight mismatch: {key}')
    for key in ('eligible_samples','eligible_indices_sha256'):
        if base.get(key)!=previous.get(key):raise ValueError(f'Preflight eligible-set mismatch: {key}')
    source=(args.source_eval_dir or baseline/'eval_all_b1_n32').expanduser().resolve(strict=True)
    evaluation=read(source/'manifest.json')
    if Path(evaluation['run_group']).resolve()!=baseline or evaluation['target']!=target:
        raise ValueError('Frozen evaluation must come from this baseline group and Target')
    old={name:fingerprint(baseline/name/'epoch_5') for name in OLD}
    for name,item in old.items():
        config=read(Path(item['path'])/'config.json')
        if config.get('architectures')!=['DFlashDraftModel'] or config.get('vlm_config',{}).get('draft_context','full')!=name:
            raise ValueError(f'Wrong baseline epoch-5 checkpoint: {name}')
        if evaluation['checkpoints'].get(name)!=item:raise ValueError(f'Frozen evaluation does not use {name}/epoch_5')
    settings=evaluation['settings']
    for key,value in dict(samples=32,video_samples=32,batch_size=1,max_new_tokens=256,seed=42,
                          image_max_tokens=1024,frames=16,max_pixels_per_frame=200704).items():
        if settings.get(key)!=value:raise ValueError(f'Frozen evaluation setting differs: {key}')
    frozen_selection(SimpleNamespace(**dict(settings,reuse_selection_from=str(source),workspace=baseline.parent.parent)))
    root.mkdir(parents=True)
    generated={}
    for name in NEW:
        cfg=copy.deepcopy(base)
        cfg.update(layout)
        for key in ('init_checkpoint','resume'):cfg.pop(key,None)
        cfg.update(upstream_config='scripts/config/dflash.yaml',draft_context=name,context_experiment_variant=name,context_projection='shared',
            position_layer_bias=False,output_dir=str(root/name),save_every_epoch=True,final_only=False,
            final_save=True,save_interval=10**12,log_interval=1,debug_max_samples=0,debug_max_steps=None)
        directory=root/name;directory.mkdir()
        path=directory/'config.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        generated[name]=digest(path)
    manifest=dict(schema='dflash_recent_supplement_v1',baseline_run_group=str(baseline),
        source_evaluation=str(source),target=target,baseline_checkpoints=old,
        source_evaluation_manifest_sha256=digest(source/'manifest.json'),
        baseline_effective_config_sha256={str(baseline/name/'effective_config.json'):digest(baseline/name/'effective_config.json') for name in OLD},
        source_experiment_sha256=digest(baseline/'experiment.json'),
        train_data=str(data),data_sha256=base['data_sha256'],eligible_samples=base['eligible_samples'],
        eligible_indices_sha256=base['eligible_indices_sha256'],reuse_validation_from=str(reuse),
        new_modes=list(NEW),comparison_modes=list(ALL),initialization='from_scratch',global_batch=256,
        training_layout=layout,baseline_world_size=base['world_size'],
        config_sha256=generated,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    write_json(root/'experiment.json',manifest)
    env=dict(BASELINE_RUN_GROUP=str(baseline),BASELINE_EVAL_DIR=str(source),TARGET_MODEL=target['path'],
        TRAIN_DATA=str(data),REUSE_VALIDATION_FROM=str(reuse),ALGORITHM='dflash',NPROC_PER_NODE=str(layout['world_size']),
        EPOCHS='5',MICRO_BATCH_SIZE=str(layout['batch_size']),ACCUMULATION_STEPS=str(layout['accumulation_steps']),IMAGE_MAX_TOKENS='1024',MAX_SAMPLES='0',
        SAVE_EVERY_EPOCH='1',SAVE_TRAINING_STATE='0',REPORT_TO=str(base.get('report_to','tensorboard')))
    (root/'launch.env').write_text(''.join(f'export {key}={shlex.quote(value)}\n' for key,value in env.items()))
    print(f'Prepared three fresh runs from {baseline}\nOutput: {root}\nEligible records: {base["eligible_samples"]}',flush=True)
    return manifest


def finalize(root):
    root=Path(root).expanduser().resolve(strict=True)
    manifest=read(root/'experiment.json')
    if manifest.get('schema')!='dflash_recent_supplement_v1':raise ValueError('Not a recent supplement')
    verify_baseline(manifest)
    new={}
    for name in NEW:
        if digest(root/name/'config.yaml')!=manifest['config_sha256'][name]:raise ValueError(f'Training config changed: {name}')
        cfg=read(root/name/'effective_config.json')
        original=read(Path(manifest['baseline_run_group'])/'full/effective_config.json')
        expected=dict(original,**manifest.get('training_layout',{}))
        for key in MATCH_KEYS:
            if cfg.get(key)!=expected.get(key):raise ValueError(f'Completed training recipe mismatch: {name}/{key}')
        if cfg.get('draft_context')!=name or cfg.get('init_checkpoint'):raise ValueError(f'Wrong trained variant: {name}')
        item=fingerprint(root/name/'epoch_5')
        checkpoint=read(Path(item['path'])/'config.json')
        if checkpoint.get('architectures')!=['DFlashDraftModel'] or checkpoint.get('vlm_config',{}).get('draft_context')!=name:
            raise ValueError(f'Wrong new epoch-5 export: {name}')
        new[name]=item
    manifest['checkpoints']=new
    manifest['all_checkpoints']={name:(new if name in NEW else manifest['baseline_checkpoints'])[name] for name in ALL}
    write_json(root/'experiment.json',manifest)
    print(f'All eight epoch-5 checkpoints verified: {root}',flush=True)
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--baseline-run-group',type=Path,required=True)
    prep.add_argument('--output-dir',type=Path,required=True);prep.add_argument('--reuse-validation-from',type=Path,required=True)
    prep.add_argument('--source-eval-dir',type=Path)
    prep.add_argument('--world-size',type=int,choices=(4,8),default=8)
    done=sub.add_parser('finalize');done.add_argument('--run-group',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=='prepare':prepare(args)
    else:finalize(args.run_group)


if __name__=='__main__':main()
