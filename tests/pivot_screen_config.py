"""Explicitly separate shared-SFT and cached-rollout experiment budgets."""
import argparse
import os
from pathlib import Path
import yaml

SFT_MODES = ("A","B","C","E")
ROLLOUT_MODES = ("D0","D1","D2")


def make_config(mode, environ=None):
    e = os.environ if environ is None else environ
    if mode not in (*SFT_MODES,*ROLLOUT_MODES):
        raise ValueError(f"unsupported experiment {mode}")
    root = Path(__file__).resolve().parents[1]
    c = yaml.safe_load((root/'scripts/config/vlm/sft_student_aligned.yaml').read_text())
    rollout = mode in ROLLOUT_MODES
    c.update(pivot_two_rounds=rollout,
        training_protocol='joint_sft_rollout_v3' if rollout else 'shared_sft_v2',
        transition_policy=e.get('TRANSITION_POLICY','trajectory'),
        rollout_loss_weight=float(e.get('ROLLOUT_LOSS_WEIGHT',.25)),
        rollout_backend='cached',final_only=True,context_window_size=1,
        pivot_augmentation=dict(mode=mode,rank=128,history_positions=4),
        num_epochs=int(e.get('EPOCHS',2)),batch_size=int(e.get('MICRO_BATCH_SIZE',4)),
        accumulation_steps=int(e.get('ACCUMULATION_STEPS',2)),
        rollout_anchors=int(e.get('ROLLOUT_ANCHORS',4)),
        num_anchors=int(e.get('NUM_ANCHORS',512)),
        sw_train_chunk_blocks=int(e.get('SFT_CHUNK_BLOCKS',16)),block_size=8,
        num_draft_layers=5,chs_num_layers=12,sequential_rank=320,
        target_layer_ids=[0,1,4,7,11,14,17,20,24,27,30,31],
        image_max_tokens=int(e.get('IMAGE_MAX_TOKENS',1024)),max_length=4096,
        report_to=e.get('REPORT_TO','tensorboard'),save_interval=10**12,log_interval=1,
        final_save=True,seed=42)
    for key in ('num_epochs','batch_size','accumulation_steps','num_anchors','rollout_anchors','sw_train_chunk_blocks'):
        if c[key] < 1:
            raise ValueError(f'{key} must be positive')
    import math
    if c['transition_policy'] not in ('trajectory','verifier'):
        raise ValueError('TRANSITION_POLICY must be trajectory or verifier')
    if not math.isfinite(c['rollout_loss_weight']) or c['rollout_loss_weight'] <= 0:
        raise ValueError('ROLLOUT_LOSS_WEIGHT must be positive and finite')
    return c


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',required=True)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    c=make_config(args.mode)
    args.output.write_text(yaml.safe_dump(c,sort_keys=False))
    print(f"{args.mode}: {c['training_protocol']}; anchors={c['num_anchors']}; final weights only",flush=True)
