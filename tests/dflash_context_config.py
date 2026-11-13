"""Generate a matched DFlash control; environment overrides are explicit."""
import os
from pathlib import Path
import sys
import yaml
from litespec.vlm.dflash_context import CONTEXTS


def build(mode):
    variant = mode
    mode = "pivot5" if variant in ("pivot5_B", "pivot5_C") else variant
    if mode not in CONTEXTS:
        raise ValueError(mode)
    epoch_save = os.getenv('SAVE_EVERY_EPOCH', '0')
    if epoch_save not in ('0', '1'):
        raise ValueError('SAVE_EVERY_EPOCH must be 0 or 1')
    epoch_save = epoch_save == '1'
    config = yaml.safe_load(Path('scripts/config/vlm/sft_dflash.yaml').read_text())
    config.update(dflash_context_experiment=True, draft_context=mode, num_epochs=int(os.getenv('EPOCHS','1')),
        batch_size=int(os.getenv('MICRO_BATCH_SIZE','4')),
        accumulation_steps=int(os.getenv('ACCUMULATION_STEPS','2')), block_size=int(os.getenv('BLOCK_SIZE','8')),
        num_anchors=512, save_every_epoch=epoch_save, final_only=not epoch_save, final_save=True, save_interval=10**12,
        image_max_tokens=int(os.getenv('IMAGE_MAX_TOKENS','1024')), log_interval=1)
    if config['block_size'] < 2:
        raise ValueError('BLOCK_SIZE must be >= 2')
    config.update(context_projection="per_layer" if variant in ("pivot5_B", "pivot5_C") else "shared",
                  position_layer_bias=variant == "pivot5_C", context_experiment_variant=variant,
                  loss_decay_gamma=4.0)
    if os.getenv('INIT_CHECKPOINT'):
        config['init_checkpoint'] = str(Path(os.environ['INIT_CHECKPOINT']).expanduser().resolve(strict=True))
    return config


if __name__ == '__main__':
    Path(sys.argv[2]).write_text(yaml.safe_dump(build(sys.argv[1]), sort_keys=False))

