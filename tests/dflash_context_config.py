"""Generate a matched DFlash control; environment overrides are explicit."""
import os
from pathlib import Path
import sys
import yaml
from litespec.vlm.dflash_context import CONTEXTS


def build(mode):
    if mode not in CONTEXTS:
        raise ValueError(mode)
    config = yaml.safe_load(Path('scripts/config/vlm/sft_dflash.yaml').read_text())
    config.update(dflash_context_experiment=True, draft_context=mode, num_epochs=int(os.getenv('EPOCHS','1')),
        batch_size=int(os.getenv('MICRO_BATCH_SIZE','4')),
        accumulation_steps=int(os.getenv('ACCUMULATION_STEPS','2')), block_size=8,
        num_anchors=512, final_only=True, final_save=True, save_interval=10**12,
        image_max_tokens=int(os.getenv('IMAGE_MAX_TOKENS','1024')), log_interval=1)
    if os.getenv('INIT_CHECKPOINT'):
        config['init_checkpoint'] = str(Path(os.environ['INIT_CHECKPOINT']).expanduser().resolve(strict=True))
    return config


if __name__ == '__main__':
    Path(sys.argv[2]).write_text(yaml.safe_dump(build(sys.argv[1]), sort_keys=False))
