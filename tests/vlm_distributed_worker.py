"""CPU torchrun integration fixture using a real tiny Qwen3.5 vision target."""
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from test_vlm import tiny_image_inputs, tiny_qwen35
from litespec.vlm import train


class Processor:
    tokenizer = type("Tokenizer", (), {"get_vocab": lambda self: {str(i): i for i in range(121)}})()

    def apply_chat_template(self, messages, **kwargs):
        inputs = tiny_image_inputs()
        if not kwargs["add_generation_prompt"]:
            inputs["input_ids"] = torch.cat([inputs["input_ids"], torch.tensor([[4, 5, 6, 7, 120]])], dim=1)
            inputs["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"], torch.zeros(1, 5, dtype=torch.long)], dim=1)
        return inputs


if __name__ == "__main__":
    torch.set_num_threads(1)
    mode, config, output = sys.argv[1:]
    train.load_target = lambda *a, **kw: (tiny_qwen35().requires_grad_(False), Processor())
    original = train.train_block
    calls = 0

    def objective(*args, **kwargs):
        global calls
        calls += 1
        if mode == "interrupt" and calls == 2:
            raise RuntimeError("simulated distributed interruption")
        return original(*args, **kwargs)

    train.train_block = objective
    adamw = torch.optim.AdamW

    class CheckedAdamW(adamw):
        def step(self, *args, **kwargs):
            result = super().step(*args, **kwargs)
            values = torch.cat([p.detach().reshape(-1) for group in self.param_groups for p in group["params"]])
            copies = [torch.empty_like(values) for _ in range(dist.get_world_size())]
            dist.all_gather(copies, values)
            for copy in copies:
                torch.testing.assert_close(copy, values, rtol=0, atol=0)
            return result

    torch.optim.AdamW = CheckedAdamW
    # Capture each rank's synchronized parameters, beyond rank-zero checkpointing.
    save = train.save_checkpoint

    def save_and_check(path, draft, optimizer, state, cfg, rng_states=None):
        assert len(rng_states) == int(os.environ["WORLD_SIZE"])
        return save(path, draft, optimizer, state, cfg, rng_states)

    train.save_checkpoint = save_and_check
    sys.argv = ["train", "--config", config]
    if mode == "resume":
        sys.argv += ["--resume", str(Path(output) / "last")]
    try:
        train.main()
    except RuntimeError as exc:
        if mode != "interrupt" or str(exc) != "simulated distributed interruption":
            raise
