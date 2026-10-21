"""Text-only control using LiteSpec's original DLite spec_generate entry point."""
import torch

from litespec.modeling.draft.dlite import DLiteDraftModel
from .runtime import clock


@torch.no_grad()
def generate_native_text(target, draft, example, *, max_new_tokens=128, verification="block"):
    inputs = example.inputs
    if any(key in inputs for key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw")):
        raise ValueError("native_text accepts text-only prompts; never pass image inputs through the text entry point")
    ids = inputs["input_ids"]
    if ids.shape != (1, example.prefill_length) or max_new_tokens < 1:
        raise ValueError("one unpadded prompt and a positive output cap required")
    device = ids.device
    eos = getattr(target.config, "text_config", target.config).eos_token_id
    eos = list(eos) if isinstance(eos, (list, tuple)) else [eos]
    eos = [value for value in eos if value is not None]
    start = clock(device)
    if draft is not None:
        if not isinstance(draft, DLiteDraftModel) or not draft.is_student:
            raise ValueError("native_text control currently requires a DLite student")
        output = draft.spec_generate(target, ids, max_new_tokens=max_new_tokens,
                                     stop_token_ids=eos, temperature=0.0,
                                     strict_greedy_verification=verification == "sequential")
        elapsed = clock(device) - start
        tokens = output[0, example.prefill_length:].tolist()
        stats = draft.get_last_decode_stats()
        lengths, remaining = [], len(tokens)
        # Native generation may verify beyond the final output/EOS boundary.
        # Normalize acceptance to the tokens actually returned, as the VLM path does.
        for length in stats["accept_lengths"]:
            if remaining == 0:
                break
            kept = min(int(length), remaining)
            lengths.append(kept)
            remaining -= kept
        decode_seconds = stats["decode_wall_time"]
        prefill_seconds = None  # Not separately instrumented by the native entry.
    else:
        positions = torch.arange(example.prefill_length, device=device)[None]
        output = target(input_ids=ids, position_ids=positions, use_cache=True, logits_to_keep=1)
        cache = output.past_key_values
        pending = output.logits[:, -1:].argmax(-1)
        prefill_seconds = clock(device) - start
        tokens = []
        for offset in range(max_new_tokens):
            token = int(pending.item())
            tokens.append(token)
            if token in eos or len(tokens) == max_new_tokens:
                break
            output = target(input_ids=pending, position_ids=ids.new_tensor([[example.prefill_length + offset]]),
                            past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = output.past_key_values
            pending = output.logits[:, -1:].argmax(-1)
        elapsed = clock(device) - start
        decode_seconds, lengths = elapsed - prefill_seconds, []
    return dict(token_ids=tokens, accept_lengths=lengths, total_seconds=elapsed,
                decode_seconds=decode_seconds, native_prefill_seconds=prefill_seconds,
                prefill_tokens=example.prefill_length,
                context_counts=[], verification=verification, execution="native_text",
                stage_profile_available=False)
