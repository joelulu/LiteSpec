"""Position-based draft context selection. Keep in sync with LiteSpec's copy."""

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class ContextPolicy:
    prefill_text: str = "keep"
    prefill_vision: str = "keep"
    decode_mode: str = "all"
    window_size: int = 0
    implementation: str = "mask"

    def __post_init__(self):
        if self.prefill_text not in {"keep", "drop"} or self.prefill_vision not in {"keep", "drop"}:
            raise ValueError("prefill selectors must be keep or drop")
        if self.decode_mode not in {"all", "none", "keep_last", "drop_recent"}:
            raise ValueError("invalid decode_mode")
        if self.implementation not in {"mask", "compact"}:
            raise ValueError("implementation must be mask or compact")
        if self.decode_mode in {"keep_last", "drop_recent"}:
            if self.window_size <= 0:
                raise ValueError("window modes require a positive window_size")
        elif self.window_size != 0:
            raise ValueError("window_size is only valid for window modes")

    @classmethod
    def no_history(cls):
        return cls("drop", "drop", "none", implementation="compact")

    @property
    def is_empty(self):
        return self.prefill_text == self.prefill_vision == "drop" and self.decode_mode == "none"

    def to_dict(self):
        return asdict(self)

    def keep(self, positions, *, anchor, prefill_length, vision_mask):
        """History excludes pivot h[a-1]; anchor x[a] is a separate query.

        Windows count absolute generated-token positions, not the number of
        entries remaining after another mask. The same function runs in train
        and decode. No position at/after the pivot can ever become history.
        """
        if anchor < 1 or prefill_length < 1:
            raise ValueError("anchor and prefill_length must be positive")
        if vision_mask.ndim != 1 or vision_mask.numel() != prefill_length:
            raise ValueError("vision_mask must cover the entire prefill")
        prefill = positions < prefill_length
        visual = prefill & vision_mask[positions.clamp(0, prefill_length - 1)]
        text = prefill & ~visual
        decode = ~prefill
        if self.decode_mode == "none":
            decode = torch.zeros_like(decode)
        elif self.decode_mode == "keep_last":
            decode = decode & (positions >= anchor - 1 - self.window_size)
        elif self.decode_mode == "drop_recent":
            decode = decode & (positions < anchor - 1 - self.window_size)
        return (positions >= 0) & (positions < anchor - 1) & (
            (text & (self.prefill_text == "keep"))
            | (visual & (self.prefill_vision == "keep")) | decode
        )


def visual_positions(input_ids, config):
    """Include visual delimiters in the visual interface; never leave them behind."""
    ids = [getattr(config, key, None) for key in (
        "image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"
    )]
    result = torch.zeros_like(input_ids, dtype=torch.bool)
    for value in ids:
        if value is not None:
            result |= input_ids == value
    return result
