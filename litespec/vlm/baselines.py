"""Original DFlash/DSpark models and losses with a frozen HF image target."""
from transformers import Qwen3Config
import torch

from litespec.modeling.draft.dflash import DFlashDraftModel, build_target_layer_ids
from litespec.modeling.draft.dspark import DSparkDraftModel
from litespec.core.dflash import OnlineDFlashModel
from litespec.core.dspark import OnlineDSparkModel
from .runtime import selected_layers


def is_baseline(draft):
    return isinstance(draft, DFlashDraftModel)


def load_draft(path, **kwargs):
    from transformers import AutoConfig
    from litespec.modeling.draft.dlite import DLiteDraftModel
    config = AutoConfig.from_pretrained(path)
    classes = {cls.__name__: cls for cls in (DLiteDraftModel, DFlashDraftModel, DSparkDraftModel)}
    name = config.architectures[0]
    if name not in classes:
        raise ValueError(f"unsupported VLM draft architecture: {name}")
    return classes[name].from_pretrained(path, **kwargs)


def make_draft(target, processor, cfg):
    text = target.config.text_config
    used = set(processor.tokenizer.get_vocab().values())
    mask = next((i for i in range(text.vocab_size-1, -1, -1) if i not in used), None)
    if mask is None:
        raise ValueError("no unused in-vocabulary MASK row")
    family = cfg["algorithm"]
    cls = DFlashDraftModel if family == "dflash" else DSparkDraftModel
    config = Qwen3Config(vocab_size=text.vocab_size, hidden_size=text.hidden_size,
                        intermediate_size=text.intermediate_size, num_hidden_layers=cfg["num_draft_layers"],
                        num_attention_heads=text.num_attention_heads, num_key_value_heads=text.num_key_value_heads,
                        head_dim=text.head_dim, rms_norm_eps=text.rms_norm_eps,
                        max_position_embeddings=text.max_position_embeddings,
                        layer_types=["full_attention"] * cfg["num_draft_layers"])
    config.architectures = [cls.__name__]
    config.num_target_layers = text.num_hidden_layers
    config.block_size = cfg["block_size"]
    config.sliding_window, config.use_sliding_window = None, False
    config._attn_implementation = "sdpa"
    layers = cfg.get("target_layer_ids") or build_target_layer_ids(text.num_hidden_layers, cfg["num_draft_layers"])
    method = dict(target_layer_ids=layers, mask_token_id=mask, attention_mode="gqa")
    if family == "dspark":
        method.update(projector_type="dspark", markov_head_type=cfg["markov_head_type"],
                      markov_rank=cfg["markov_rank"], enable_confidence_head=cfg["confidence_head"],
                      confidence_head_with_markov=cfg["confidence_head_with_markov"],
                      dspark_backbone_conv_enabled=False)
    setattr(config, family + "_config", method)
    config.vlm_config = dict(algorithm=family, target_model=cfg["target_model"],
                             context_policy=cfg["context_policy"], training="upstream_sft_v1")
    return cls(config).to(device=target.device, dtype=target.dtype)


def make_online(target, draft, cfg):
    options = dict(draft_model=draft, target_lm_head=target.get_output_embeddings(),
                   target_embed_tokens=target.get_input_embeddings(), mask_token_id=draft.mask_token_id,
                   block_size=cfg["block_size"], num_anchors=cfg["num_anchors"],
                   attention_backend="sdpa" if draft.device.type == "cpu" else "flex_attention",
                   flex_attention_backend=cfg["flex_attention_backend"],
                   loss_decay_gamma=cfg.get("loss_decay_gamma"),
                   objective_chunk_blocks=cfg.get("objective_chunk_blocks", 128))
    if cfg["algorithm"] == "dspark":
        return OnlineDSparkModel(**options, dspark_ce_loss_alpha=cfg["ce_loss_alpha"],
                                 dspark_l1_loss_alpha=cfg["l1_loss_alpha"],
                                 dspark_confidence_head_alpha=cfg["confidence_loss_alpha"])
    return OnlineDFlashModel(**options, selector_loss_alpha=0)


def prepare_forward(target, online, example, mask):
    with torch.no_grad():
        output = target(**example.inputs, output_hidden_states=True, use_cache=False, logits_to_keep=1)
        draft = online.draft_model
        hidden = selected_layers(output.hidden_states, draft.target_layer_ids, draft.config.num_target_layers).flatten(-2)
    return dict(input_ids=example.inputs["input_ids"], hidden_states=hidden, loss_mask=mask,
                target_last_hidden_states=output.hidden_states[-1])


class DraftCache:
    """Keep only projected target-history KV; discard speculative query KV."""
    def __init__(self, draft, target, output, prefill_length):
        from transformers import DynamicCache
        self.cache = DynamicCache()
        self.hidden = self.features(draft, output)
        self.history_tokens = prefill_length

    @staticmethod
    def features(draft, output):
        return selected_layers(output.hidden_states, draft.target_layer_ids,
                               draft.config.num_target_layers).flatten(-2).detach()

    def propose(self, draft, target, pending, anchor):
        ids = pending.new_full((pending.shape[0], draft.block_size), draft.mask_token_id)
        ids[:, :1] = pending
        start = self.cache.get_seq_length()
        positions = torch.arange(start, anchor + draft.block_size, device=pending.device)[None].expand(pending.shape[0], -1)
        hidden = draft(noise_embedding=target.get_input_embeddings()(ids), target_hidden=self.hidden,
                       position_ids=positions, past_key_values=self.cache, use_cache=True, is_causal=False)
        remove = self.cache.get_seq_length() - anchor
        if remove:
            # Negative removal counts work with both Transformers 4.x and 5.x.
            self.cache.crop(-remove)
        tokens, _, _ = draft._propose_draft_tokens(target, hidden, ids, temperature=0.0)
        return tokens, dict(physical_history=anchor, visible_history=anchor)

    def commit(self, draft, output, kept):
        self.hidden = self.features(draft, output)[:, :kept]
