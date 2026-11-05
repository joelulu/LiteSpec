"""Two actual draft/verify rounds, inside the FSDP forward boundary.

Target verification uses independent replicated examples (including images),
never a packed causal sequence whose earlier errors could poison other anchors.
Full-prefix HF recomputation is a correctness-first TRAINING path only.
"""
import torch

from litespec.core.dlite import OnlineDLiteModel
from litespec.core.outputs import TrainingOutput, RatioTerms
from litespec.core.block_metrics import fsdp_global_weighted_mean
from .pivot_augmentation import gather_positions, prefix_data
from .sw_token import source_layers


def eligible_anchors(mask, block_size):
    """Both rounds fit inside one contiguous supervised answer segment."""
    width = 2 * block_size
    if mask.size(1) < width:
        return mask.new_zeros(mask.shape, dtype=torch.bool)
    valid = (mask > .5).unfold(1, width, 1).all(-1)
    result = torch.zeros_like(mask, dtype=torch.bool)
    result[:, :valid.size(1)] = valid
    result[:, 0] = False  # pivot at a-1 must exist
    return result


def repeat_examples(inputs, count):
    """N-major repetition: the entire image/grid batch is repeated N times."""
    result = {}
    b = inputs["input_ids"].size(0)
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            raise ValueError(f"unsupported processor input {key}")
        if key == "position_ids" and value.ndim == 3:
            result[key] = value.repeat(1, count, 1)
        elif value.ndim:
            result[key] = value.repeat(count, *([1] * (value.ndim-1)))
        else:
            result[key] = value
    assert result["input_ids"].size(0) == b * count
    return result


def transition(ids, anchors, candidates, predictions):
    """Target-exact boundary; corrected input includes accepted tokens + bonus."""
    kept = (candidates[:, 1:] == predictions[:, :-1]).cumprod(-1).sum(-1) + 1
    next_anchor = anchors[:, 0] + kept
    corrected = ids.clone()
    row = torch.arange(ids.size(0), device=ids.device)
    for j in range(candidates.size(1)):
        active = j < kept
        corrected[row[active], anchors[active, 0]+j] = candidates[active, j]
    bonus = predictions.gather(1, (kept-1)[:, None])[:, 0]
    corrected[row, next_anchor] = bonus
    return kept, next_anchor[:, None], bonus, corrected


class OnlinePivot(OnlineDLiteModel):
    def bind_target(self, target):
        # Target is frozen and must not become an FSDP-owned child module.
        object.__setattr__(self, "_frozen_target", target)

    def target_pass(self, inputs, anchors, *, verify=False):
        span = self.block_size if verify else self.block_size-1
        positions = anchors + torch.arange(span, device=anchors.device)
        unique, inverse = positions.flatten().unique(sorted=True, return_inverse=True)
        with torch.no_grad():
            output = self._frozen_target(**inputs, use_cache=False,
                output_hidden_states=True, logits_to_keep=unique)
            logits = output.logits.gather(1, inverse.reshape(inputs["input_ids"].size(0), -1)
                [..., None].expand(-1, -1, output.logits.size(-1)))[:, None].detach()
            source = {i: x.detach() for i, x in source_layers(output.hidden_states, self.draft_model).items()}
        return source, logits

    def make_batch(self, ids, source, mask, anchors):
        return self.prepare_batch(ids, source, mask, anchor_positions=anchors,
            block_keep_mask=torch.ones_like(anchors, dtype=torch.bool))

    def backbone(self, batch, source, feedback=None):
        d = self.draft_model
        data = prefix_data(source, batch.anchor_positions, d)
        if feedback:
            data.update(feedback)
        b = batch.query_embeddings.size(0)
        cp, qp = d.build_block_position_ids(batch.anchor_positions, batch.token_position_ids, batch.token_keep_mask)
        query = batch.query_embeddings[:, 0]
        out = d(noise_embedding=query, target_hidden=batch.target_hidden,
            history_hidden=query.new_empty(b, 1, 0, query.size(-1)),
            position_ids=qp, rotary_position_ids=torch.cat([cp, qp], -1),
            attention_mask=None, is_causal=False, augmentation_data=data)
        return d._prediction_hidden(out)[:, None]

    def forward(self, *, rollout_inputs, rollout_mask, collect_detailed_metrics=True):
        d = self.draft_model
        valid = eligible_anchors(rollout_mask, self.block_size)
        if not valid.any(-1).all():
            raise ValueError("two-round training requires 2*block_size contiguous answer tokens per sample")
        # Each replica has exactly one independent random anchor.
        anchors = valid.float().multinomial(1)
        ids = rollout_inputs["input_ids"]
        source, reference = self.target_pass(rollout_inputs, anchors)
        batch = self.make_batch(ids, source, rollout_mask, anchors)
        hidden = self.backbone(batch, source)
        first = self.compute_supervised_loss(hidden, batch, reference,
            collect_detailed_metrics=collect_detailed_metrics)
        with torch.no_grad():
            proposals = d.sample_draft_tokens(draft_hidden=hidden[:, 0].detach(),
                first_prev_token_ids=ids.gather(1, anchors)[:, 0], temperature=0.)[0]
            candidates = torch.cat([ids.gather(1, anchors), proposals], 1)
            wrong_inputs = dict(rollout_inputs)
            wrong_ids = ids.clone()
            positions = anchors + torch.arange(self.block_size, device=ids.device)
            wrong_ids.scatter_(1, positions, candidates)
            wrong_inputs["input_ids"] = wrong_ids
            wrong_source, verify_logits = self.target_pass(wrong_inputs, anchors, verify=True)
            predictions = verify_logits[:, 0].argmax(-1)
            kept, next_anchor, bonus, corrected = transition(ids, anchors, candidates, predictions)
            has_feedback = kept < self.block_size
            reject = torch.where(has_feedback[:, None], next_anchor, anchors)
            wrong = candidates.gather(1, kept.clamp(max=self.block_size-1)[:, None])[:, 0]
            # One rejected input position predicts the next anchor's first candidate.
            raw_feedback = gather_positions(wrong_source, reject, d.target_layer_ids)
            raw_feedback = raw_feedback * has_feedback[:, None, None, None].to(raw_feedback.dtype)
            feedback = dict(feedback=raw_feedback,
                feedback_valid=has_feedback, wrong_embedding=self.embed_tokens(wrong),
                correct_embedding=self.embed_tokens(bonus), distance=torch.ones_like(bonus))
        del wrong_source, verify_logits, source, reference
        second_inputs = dict(rollout_inputs, input_ids=corrected)
        second_source, second_reference = self.target_pass(second_inputs, next_anchor)
        second_batch = self.make_batch(corrected, second_source, rollout_mask, next_anchor)
        # An EOS in the kept block or bonus ends the request. Keep a zero-weight
        # dummy row if all local rows stop, so all ranks execute the same collectives.
        eos = getattr(self._frozen_target.config, "text_config", self._frozen_target.config).eos_token_id
        eos = eos if isinstance(eos, (list, tuple)) else [eos]
        eos = ids.new_tensor([x for x in eos if x is not None])
        kept_positions = torch.arange(self.block_size, device=ids.device)[None] < kept[:, None]
        stopped = (torch.isin(candidates, eos) & kept_positions).any(-1) | torch.isin(bonus, eos)
        second_batch.raw_weight_mask[stopped] = 0
        second_batch.binary_eval_mask[stopped] = False
        if not second_batch.binary_eval_mask.any():
            second_batch.binary_eval_mask[0, 0, 0] = True
        feedback["feedback_valid"] &= ~stopped
        second_hidden = self.backbone(second_batch, second_source, feedback)
        second = self.compute_supervised_loss(second_hidden, second_batch, second_reference,
            collect_detailed_metrics=collect_detailed_metrics)
        numerator = first.loss_terms.numerator + second.loss_terms.numerator
        denominator = first.loss_terms.denominator + second.loss_terms.denominator
        metrics = {}
        for label, item in (("round1", first), ("round2", second)):
            for key, value in item.ratio_metrics.items():
                metrics[f"{label}/{key}"] = value
        metrics["rollout/feedback_available"] = RatioTerms(has_feedback.sum().detach(), has_feedback.new_tensor(has_feedback.numel(), dtype=torch.float))
        return TrainingOutput(fsdp_global_weighted_mean(numerator, denominator, self.process_group),
            RatioTerms(numerator, denominator), metrics)
