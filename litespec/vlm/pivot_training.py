"""D0/D1/D2 two-round feedback training; cached prefix by default.

The original full-prefix implementation remains an explicit numerical reference,
not a fallback selected silently after cache errors.
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

    def make_batch(self, ids, source, mask, anchors, *, target_hidden=None):
        return self.prepare_batch(ids, source, mask, anchor_positions=anchors,
            block_keep_mask=torch.ones_like(anchors, dtype=torch.bool),target_hidden=target_hidden)

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
        copies = self.rollout_anchors
        repeat_inputs = rollout_inputs if self.rollout_backend == "full_replay" else {
            k:v for k,v in rollout_inputs.items() if k != "pixel_values"}
        repeated = repeat_examples(repeat_inputs,copies)
        mask = rollout_mask.repeat(copies,1)
        valid = eligible_anchors(mask,self.block_size)
        if not valid.any(-1).all():
            raise ValueError("two-round training requires 2*block_size contiguous answer tokens per sample")
        anchors = valid.float().multinomial(1)
        if self.rollout_backend == "full_replay":
            return self.forward_full_replay(rollout_inputs=repeated,rollout_mask=mask,
                anchors=anchors,collect_detailed_metrics=collect_detailed_metrics)
        return self.forward_cached(rollout_inputs,repeated,mask,anchors,copies,
                                   collect_detailed_metrics=collect_detailed_metrics)

    def forward_cached(self, original, repeated, mask, anchors, copies, *, collect_detailed_metrics):
        from .pivot_cache import CachedRolloutPrefix
        d = self.draft_model
        if d.pivot_augmentation.mode not in ("A","D0","D1","D2"):
            raise ValueError("cached two-round training is only for D0/D1/D2; A/B/C/E use shared SFT")
        ids = repeated["input_ids"]
        prefix = CachedRolloutPrefix(self._frozen_target,d,original,repeated,anchors,copies)
        offsets = torch.arange(self.block_size-1,device=ids.device)
        # Clean round-one labels require only the short teacher-forced block.
        teacher = prefix.run(ids.gather(1,anchors+offsets),hidden=False)
        reference = teacher.logits[:,None].detach()
        del teacher
        batch = self.make_batch(ids,None,mask,anchors,target_hidden=prefix.pivot)
        hidden = self.backbone(batch,{})
        first = self.compute_supervised_loss(hidden,batch,reference,
            collect_detailed_metrics=collect_detailed_metrics)
        with torch.no_grad():
            proposals = d.sample_draft_tokens(draft_hidden=hidden[:,0].detach(),
                first_prev_token_ids=ids.gather(1,anchors)[:,0],temperature=0.)[0]
            candidates = torch.cat([ids.gather(1,anchors),proposals],1)
            verified = prefix.run(candidates)
            predictions = verified.logits.argmax(-1)
            kept,next_anchor,bonus,corrected = transition(ids,anchors,candidates,predictions)
            has_feedback = kept < self.block_size
            reject = torch.where(has_feedback,kept,torch.zeros_like(kept))[:,None]
            raw = gather_positions(source_layers(verified.hidden_states,d),reject,d.target_layer_ids).clone()
            raw *= has_feedback[:,None,None,None].to(raw.dtype)
            wrong = candidates.gather(1,kept.clamp(max=self.block_size-1)[:,None])[:,0]
            feedback = dict(feedback=raw,feedback_valid=has_feedback,
                wrong_embedding=self.embed_tokens(wrong),correct_embedding=self.embed_tokens(bonus),
                distance=torch.ones_like(bonus))
        del verified,reference
        # Variable accept lengths share the same original prefix. Replaying at
        # most 2B-1 TEXT tokens avoids padded-tail GDN updates or full-prefix replay.
        segment = corrected.gather(1,anchors+torch.arange(2*self.block_size-1,device=ids.device))
        second_teacher = prefix.run(segment,rollback=False)
        second_source = source_layers(second_teacher.hidden_states,d)
        second_pivot = gather_positions(second_source,(kept-1)[:,None],d.target_layer_ids)
        indices = kept[:,None]+offsets
        second_reference = second_teacher.logits.gather(1,indices[...,None]
            .expand(-1,-1,second_teacher.logits.size(-1)))[:,None].detach()
        second_batch = self.make_batch(corrected,None,mask,next_anchor,target_hidden=second_pivot)
        self.mask_stopped(second_batch,candidates,kept,bonus,feedback)
        second_hidden = self.backbone(second_batch,{},feedback)
        second = self.compute_supervised_loss(second_hidden,second_batch,second_reference,
            collect_detailed_metrics=collect_detailed_metrics)
        result = self.combine(first,second,has_feedback)
        result.sum_metrics.update({
            "rollout/target_prefix_calls":ids.new_tensor(1),
            "rollout/target_branch_calls":ids.new_tensor(prefix.branch_calls),
            "rollout/visual_encoder_calls":ids.new_tensor(prefix.vision_encodes),
            "rollout/target_prefix_padded_tokens":ids.new_tensor(prefix.target_tokens),
            "rollout/target_branch_tokens":ids.new_tensor(prefix.branch_tokens),
        })
        return result

    def mask_stopped(self,batch,candidates,kept,bonus,feedback):
        eos = getattr(self._frozen_target.config,"text_config",self._frozen_target.config).eos_token_id
        eos = eos if isinstance(eos,(list,tuple)) else [eos]
        eos = candidates.new_tensor([x for x in eos if x is not None])
        within = torch.arange(self.block_size,device=candidates.device)[None] < kept[:,None]
        stopped = (torch.isin(candidates,eos)&within).any(-1)|torch.isin(bonus,eos)
        batch.raw_weight_mask[stopped] = 0
        batch.binary_eval_mask[stopped] = False
        if not batch.binary_eval_mask.any():
            batch.binary_eval_mask[0,0,0] = True  # zero-weight dummy, same rank collectives
        feedback["feedback_valid"] &= ~stopped

    def combine(self,first,second,has_feedback):
        numerator = first.loss_terms.numerator+second.loss_terms.numerator
        denominator = first.loss_terms.denominator+second.loss_terms.denominator
        metrics = {}
        second_empty = not bool(second.loss_terms.denominator > 0)
        for label,item in (("round1",first),("round2",second)):
            for key,value in item.ratio_metrics.items():
                # Do not count the zero-weight EOS dummy as a real prediction.
                if label == "round2" and second_empty:
                    value = RatioTerms(value.numerator*0,value.denominator*0)
                metrics[f"{label}/{key}"] = value
        metrics["rollout/feedback_available"] = RatioTerms(has_feedback.sum().detach(),has_feedback.new_tensor(has_feedback.numel(),dtype=torch.float))
        return TrainingOutput(fsdp_global_weighted_mean(numerator,denominator,self.process_group),
                              RatioTerms(numerator,denominator),metrics)

    def forward_full_replay(self, *, rollout_inputs, rollout_mask, anchors,
                            collect_detailed_metrics=True):
        d = self.draft_model
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
        self.mask_stopped(second_batch,candidates,kept,bonus,feedback)
        second_hidden = self.backbone(second_batch, second_source, feedback)
        second = self.compute_supervised_loss(second_hidden, second_batch, second_reference,
            collect_detailed_metrics=collect_detailed_metrics)
        return self.combine(first,second,has_feedback)
