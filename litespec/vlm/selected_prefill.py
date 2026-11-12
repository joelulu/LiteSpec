"""Native Transformers selective hidden capture; no activation math changes.

HF's layer list uses decoder OUTPUT indices, excludes the embedding slot and
preserves the final-norm convention for the last layer. Unselected positions in
the returned depth-length tuple are None, compatible with selected_layers().
"""


def target_prefill(target, draft, inputs, cache, *, needs_hidden, mode='all'):
    if mode not in ('all', 'selected'):
        raise ValueError(f'Unknown prefill hidden mode: {mode}')
    capture = bool(needs_hidden)
    if needs_hidden and mode == 'selected':
        depth = int(draft.config.num_target_layers)
        ids = list(draft.target_layer_ids)
        if not ids or any(type(i) is not int or i < 0 or i >= depth for i in ids):
            raise ValueError('Invalid selected Target layer indices')
        if getattr(target.config, 'model_type', '') != 'qwen3_5':
            raise ValueError('Selective prefill is validated only for native Qwen3.5 VLM; use mode=all')
        capture = sorted(set(ids))
    output = target(**inputs, past_key_values=cache, use_cache=True, logits_to_keep=1,
                    output_hidden_states=capture)
    if needs_hidden and mode == 'selected':
        states = output.hidden_states
        if len(states) != depth or any(states[i] is None for i in ids):
            raise RuntimeError('Transformers selective hidden capture ABI differs; use --prefill-hidden-mode all')
        if any(h is not None for i,h in enumerate(states) if i not in ids):
            raise RuntimeError('Target retained unrequested layer outputs; selective capture unavailable')
    return output
