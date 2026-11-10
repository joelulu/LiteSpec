"""Numerical B/C smoke check; no Target weights or external data required.

Run with the training interpreter on the server:
  python tests/check_dflash_layer_adapters.py --device cuda --flex
"""
import argparse
import tempfile
from pathlib import Path


def check(device='cpu', flex=False):
    import torch
    from transformers import Qwen3Config
    from litespec.modeling.draft.dflash import DFlashDraftModel
    from litespec.modeling.draft.context_layers import dense_layer_prior_mask
    from litespec.core.attention_masks import create_dflash_sdpa_mask, create_dflash_flex_mask
    from litespec.training.optimizer import register_optimizer_parameters

    torch.manual_seed(42)
    block, slots, width, sequence = 8, 5, 64, 20

    def model(projection, prior):
        cfg = Qwen3Config(vocab_size=128, hidden_size=width, intermediate_size=128,
                          num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                          head_dim=16, layer_types=['full_attention'], attention_dropout=0.0)
        cfg.architectures = ['DFlashDraftModel']
        cfg.num_target_layers, cfg.block_size = 32, block
        cfg.dflash_config = dict(target_layer_ids=[1,8,15,22,29], mask_token_id=127,
                                 context_projection=projection, position_layer_bias=prior)
        cfg.vlm_config = dict(draft_context='pivot5')
        cfg._attn_implementation = 'sdpa'
        return DFlashDraftModel(cfg).to(device).eval()

    base, b, c = model('shared', False), model('per_layer', False), model('per_layer', True)
    common = {k: v for k, v in base.state_dict().items() if k != 'fc.weight'}
    b.load_state_dict(common, strict=False)
    with torch.no_grad():
        for projection in b.fc.projections:
            projection.weight.copy_(base.fc.weight)
    missing = c.load_state_dict(b.state_dict(), strict=False)
    assert missing.missing_keys == ['context_layer_bias'] and not missing.unexpected_keys
    assert torch.count_nonzero(c.context_layer_bias) == 0
    _, specs = register_optimizer_parameters(c)
    assert len(specs) == len(list(c.parameters()))

    # Two training anchors share a full Target sequence. Only the appropriate
    # preceding pivot and each anchor's own draft block are visible.
    anchors = torch.tensor([[6,13]], device=device)
    keep = torch.ones_like(anchors, dtype=torch.bool)
    mask = create_dflash_sdpa_mask(anchors, keep, sequence, block, device,
                                  context_window=1, context_slots=slots)
    context = torch.randn(1, sequence, slots * width, device=device)
    query = torch.randn(1, 2 * block, width, device=device)
    positions = torch.cat([torch.arange(sequence, device=device).repeat_interleave(slots)[None],
                           (anchors[...,None] + torch.arange(block, device=device)).flatten(1)], dim=1)

    def forward(draft, attention_mask=mask):
        return draft(position_ids=positions, target_hidden=context, noise_embedding=query,
                     attention_mask=attention_mask, is_causal=False)

    with torch.no_grad():
        torch.testing.assert_close(forward(base), forward(b), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(forward(b), forward(c), atol=1e-6, rtol=1e-5)
        c.context_layer_bias.normal_(0, 0.3)
        trained = forward(c)
        for index, anchor in enumerate([6,13]):
            start, stop = index * block, (index + 1) * block
            pos = torch.cat([torch.full((slots,), anchor-1, device=device),
                             torch.arange(anchor, anchor+block, device=device)])[None]
            inferred = c(position_ids=pos, target_hidden=context[:,anchor-1:anchor],
                         noise_embedding=query[:,start:stop], is_causal=False)
            torch.testing.assert_close(trained[:,start:stop], inferred, atol=2e-6, rtol=1e-5)

    # Offset indexing restarts at every anchor; only context columns change.
    prior = dense_layer_prior_mask(None, c.context_layer_bias, block, slots, sequence*slots,
                                   2*block, sequence*slots+2*block, torch.float32, device)
    assert torch.count_nonzero(prior[[0,block]]) == 0
    assert torch.count_nonzero(prior[:,sequence*slots:]) == 0
    torch.testing.assert_close(prior[1,:slots], c.context_layer_bias[0])
    torch.testing.assert_close(prior[block+7,:slots], c.context_layer_bias[6])

    readout = torch.randn_like(query)
    dense_output = forward(c)
    (dense_output * readout).mean().backward()
    dense_grad = c.context_layer_bias.grad.detach().clone()
    assert torch.isfinite(dense_grad).all() and dense_grad.abs().sum() > 0
    for projection in c.fc.projections:
        assert projection.weight.grad is not None and projection.weight.grad.abs().sum() > 0

    # Check the real optimizer can update the table and matrices.
    optimizer = torch.optim.AdamW(c.parameters(), lr=1e-3)
    before = c.context_layer_bias.detach().clone()
    optimizer.step()
    assert not torch.equal(before, c.context_layer_bias)
    optimizer.zero_grad(set_to_none=True)
    with tempfile.TemporaryDirectory() as directory:
        c.save_pretrained(directory)
        restored = DFlashDraftModel.from_pretrained(directory, attn_implementation='sdpa').to(device).eval()
        assert restored.context_projection == 'per_layer'
        torch.testing.assert_close(forward(restored), forward(c))

    if flex:
        if not str(device).startswith('cuda'):
            raise ValueError('--flex requires CUDA')
        # A nonzero prior verifies both forward application and captured-buffer
        # gradients through the exact compiled FlexAttention training path.
        dense_output = forward(c)
        (dense_output * readout).mean().backward()
        dense_grad = c.context_layer_bias.grad.detach().clone()
        projection_grad = c.fc.projections[0].weight.grad.detach().clone()
        c.zero_grad(set_to_none=True)
        c.config._attn_implementation = 'flex_attention'
        flex_mask = create_dflash_flex_mask(anchors, keep, sequence, block, device,
                                           context_window=1, context_slots=slots)
        actual = forward(c, flex_mask)
        (actual * readout).mean().backward()
        torch.testing.assert_close(actual, dense_output, atol=3e-4, rtol=3e-3)
        torch.testing.assert_close(c.context_layer_bias.grad, dense_grad, atol=3e-5, rtol=5e-3)
        torch.testing.assert_close(c.fc.projections[0].weight.grad, projection_grad, atol=3e-5, rtol=5e-3)
        assert c.context_layer_bias.grad.abs().sum() > 0
    print(f'B/C numerical check passed: device={device}, Flex={flex}; projection, prior, gradients, parity and reload', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--flex', action='store_true')
    args = parser.parse_args()
    check(args.device, args.flex)
