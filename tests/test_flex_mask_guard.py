"""CPU regression checks for CUDA mask closure/guard handling."""
import inspect

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask

from litespec.core import attention_masks as masks
import litespec.modeling.draft.flex_attention as runtime


def inputs(blocks=12):
    anchors = torch.arange(blocks)[None].expand(2, -1) + 32
    keep = torch.ones(2, blocks, dtype=torch.bool)
    keep[1, -1] = False
    tokens = keep.unsqueeze(-1).expand(-1, -1, 1)
    return anchors, keep, tokens


@pytest.mark.parametrize('role', ['student', 'swa_teacher'])
def test_dlite_closure_owns_tensors_and_preserves_visibility(monkeypatch, role):
    monkeypatch.setattr(masks, 'create_flex_block_mask', lambda mask_mod, **kw: mask_mod)
    anchors, keep, tokens = inputs(2)
    mod = masks.create_dlite_flex_mask(model_role=role, anchor_positions=anchors,
        block_keep_mask=keep, token_keep_mask=tokens, seq_len=64, swa_window_size=16,
        chs_slots=12, query_len=16, device='cpu')
    captured = inspect.getclosurevars(mod).nonlocals
    for key in ('anchor_positions', 'block_keep_mask', 'token_keep_mask'):
        tensor = captured[key]
        assert tensor._base is None and tensor.is_contiguous()
    prefix = 64 if role == 'swa_teacher' else 0
    kv_len = prefix + 2*(12+16)
    q = torch.arange(32)[:, None]
    kv = torch.arange(kv_len)[None, :]
    for b in range(2):
        actual = mod(b, 0, q, kv)
        expected = torch.zeros(32, kv_len, dtype=torch.bool)
        for query in range(32):
            block = query // 16
            expected[query, prefix+block*12:prefix+(block+1)*12] = True
            if keep[b, block]:
                expected[query, prefix+24+block*16:prefix+24+(block+1)*16] = True
                if role == 'swa_teacher':
                    anchor = int(anchors[b, block])
                    expected[query, anchor-16:anchor-1] = True
        torch.testing.assert_close(actual, expected)


def test_dflash_closure_owns_tensors_and_matches_sdpa(monkeypatch):
    monkeypatch.setattr(masks, 'create_flex_block_mask', lambda mask_mod, **kw: mask_mod)
    anchors, keep, _ = inputs(2)
    mod = masks.create_dflash_flex_mask(anchors, keep, 64, 16, 'cpu')
    for value in inspect.getclosurevars(mod).nonlocals.values():
        if isinstance(value, torch.Tensor):
            assert value._base is None and value.is_contiguous()
    q, kv = torch.arange(32)[:, None], torch.arange(96)[None, :]
    expected = masks.create_dflash_sdpa_mask(anchors, keep, 64, 16, 'cpu')
    for b in range(2):
        torch.testing.assert_close(mod(b, 0, q, kv), expected[b, 0])


def test_dynamo_auto_dynamic_masks_accept_changing_view_shapes(monkeypatch):
    torch.set_num_threads(1)
    compiled = torch.compile(create_block_mask, backend='eager')
    monkeypatch.setattr(runtime, 'WrappedCreateBlockMask', lambda: lambda: compiled)
    monkeypatch.setattr(runtime, '_block_mask_guard_fallback', False)
    monkeypatch.delenv('LITESPEC_EAGER_BLOCK_MASK', raising=False)
    for width in (12, 15, 13):
        anchors, keep, tokens = inputs(width)
        block = masks.create_dlite_flex_mask(model_role='student', anchor_positions=anchors,
            block_keep_mask=keep, token_keep_mask=tokens, seq_len=64, swa_window_size=0,
            chs_slots=12, query_len=16, device='cpu')
        eager = create_block_mask(block.mask_mod, 2, None, width*16, width*28, 'cpu')
        torch.testing.assert_close(block.to_dense(), eager.to_dense())


def test_guard_assertion_falls_back_once_without_changing_mask(monkeypatch):
    monkeypatch.setattr(runtime, '_block_mask_guard_fallback', False)
    monkeypatch.delenv('LITESPEC_EAGER_BLOCK_MASK', raising=False)
    calls = []
    def broken(*args, **kwargs):
        calls.append(1)
        raise AssertionError("s94 (could be from L['mask_mod']._base.size()[1]) not in {s94: []}")
    monkeypatch.setattr(runtime, 'WrappedCreateBlockMask', lambda: lambda: broken)
    mod = lambda b, h, q, kv: q >= kv
    with pytest.warns(RuntimeWarning, match='symbolic-shape'):
        actual = runtime.compile_friendly_create_block_mask(mod, 2, None, 192, 256, 'cpu')
    again = runtime.compile_friendly_create_block_mask(mod, 2, None, 192, 256, 'cpu')
    expected = create_block_mask(mod, 2, None, 192, 256, 'cpu')
    assert calls == [1]
    torch.testing.assert_close(actual.to_dense(), expected.to_dense())
    torch.testing.assert_close(again.to_dense(), expected.to_dense())


def test_unrelated_assertion_is_not_suppressed(monkeypatch):
    monkeypatch.setattr(runtime, '_block_mask_guard_fallback', False)
    monkeypatch.delenv('LITESPEC_EAGER_BLOCK_MASK', raising=False)
    def broken(*args, **kwargs):
        raise AssertionError('unrelated shape error')
    monkeypatch.setattr(runtime, 'WrappedCreateBlockMask', lambda: lambda: broken)
    with pytest.raises(AssertionError, match='unrelated shape error'):
        runtime.compile_friendly_create_block_mask(lambda b,h,q,kv: q>=kv, 2, None, 192, 256, 'cpu')


def test_explicit_eager_mask_does_not_invoke_compiler(monkeypatch):
    monkeypatch.setenv('LITESPEC_EAGER_BLOCK_MASK', '1')
    def forbidden():
        raise AssertionError('mask compilation must be bypassed')
    monkeypatch.setattr(runtime, 'WrappedCreateBlockMask', forbidden)
    block = runtime.compile_friendly_create_block_mask(lambda b,h,q,kv: q>=kv, 2, None, 192, 256, 'cpu')
    assert block.mask_mod(0, 0, torch.tensor(4), torch.tensor(3))
