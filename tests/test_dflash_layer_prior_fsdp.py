"""Capture-boundary regression plus optional real Torch flat-buffer checks."""
import ast
import __future__
from pathlib import Path
from types import SimpleNamespace
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]


def actual_helper(torch_module):
    tree = ast.parse((ROOT/'litespec/modeling/draft/context_layers.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'materialize_layer_prior')
    namespace = dict(torch=torch_module)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'actual_helper', 'exec'), namespace)
    return namespace['materialize_layer_prior']


class CaptureBoundaryTests(unittest.TestCase):
    def test_actual_forward_clones_once_before_decoder_layers_and_keeps_none(self):
        tree = ast.parse((ROOT/'litespec/modeling/draft/dflash.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DFlashDraftModel')
        forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
        memory_format = object()
        helper = actual_helper(SimpleNamespace(contiguous_format=memory_format))
        namespace = dict(materialize_layer_prior=helper)
        exec(compile(ast.Module(body=[forward], type_ignores=[]), 'actual_forward', 'exec',
                     flags=__future__.annotations.compiler_flag), namespace)

        class Feature:
            shape = (1, 2, 320)
            def reshape(self, *args): return self
            def flatten(self, *args): return self
            def __add__(self, other): return self
            def __getitem__(self, key): return self

        class Bias:
            def __init__(self): self.clones = 0
            def clone(self, **kwargs):
                self.clones += 1
                if kwargs != dict(memory_format=memory_format): raise AssertionError(kwargs)
                return SimpleNamespace(source=self, _base=None)
            def detach(self): raise AssertionError('must preserve the learned prior gradient')

        for bias in (None, Bias()):
            captures = []
            def layer(**kwargs):
                captures.append(kwargs['context_layer_bias'])
                if bias is not None: self.assertEqual(bias.clones, 1)
                return kwargs['hidden_states']
            draft = SimpleNamespace(context_slots=5, context_projection='per_layer',
                context_depth=SimpleNamespace(weight=Feature()), fc=lambda x:x, hidden_norm=lambda x:x,
                rotary_emb=lambda *args: (), norm=lambda x:x, layer_types=['full_attention']*2,
                layers=[layer,layer], context_layer_bias=bias)
            namespace['forward'](draft, position_ids=Feature(), target_hidden=Feature(), noise_embedding=Feature())
            self.assertEqual(len(captures), 2)
            self.assertIs(captures[0], captures[1])
            if bias is None:
                self.assertIsNone(captures[0])
            else:
                self.assertIsNot(captures[0], bias)
                self.assertIs(captures[0].source, bias)
                self.assertIsNone(captures[0]._base)


@unittest.skipUnless(torch is not None, 'real Torch numerical tests require the VLM environment')
class RealTensorTests(unittest.TestCase):
    def test_clone_has_independent_storage_and_identical_gradient(self):
        helper = actual_helper(torch)
        bias = torch.nn.Parameter(torch.randn(7,5))
        clone = helper(bias)
        self.assertIsNone(clone._base)
        self.assertNotEqual(clone.data_ptr(), bias.data_ptr())
        self.assertTrue(clone.requires_grad)
        readout = torch.randn_like(bias)
        (clone * readout).sum().backward()
        torch.testing.assert_close(bias.grad, readout)
        self.assertIsNone(helper(None))

    def test_fsdp_style_full_base_local_gradient_is_not_captured(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        # Simulate the state seen during SHARD_GRAD_OP accumulation: a full
        # parameter storage and a gradient whose physical storage is one shard.
        flat = torch.nn.Parameter(torch.zeros(128))
        flat.grad = torch.ones_like(flat)
        flat.data = torch.zeros(512)
        prior = flat[:35].reshape(7,5)
        self.assertEqual(flat.numel(), 4 * flat.grad.numel())
        self.assertIsNotNone(prior._base)
        clone = actual_helper(torch)(prior)
        self.assertIsNone(clone._base)
        fake = FakeTensorMode().from_tensor(clone)
        self.assertEqual(tuple(fake.shape), (7,5))
        self.assertTrue(fake.requires_grad)
        # Restore consistent leaf metadata, then verify the clone still reaches
        # the underlying parameter rather than silently disabling C learning.
        flat.grad = None
        clone.sum().backward()
        torch.testing.assert_close(flat.grad[:35], torch.ones(35))
        self.assertEqual(torch.count_nonzero(flat.grad[35:]).item(), 0)


if __name__ == '__main__': unittest.main()
