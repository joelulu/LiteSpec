"""Fail-fast CUDA validation of fused GDN verify, prefix selection and replay.

No checkpoint or pytest dependency. This checks the GPU kernel, not task quality.
The full HF hybrid cache is separately covered by tests/test_gdn_state.py.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from litespec.vlm.gdn_state import reference_recurrence, recurrence


def check(device="cuda", width=16):
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for state-kernel validation")
    torch.manual_seed(42)
    rows = []
    for dtype in (torch.float32, torch.bfloat16):
        # Native Qwen3.5 state dimensions, nonzero start state, B > 1.
        shape = (2, width, 4)
        q, k = [torch.randn(*shape, 128, device=device, dtype=dtype) for _ in range(2)]
        v = torch.randn(*shape, 128, device=device, dtype=dtype)
        g = -torch.rand(*shape, device=device, dtype=torch.float32)
        beta = torch.rand(*shape, device=device, dtype=dtype)
        initial = torch.randn(2, 4, 128, 128, device=device)
        expected, final, states = reference_recurrence(q, k, v, g, beta, initial, save_states=True)
        actual, got_final, got_states = recurrence(q, k, v, g, beta, initial, save_states=True)
        torch.testing.assert_close(actual, expected, atol=.02 if dtype == torch.bfloat16 else 2e-5,
                                   rtol=.02 if dtype == torch.bfloat16 else 2e-5)
        torch.testing.assert_close(got_states, states, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(got_final, final, atol=2e-5, rtol=2e-5)
        errors = []
        for kept in range(1, width+1):
            destination = torch.empty_like(initial)
            _, committed, _ = recurrence(None, k, v, g, beta, initial, out_state=destination, steps=kept)
            torch.testing.assert_close(committed, got_states[:, kept-1], atol=2e-5, rtol=2e-5)
            errors.append((committed-got_states[:, kept-1]).abs().max().item())
        rows.append(dict(dtype=str(dtype), width=width, state_max_abs_error=(got_states-states).abs().max().item(),
                         replay_vs_selected_state_max_abs_error=max(errors), passed=True))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=16)
    args = parser.parse_args()
    if args.width < 1:
        parser.error("positive width required")
    print(json.dumps(check(args.device, args.width), indent=2))
