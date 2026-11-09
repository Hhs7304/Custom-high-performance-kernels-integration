from __future__ import annotations

import pytest
import torch

from rmsnorm_lab import rms_norm_reference, rms_norm_triton


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernel tests")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(1, 128), (7, 1024), (32, 4096)])
def test_triton_matches_reference(dtype, shape):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    weight = torch.randn(shape[-1], device="cuda", dtype=dtype)
    expected = rms_norm_reference(x, weight)
    actual = rms_norm_triton(x, weight)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_accepts_non_contiguous_outer_dimensions():
    base = torch.randn((4, 8, 1024), device="cuda", dtype=torch.float16)
    x = base.transpose(0, 1)
    weight = torch.randn(1024, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(rms_norm_triton(x, weight), rms_norm_reference(x, weight), rtol=2e-2, atol=2e-2)
