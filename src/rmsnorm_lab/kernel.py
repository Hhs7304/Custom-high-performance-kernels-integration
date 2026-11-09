from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # Allows importing the reference/runtime on CPU-only installs.
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _rmsnorm_kernel(
        X, W, Y,
        N: tl.constexpr,
        EPS: tl.constexpr,
        X_ROW_STRIDE: tl.constexpr,
        Y_ROW_STRIDE: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X + row * X_ROW_STRIDE + cols, mask=mask, other=0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0).to(tl.float32)
        mean_square = tl.sum(x * x, axis=0) / N
        inv_rms = tl.rsqrt(mean_square + EPS)
        y = x * inv_rms * w
        tl.store(Y + row * Y_ROW_STRIDE + cols, y, mask=mask)


def rms_norm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, num_warps: int = 4) -> torch.Tensor:
    """Apply last-dimension RMSNorm using one Triton program per row."""
    if triton is None:
        raise RuntimeError("Triton is required for rms_norm_triton; install the CUDA-enabled project dependencies")
    if not x.is_cuda or not weight.is_cuda:
        raise ValueError("x and weight must be CUDA tensors")
    if x.ndim < 1 or x.shape[-1] == 0:
        raise ValueError("x must have a non-empty last dimension")
    if weight.ndim != 1 or weight.numel() != x.shape[-1]:
        raise ValueError("weight must be 1D and match x.shape[-1]")
    if x.device != weight.device:
        raise ValueError("x and weight must be on the same device")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32) or weight.dtype != x.dtype:
        raise TypeError("x and weight must have the same dtype: float16, bfloat16, or float32")
    if not x.is_contiguous() and x.stride(-1) != 1:
        raise ValueError("x must have a contiguous last dimension")
    if num_warps not in (4, 8, 16, 32):
        raise ValueError("num_warps must be one of 4, 8, 16, or 32")

    n = x.shape[-1]
    rows = x.numel() // n
    x_2d = x.reshape(rows, n)
    y_2d = torch.empty_like(x_2d)
    block_n = triton.next_power_of_2(n)
    _rmsnorm_kernel[(rows,)](
        x_2d, weight, y_2d,
        N=n,
        EPS=eps,
        X_ROW_STRIDE=x_2d.stride(0),
        Y_ROW_STRIDE=y_2d.stride(0),
        BLOCK_N=block_n,
        num_warps=num_warps,
    )
    return y_2d.reshape(x.shape)
