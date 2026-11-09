from __future__ import annotations

import torch


def rms_norm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Eager PyTorch reference for last-dimension RMSNorm."""
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")
    if weight.ndim != 1 or weight.numel() != x.shape[-1]:
        raise ValueError("weight must be 1D and match x.shape[-1]")
    variance = x.float().square().mean(dim=-1, keepdim=True)
    normalized = x.float() * torch.rsqrt(variance + eps)
    return (normalized * weight.float()).to(dtype=x.dtype)
