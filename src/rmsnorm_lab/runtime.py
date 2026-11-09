from __future__ import annotations

import torch
from torch import nn

from .kernel import rms_norm_triton
from .reference import rms_norm_reference


class RMSNorm(nn.Module):
    """Small runtime adapter dispatching CUDA inputs to Triton."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, dtype: torch.dtype = torch.float16):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.weight.numel():
            raise ValueError("input hidden dimension does not match RMSNorm weight")
        if x.is_cuda:
            return rms_norm_triton(x, self.weight, self.eps)
        return rms_norm_reference(x, self.weight, self.eps)
