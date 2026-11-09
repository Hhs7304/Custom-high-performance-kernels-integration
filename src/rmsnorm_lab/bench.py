from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from .kernel import rms_norm_triton
from .reference import rms_norm_reference


def _parse_shapes(value: str) -> list[tuple[int, int]]:
    shapes = []
    for item in value.split(","):
        try:
            rows, cols = (int(part) for part in item.lower().split("x", maxsplit=1))
        except (ValueError, TypeError) as exc:
            raise argparse.ArgumentTypeError(f"invalid shape {item!r}; expected ROWSxCOLS") from exc
        if rows <= 0 or cols <= 0:
            raise argparse.ArgumentTypeError("shape dimensions must be positive")
        shapes.append((rows, cols))
    return shapes


def _time_ms(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iterations):
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def _dtype(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; run this benchmark on a CUDA-enabled host")
    dtype = _dtype(args.dtype)
    device = torch.device(args.device)
    results = []
    for rows, cols in args.shapes:
        x = torch.randn((rows, cols), device=device, dtype=dtype)
        weight = torch.randn((cols,), device=device, dtype=dtype)
        expected = rms_norm_reference(x, weight, args.eps)
        actual = rms_norm_triton(x, weight, args.eps, num_warps=args.num_warps)
        torch.testing.assert_close(actual, expected, rtol=args.rtol, atol=args.atol)

        ref_ms = _time_ms(lambda: rms_norm_reference(x, weight, args.eps), args.warmup, args.iterations)
        tri_ms = _time_ms(lambda: rms_norm_triton(x, weight, args.eps, num_warps=args.num_warps), args.warmup, args.iterations)
        element_bytes = torch.empty((), dtype=dtype).element_size()
        min_bytes = (2 * rows * cols + cols) * element_bytes
        effective_gbps = min_bytes / (tri_ms * 1e-3) / 1e9
        results.append({
            "rows": rows,
            "hidden_size": cols,
            "dtype": args.dtype,
            "reference_latency_ms": round(ref_ms, 6),
            "triton_latency_ms": round(tri_ms, 6),
            "speedup": round(ref_ms / tri_ms, 4),
            "estimated_minimum_traffic_bytes": min_bytes,
            "estimated_effective_bandwidth_gbps": round(effective_gbps, 3),
            "max_abs_error": float((actual.float() - expected.float()).abs().max().item()),
        })

    return {
        "environment": {
            "gpu": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "triton_version": __import__("triton").__version__,
            "cuda_runtime": torch.version.cuda,
            "dtype": args.dtype,
            "eps": args.eps,
            "num_warps": args.num_warps,
            "warmup": args.warmup,
            "iterations": args.iterations,
        },
        "results": results,
        "note": "Bandwidth is estimated from minimum activation read/write plus scale traffic; it is not a hardware-counter measurement.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Correctness-checked RMSNorm reference vs Triton benchmark")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--shapes", type=_parse_shapes, default=_parse_shapes("128x4096,512x4096,128x8192"))
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("warmup must be non-negative and iterations must be positive")
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
