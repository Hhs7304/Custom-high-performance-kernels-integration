# Triton RMSNorm kernel lab

A small, runnable GPU-kernel project focused on fused RMSNorm: Triton implementation, eager PyTorch reference, correctness tests, a benchmark harness, and a lightweight runtime adapter. It is intended as an engineering starting point for inference-engine kernel work—not as a claimed upstream vLLM patch.

## Why RMSNorm?

RMSNorm is used in decoder blocks and is a useful kernel case study: each row needs a reduction, then normalization and scale. A decomposed eager implementation can launch several kernels and materialize intermediate tensors. A fused implementation reads the activation once, reduces in FP32, and writes the result once, avoiding intermediate tensors.

## Implementation

`src/rmsnorm_lab/kernel.py` implements one CTA per row. It uses masked power-of-two lanes, FP32 accumulation for the sum of squares, and writes the input dtype. Input must be a CUDA tensor with a contiguous last dimension; non-contiguous row strides are supported. The benchmark compares against a simple eager PyTorch reference with the same epsilon and formula.

The theoretical minimum traffic for an `M x N` activation is approximately `2*M*N*element_size` bytes (one activation read and one output write), plus `N*element_size` for the scale vector. This is a lower-bound traffic model, not an exact measurement: cache behavior, transaction granularity, launch overhead, and the repeated scale-vector access affect observed bandwidth.

## Run

Install a PyTorch build compatible with your CUDA driver, then:

```bash
pip install -e '.[dev]'
pytest -q
python -m rmsnorm_lab.bench --device cuda --dtype fp16 --shapes 128x4096,512x4096,128x8192
```

The benchmark checks output correctness before timing and writes JSON to `results/benchmark.json` by default. It records the GPU name, dtype, shape, latency, effective bandwidth estimate, and speedup. Results are hardware- and software-specific; no performance gain is asserted until you run it on a GPU. For a useful report, include the GPU model, driver, CUDA, PyTorch, Triton versions and run several representative shapes.

Try `--dtype bf16` or `--dtype fp32`. `--help` lists timing and output options. Tests and benchmark cleanly skip/fail with actionable messages when CUDA/Triton is unavailable.

## Performance analysis checklist

RMSNorm performs roughly a handful of arithmetic operations per element, while at least one read and one write are required. For large rows it is generally expected to be memory-traffic-sensitive; for small batches or narrow rows, reduction and launch overhead can dominate. The benchmark estimates effective bandwidth from the minimum traffic model. Compare it with the device's documented peak bandwidth only as a rough roofline reference—this estimate is not a hardware-counter measurement.

For deeper analysis, profile both paths with Nsight Compute (for example, inspect achieved DRAM bandwidth, occupancy, register use, and memory transactions). Triton launch parameters (`num_warps`) are configurable in the Python wrapper. Sweep them on target hardware instead of assuming one setting is universally best. The current kernel uses one CTA per row; extremely wide rows and small row counts are candidates for further tuning.

## Inference-engine integration

`src/rmsnorm_lab/runtime.py` provides a small module that selects the Triton kernel on CUDA and the PyTorch reference otherwise. This is the runnable integration boundary for a small custom runtime. To integrate in vLLM, adapt the call at the RMSNorm module boundary and use the exact custom-op registration API supported by the target vLLM version; that API is version-sensitive and is intentionally not guessed here. Before upstreaming, add engine-specific dispatch, supported shape/dtype constraints, graph-capture validation, and tests on the supported GPU matrix.

## Scope and limitations

- This is a standalone implementation, not a change to the vLLM repository.
- It does not claim measured speedups; run the benchmark on your target GPU.
- It currently supports CUDA tensors with FP16, BF16, or FP32 input and scale, with a contiguous last dimension.
- It does not implement residual fusion, quantized scale, distributed execution, or a full vLLM custom-op registration.

## Repository layout

```text
src/rmsnorm_lab/kernel.py       Triton kernel and validated launcher
src/rmsnorm_lab/reference.py    eager PyTorch reference
src/rmsnorm_lab/runtime.py      small runtime adapter
src/rmsnorm_lab/bench.py        correctness-checked benchmark and traffic estimate
 tests/test_rmsnorm.py           correctness tests
```

## License

MIT
