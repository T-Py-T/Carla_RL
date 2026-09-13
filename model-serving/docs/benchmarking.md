# Benchmarking

Benchmark the exact model and deployment configuration you intend to use. The
scripts can measure latency, throughput, memory, batch sizes, and detected
hardware; configuration thresholds are not measured results.

## Local run

Generate and validate an artifact before collecting a service benchmark:

```bash
uv run python -m scripts.create_example_artifacts \
  --output artifacts \
  --version v0.1.0
uv run python scripts/validate_artifacts.py \
  --artifact-dir artifacts/v0.1.0
uv run python scripts/local_benchmark.py --help
```

Choose the quick or full mode from the script help, and write output to a new
file rather than replacing a previous run.

## Record with every result

- repository commit and model-artifact hashes;
- model card and preprocessing state;
- Python, PyTorch, operating system, CPU/GPU, and memory details;
- device, worker count, batch sizes, warmup, duration, and concurrency;
- exact command, raw output, exit status, and errors; and
- repetitions and run-to-run variation.

Use the same artifact, request population, warmup, and process settings for a
before-and-after comparison. A benchmark of the generated example policy
measures service mechanics, not a trained driving model.
