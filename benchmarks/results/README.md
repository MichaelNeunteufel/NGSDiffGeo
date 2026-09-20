# Tensor-algebra benchmarks

The benchmark programs in `benchmarks/` measure construction, compilation,
evaluation, assembly, and retained-memory costs of tensor and differential-form
operations. Every timed workload first validates its result against an
independent or dense reference. Timing thresholds are intentionally not used in
CI; CI only smoke-tests benchmark modes, schemas, and correctness checks.

Generated reports belong in this directory but are ignored by Git by default.
They depend on the machine, compiler, NGSolve revision, thread count, and
runtime state. Summarize relevant comparisons in a merge request and archive
full raw reports externally when they are needed for long-term provenance.

## Stored reference reports

The repository retains a small historical reference set for the low-rank
`Raise` specialization:

- `tensor_algebra_before.json`: implementation before the low-rank matrix-product path;
- `tensor_algebra_after.json`: first paired comparison with symbolic einsum;
- `tensor_algebra_after_confirmation.json`: confirmation on a finer mesh;
- `tensor_algebra_extended.json`: higher-rank tensors, K-forms, and a `(2,2)` double form.

These files were recorded on Apple arm64 with NGSolve
`6.2.2606-330-g8cea3e679`. They are historical evidence, not portable
performance expectations. Already tracked reports remain tracked despite the
ignore rule; adding any new permanent result must be an explicit review
decision.

## Benchmark programs

| Program | Purpose |
| --- | --- |
| `benchmark_compressed_forms.py` | Compact/public/native/dense construction, graph compilation, scalar/SIMD evaluation, and retained expression graphs |
| `benchmark_form_dispatch.py` | Python dispatch and native-construction attribution |
| `benchmark_forms_application.py` | Construction through first assembly and repeated assembly of a mixed-form application |
| `benchmark_form_consumers.py` | Metric inner products, differentiated consumers, trace, slot inner product, and Hodge operations |
| `benchmark_form_direct_evaluation.py` | Direct interpreted evaluation at retained mapped points |
| `benchmark_revision_forms.py` | Workloads intended for comparisons between two source or installation revisions, including native compilation |
| `compare_compressed_forms.py` | Aggregate independently recorded compact-form reports and evaluate optional acceptance rules |
| `compare_form_revisions.py` | Run counterbalanced baseline/candidate comparisons from two Python package locations |

All programs accept `--help`. Reports include the effective configuration,
correctness errors, raw timing samples or process summaries, and available
source/build provenance.

## Typical local runs

Run a warm compact-form benchmark with scalar and SIMD evaluation:

```bash
python3 benchmarks/benchmark_compressed_forms.py \
  --label local-candidate --mode all --construction-path both \
  --cache-state warm --threads 1 \
  --output benchmarks/results/local-compressed-forms.json
```

Measure a representative application separately for each backend:

```bash
python3 benchmarks/benchmark_forms_application.py \
  --label local-scalar --backend scalar --threads 1 \
  --output benchmarks/results/local-application-scalar.json

python3 benchmarks/benchmark_forms_application.py \
  --label local-simd --backend simd --threads 1 \
  --output benchmarks/results/local-application-simd.json
```

Run all compact-aware consumers and direct interpreted evaluation:

```bash
python3 benchmarks/benchmark_form_consumers.py \
  --label local-consumers --group all --backend simd --threads 1 \
  --output benchmarks/results/local-consumers.json

python3 benchmarks/benchmark_form_direct_evaluation.py \
  --label local-direct --points 5000 --threads 1 \
  --output benchmarks/results/local-direct-evaluation.json
```

Memory measurements should run separately in fresh processes so RSS sampling
does not perturb timing runs. For example:

```bash
python3 benchmarks/benchmark_compressed_forms.py \
  --label local-memory --mode construction --retained-graphs 5000 \
  --threads 1 --output benchmarks/results/local-graph-memory.json

python3 benchmarks/benchmark_form_consumers.py \
  --label local-consumer-memory --backend simd --retained-pipelines 100 \
  --threads 1 --output benchmarks/results/local-consumer-memory.json
```

Compare two independently importable package installations directly:

```bash
python3 benchmarks/compare_form_revisions.py \
  --baseline-pythonpath /path/to/main/python \
  --candidate-pythonpath /path/to/candidate/python \
  --processes 5 --mode all --threads 1 \
  --output benchmarks/results/local-main-vs-candidate.json
```

For native compilation, add `--mode compile --compile-kind real` and use
`--realcompile-timeout` to bound every compiler worker. The 4D `(4,4)` stress
case is intentionally not part of ordinary smoke runs.

## Measurement and reporting policy

- Compare identical workloads with identical correctness tolerances.
- Use independent processes and counterbalanced candidate order for performance claims.
- Prefer paired baseline/candidate or compact/dense ratios over absolute timings across sessions.
- Record cold-cache and warmed-cache behavior separately.
- Keep retained-memory runs separate from timing runs.
- Report timeouts, fallbacks, and failed correctness checks instead of discarding them.
- State the hardware, operating system, compiler, NGSolve revision, build type, and thread count with every published comparison.
- Do not commit generated reports merely because they support a favorable result.

The benchmark smoke tests live in `tests/test_*benchmark.py`. Run the normal
test suite after changing a benchmark interface or report schema.
