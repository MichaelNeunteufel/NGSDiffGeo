# Tensor-algebra benchmark reports

These JSON files were recorded on the same Apple arm64 host with NGSolve
`6.2.2606-330-g8cea3e679`. They are evidence, not CI thresholds.

- `tensor_algebra_before.json` records the public implementation before the
  low-rank `Raise` matrix-product specialization.
- `tensor_algebra_after.json` records the specialized implementation and a
  paired, alternating comparison with the exact symbolic-einsum alternative.
- `tensor_algebra_after_confirmation.json` repeats the paired comparison on a
  finer mesh (`maxh=0.05`, 51 assembly samples).
- `tensor_algebra_extended.json` uses the same fine mesh and 51 assembly
  samples and adds rank-three/rank-four tensors, k-forms, and a `(2,2)` double
  form. Every case is checked against an independently constructed expression
  before it is timed.

The confirmation comparison was:

| Raise case | Construction speedup | Assembly speedup |
| --- | ---: | ---: |
| rank one | 5.10× | 1.001× |
| rank two, axis 0 | 5.39× | 1.032× |
| rank two, axis 1 | 5.29× | 1.014× |

Speedups are einsum median divided by matrix-product median. The two variants
had identical values within the benchmark tolerance and equal native graph
node/visit counts; the evaluator node type changed from symbolic einsum to the
specialized symbolic matrix product.

The additional current-implementation timings were:

| Case | Construction median | Assembly median | Nodes / visits |
| --- | ---: | ---: | ---: |
| Raise rank 3, axis 1 | 23.8 us | 313.6 us | 69 / 91 |
| Lower rank 3, axis 2 | 23.2 us | 304.3 us | 68 / 90 |
| Inner product, rank 3 | 66.5 us | 315.5 us | 69 / 202 |
| Trace rank 4, axes 1 and 3 | 33.6 us | 289.7 us | 101 / 139 |
| Wedge of two 1-forms | 40.3 us | 208.1 us | 21 / 39 |
| Hodge star of a 1-form | 30.2 us | 321.7 us | 23 / 24 |
| Inner product of 2-forms | 35.9 us | 250.7 us | 39 / 122 |
| Slot inner product of a `(2,2)` double form | 28.3 us | 228.3 us | 20 / 42 |

These are absolute timings rather than before/after speedups. They establish a
stored baseline for later higher-rank and form optimizations.
