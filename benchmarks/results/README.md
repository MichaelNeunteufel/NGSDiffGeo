# Tensor-algebra benchmark reports

These JSON files were recorded on the same Apple arm64 host with NGSolve
`6.2.2606-330-g8cea3e679`. They are evidence, not CI thresholds.

- `tensor_algebra_before.json` records the public implementation before the
  low-rank `Raise` matrix-product specialization.
- `tensor_algebra_after.json` records the specialized implementation and a
  paired, alternating comparison with the exact symbolic-einsum alternative.
- `tensor_algebra_after_confirmation.json` repeats the paired comparison on a
  finer mesh (`maxh=0.05`, 51 assembly samples).

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
