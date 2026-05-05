# Sumcheck Assignment — Implementation Notes

This document summarises what was done to `student.py` for the Assignment 2
SumCheck-in-JAX prover, why each change was made, and the measured impact.

## 1. Starting point

The repository already contained a working baseline 32-bit prover in
`student.py`. It produced correct outputs (all `vars4` / `vars16` / `vars20`
base-polynomial cases passed before any edits), but it was structured naively:

- Per round it looped in Python over each evaluation point `t = 0, 1, …, d`,
  computing one length-`half` array per `t` and reducing it independently.
- Each reduction was a hand-rolled `while` tree-fold of pairwise `mod_add_32`
  calls instead of a single fused `jnp.sum`.
- For `t ≥ 2`, the base values were obtained by calling `mle_update_32`, which
  performs a modular multiplication per row.
- `claim0` was computed by a separate full pass over the hypercube (building
  `flat_f` of shape `(2**num_vars,)` and tree-folding it), even though
  `g_1(0) + g_1(1)` is exactly that sum.
- The fold after the final round used `challenges[num_rounds - 1]`, which is
  out of bounds (the harness passes `len(challenges) == num_rounds - 1`); JAX's
  lenient OOB indexing happened to make the wasted work harmless.

The 64-bit and 128-bit primitive / sumcheck stubs were left as
`NotImplementedError` (extra-credit tracks).

## 2. Verification of the baseline

Before touching anything, the baseline was run end-to-end to establish a
correctness reference and capture a "before" benchmark:

```
python -m pytest --bits 32 --num-vars 4    # 20/20 sumcheck cases pass
python -m pytest --bits 32 --num-vars 16   # 20/20 sumcheck cases pass
```

Plus the 31 modular-primitive edge-case tests on 32-bit. All pass.

## 3. Optimisations applied

The 32-bit primitives (`mod_add_32`, `mod_sub_32`, `mod_mul_32`),
`mle_update_32`, and the dispatcher API (`mod_add`, `mod_sub`, `mod_mul`,
`mle_update`, `sumcheck`) are unchanged in shape and are still
`@jax.jit`-decorated where they were before. The work is in `sumcheck_32`.

### 3.1 Vectorise across all `t` values at once

The new per-round inner loop builds, for each variable name, a single tensor
of shape `(d + 1, half)` containing its evaluations at every `t = 0..d`. The
composition of additive / multiplicative terms then runs once over that whole
tensor instead of `d + 1` times in a Python loop:

```python
acc = jnp.zeros((num_t_points, half), dtype=jnp.uint32)
for term in expr_terms:
    product = base_for_name[term[0]]
    for name in term[1:]:
        product = mod_mul_32(product, base_for_name[name], q)
    acc = mod_add_32(acc, product, q)
```

This lets XLA fuse the per-`t` work into a single kernel and removes
redundant trace-time bookkeeping.

### 3.2 Additive trick for evaluating at `t ≥ 2`

The MLE-update formula is `mle_update(z, o, t) = z + t · (o − z)`. For
consecutive integer `t`s this is an arithmetic progression, so

```
mle_update(z, o, t) = mle_update(z, o, t − 1) + (o − z)   (mod q).
```

`_base_evals_at_all_t_32` computes `diff = o − z` once and then accumulates
`diff` to walk forward through `t = 2, 3, …, d`, replacing one modular
multiplication per row per `t` with one modular addition. This is the
optimisation hinted at in the bottom of `sumcheck_intro.md`.

### 3.3 Single fused reduction

The manual tree-fold of `mod_add_32`s was replaced with

```python
def _row_sum_mod_q_32(arr, q, axis):
    return (arr.astype(jnp.uint64).sum(axis=axis) % q).astype(jnp.uint32)
```

This is safe for the assignment's largest case (`num_vars = 20`): the worst
row count we sum over is `2**(20-1) = 524 288`, each entry is `< q < 2^32`, so
the unreduced uint64 sum stays well below `2^64`. One `jnp.sum` lowers to a
much tighter XLA reduction than a 19-deep cascade of pairwise mod-adds.

### 3.4 Reuse `diff` in the fold step

The challenge-fold for the next round is also an MLE update:
`new = z + r_i · diff`. Because the round just computed `diff` per name, it is
reused directly instead of being recomputed inside `mle_update_32`.

### 3.5 Derive `claim0` from round 1

`claim0 = Σ_{x ∈ {0,1}^n} f(x)` is exactly the verifier consistency value for
round 1, namely `g_1(0) + g_1(1)`. The new implementation simply reads
`round_evals[0, 0] + round_evals[0, 1]` (mod `q`) and skips the original
separate pass over the full `2**n` hypercube. That pass was the largest single
piece of work in the old code.

### 3.6 Skip the wasted post-final-round fold

After the last round's `g_n(t)` values are computed there is no next round to
feed, so the table fold is skipped. This also eliminates the OOB read of
`challenges[num_rounds - 1]`.

## 4. Correctness re-verification

After the rewrite, the consolidated correctness command from the README was
run:

```
python -m pytest --all-32 --num-vars 4 --num-vars 16 --num-vars 20
```

Result: **60 / 60 sumcheck cases pass** (5 cases × 4 base expressions ×
3 variable counts) and **31 / 31 primitive edge-case tests pass**. No
behavioural regressions vs. the baseline.

## 5. Benchmark results (CPU, sandbox)

Measured with the same JIT wrapper the harness uses (3 warmups, 5 timed runs,
median reported), `q` and `expression` captured as static at trace time.

### vars16 base polynomials

| expression | baseline median | optimised median | speedup |
|------------|----------------:|-----------------:|--------:|
| `a`        | 0.68 ms         | 0.20 ms          | 3.4×    |
| `a*b`      | 2.12 ms         | 0.32 ms          | 6.6×    |
| `a*b + c`  | 3.43 ms         | 0.39 ms          | 8.8×    |
| `a*b*c`    | 4.28 ms         | 0.45 ms          | 9.5×    |

### vars20 base polynomials (optimised only)

| expression | median   | p90      |
|------------|---------:|---------:|
| `a`        | 1.03 ms  | 1.10 ms  |
| `a*b`      | 2.52 ms  | 2.56 ms  |
| `a*b + c`  | 3.08 ms  | 3.21 ms  |
| `a*b*c`    | 7.81 ms  | 8.04 ms  |

Compile time also dropped roughly 4–7× across the board (e.g. for `a*b*c` on
vars16: ~3.6 s → ~0.5 s), which matters because the harness JIT-traces the
prover once per `(case, expression)` pair.

These numbers are CPU only (the sandbox has no GPU). The same code should
behave well on a GPU/TPU because the per-round work is now a small number of
contiguous fused kernels.

## 6. What is _not_ implemented (extra credit, left as `NotImplementedError`)

- 64-bit primitives, `mle_update_64`, `sumcheck_64`.
- 128-bit primitives, `mle_update_128`, `sumcheck_128`.
- Hashing between SumCheck rounds.
- Advanced polynomials behind `--enable-challenge32` (`a*a*b*b*c`,
  `a*b*c + d*e`, `a*b*c*g + d*e*g`). The current `sumcheck_32` already
  handles arbitrary degree, so enabling these is just a matter of running the
  extra tracks; no further code change is required for correctness, though a
  few more targeted optimisations would help at degree 5.
- TPU / GPU experiments and report write-up.

## 7. Files touched

- `student.py` — rewritten implementation of `sumcheck_32` plus expanded
  module docstring; primitive kernels and dispatch API unchanged.

No other source files were modified.
