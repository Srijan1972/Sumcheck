"""
Assignment 2 student implementation.

Optimized 32-bit prover for the SumCheck protocol on the Boolean hypercube.

Public API (frozen by the harness):
    mod_add_32 / mod_sub_32 / mod_mul_32   -- 32-bit modular primitives
    mle_update_32                          -- single-step MLE update
    sumcheck_32(...)                       -- 32-bit prover
    mod_add / mod_sub / mod_mul / mle_update / sumcheck   -- bit-width dispatchers

Optimizations vs. the naive baseline:
  * Per-round inner work is fully vectorized:
      - For each variable name we compute its evaluations at t = 0, 1, ..., d
        using a (d+1, half)-shaped tensor.  Composition for *all* t-values
        runs in a single fused chain of jnp ops.
  * Modular extrapolation along t uses an additive trick.  Since
        mle_update(z, o, t) = z + t * (o - z),
    the values for t = 2, 3, ..., d differ from the previous one by exactly
    `diff = o - z`.  We compute t = 0 (= z), t = 1 (= o), and then add `diff`
    repeatedly -- replacing (d-1) modular multiplies per row with cheap adds.
  * Per-row sums collapsed via `jnp.sum(.. .astype(uint64)) % q`, instead of
    a Python `while` loop of pairwise mod_adds.  Safe because every entry is
    < q < 2^32 and the maximum row count we ever sum over is 2^(num_vars-1)
    so the unreduced sum stays well below 2^64 for num_vars <= 31.
  * `claim0` is derived from g_1(0) + g_1(1) instead of a separate full-table
    pass over f.  Halves the work in round 1 effectively.
  * The `diff` tensor we computed for evaluating g(t) is reused when folding
    the table by the round challenge r_i (the fold is just z + r_i * diff).
  * The fold after the final round is skipped (the baseline relied on JAX's
    lenient OOB indexing of `challenges`).

64-bit and 128-bit kernels remain unimplemented (extra-credit tracks).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# -----------------------------------------------------------------------------
# 32-bit primitives (compulsory)
# -----------------------------------------------------------------------------
@jax.jit
def mod_add_32(a, b, q):
    """Return (a + b) mod q for the 32-bit track."""
    a64 = a.astype(jnp.uint64)
    b64 = b.astype(jnp.uint64)
    res = a64 + b64
    return jnp.where(res >= q, res - q, res).astype(jnp.uint32)


@jax.jit
def mod_sub_32(a, b, q):
    """Return (a - b) mod q for the 32-bit track."""
    a64 = a.astype(jnp.int64)
    b64 = b.astype(jnp.int64)
    res = a64 - b64
    return jnp.where(res < 0, res + q, res).astype(jnp.uint32)


@jax.jit
def mod_mul_32(a, b, q):
    """Return (a * b) mod q for the 32-bit track."""
    a64 = a.astype(jnp.uint64)
    b64 = b.astype(jnp.uint64)
    return ((a64 * b64) % q).astype(jnp.uint32)


# -----------------------------------------------------------------------------
# 64-bit primitives (optional, left for future implementation)
# -----------------------------------------------------------------------------
def mod_add_64(a, b, q):
    """Optional 64-bit modular add kernel."""
    raise NotImplementedError


def mod_sub_64(a, b, q):
    """Optional 64-bit modular subtract kernel."""
    raise NotImplementedError


def mod_mul_64(a, b, q):
    """Optional 64-bit modular multiply kernel."""
    raise NotImplementedError


# -----------------------------------------------------------------------------
# 128-bit primitives (optional, left for future implementation)
# -----------------------------------------------------------------------------
def mod_add_128(a, b, q):
    """Optional 128-bit modular add kernel."""
    raise NotImplementedError


def mod_sub_128(a, b, q):
    """Optional 128-bit modular subtract kernel."""
    raise NotImplementedError


def mod_mul_128(a, b, q):
    """Optional 128-bit modular multiply kernel."""
    raise NotImplementedError


# -----------------------------------------------------------------------------
# Frozen dispatch API
# -----------------------------------------------------------------------------
def mod_add(a, b, q, *, bit_width=32):
    if int(bit_width) == 32:
        return mod_add_32(a, b, q)
    if int(bit_width) == 64:
        return mod_add_64(a, b, q)
    if int(bit_width) == 128:
        return mod_add_128(a, b, q)
    raise ValueError(f"Unsupported bit_width={bit_width}")


def mod_sub(a, b, q, *, bit_width=32):
    if int(bit_width) == 32:
        return mod_sub_32(a, b, q)
    if int(bit_width) == 64:
        return mod_sub_64(a, b, q)
    if int(bit_width) == 128:
        return mod_sub_128(a, b, q)
    raise ValueError(f"Unsupported bit_width={bit_width}")


def mod_mul(a, b, q, *, bit_width=32):
    if int(bit_width) == 32:
        return mod_mul_32(a, b, q)
    if int(bit_width) == 64:
        return mod_mul_64(a, b, q)
    if int(bit_width) == 128:
        return mod_mul_128(a, b, q)
    raise ValueError(f"Unsupported bit_width={bit_width}")


def mle_update_32(zero_eval, one_eval, target_eval, *, q):
    """Compulsory 32-bit MLE update: returns (one - zero) * target + zero  mod q."""
    diff = mod_sub_32(one_eval, zero_eval, q)
    t_diff = mod_mul_32(target_eval, diff, q)
    return mod_add_32(zero_eval, t_diff, q)


def mle_update_64(zero_eval, one_eval, target_eval, *, q):
    """Optional 64-bit MLE update."""
    raise NotImplementedError


def mle_update_128(zero_eval, one_eval, target_eval, *, q):
    """Optional 128-bit MLE update."""
    raise NotImplementedError


def mle_update(zero_eval, one_eval, target_eval, *, q, bit_width=32):
    if int(bit_width) == 32:
        return mle_update_32(zero_eval, one_eval, target_eval, q=q)
    if int(bit_width) == 64:
        return mle_update_64(zero_eval, one_eval, target_eval, q=q)
    if int(bit_width) == 128:
        return mle_update_128(zero_eval, one_eval, target_eval, q=q)
    raise ValueError(f"Unsupported bit_width={bit_width}")


# -----------------------------------------------------------------------------
# 32-bit sumcheck prover (compulsory)
# -----------------------------------------------------------------------------
def _row_sum_mod_q_32(arr, q, axis):
    """Sum a uint32 array along `axis` and reduce mod q.

    Casts to uint64 first so that summing up to ~2^31 entries each < 2^32 stays
    safely inside 2^64.  Final result is cast back to uint32.
    """
    return (arr.astype(jnp.uint64).sum(axis=axis) % q).astype(jnp.uint32)


def _base_evals_at_all_t_32(z, o, q, degree):
    """Return (base_at_t, diff) where:

        base_at_t : uint32 array of shape (degree + 1, *z.shape)
                    base_at_t[t] = mle_update(z, o, t) = z + t * (o - z)  mod q
        diff      : uint32 array of shape z.shape, diff = (o - z) mod q.

    For t = 2, 3, ..., degree we use the additive identity
        mle_update(z, o, t) = mle_update(z, o, t - 1) + diff   (mod q),
    which replaces a modular multiply per row with a single modular add.
    """
    diff = mod_sub_32(o, z, q)
    rows = [z, o]
    cur = o
    # degree - 1 extra rows to cover t = 2 .. degree
    for _ in range(degree - 1):
        cur = mod_add_32(cur, diff, q)
        rows.append(cur)
    base_at_t = jnp.stack(rows, axis=0)
    return base_at_t, diff


def sumcheck_32(eval_tables, *, q, expression, challenges, num_rounds):
    """Compulsory 32-bit sumcheck path.

    Parameters
    ----------
    eval_tables : dict[str, jax.Array]
        Flat evaluation tables of shape (2**num_rounds,). x_1 is the LSB.
    q           : JAX uint32 scalar -- prime modulus.
    expression  : list[list[str]] -- e.g. [["a","b"], ["c"]] means a*b + c.
    challenges  : JAX array of length (num_rounds - 1) -- per-round prover
                  challenges; the verifier-only final challenge is excluded.
    num_rounds  : int -- number of Boolean variables n.

    Returns
    -------
    claim0      : JAX uint32 scalar -- sum_{x in {0,1}^n} f(x) mod q.
    round_evals : JAX uint32 array of shape (num_rounds, degree + 1).
                  Row i = [g_i(0), g_i(1), ..., g_i(d)].
    """
    degree = max(len(term) for term in expression)
    num_t_points = degree + 1

    # Working copy of tables (kept as uint32).  Use of dict is fine; iteration
    # order is preserved (Python 3.7+).
    tables = {
        name: jnp.asarray(arr, dtype=jnp.uint32)
        for name, arr in eval_tables.items()
    }

    all_round_evals = []

    # Precompute the unique variable names per term as Python tuples so the
    # JIT trace doesn't have to re-resolve dict keys each iteration.
    expr_terms = [tuple(term) for term in expression]

    for rnd in range(num_rounds):
        any_name = next(iter(tables.keys()))
        N = tables[any_name].shape[0]
        half = N // 2

        # Even / odd split:
        #   z  = entries with x_i = 0  (even indices)
        #   o  = entries with x_i = 1  (odd indices)
        z_o = {
            name: (tables[name][0::2], tables[name][1::2])
            for name in tables
        }

        # For each variable: base[name] of shape (d+1, half), diff[name] of
        # shape (half,).
        base_for_name = {}
        diff_for_name = {}
        for name, (z, o) in z_o.items():
            base, diff = _base_evals_at_all_t_32(z, o, q, degree)
            base_for_name[name] = base
            diff_for_name[name] = diff

        # Vectorized composition: f at every (t, row).
        # acc has shape (d+1, half).  We accumulate additive terms.
        acc = jnp.zeros((num_t_points, half), dtype=jnp.uint32)
        for term in expr_terms:
            product = base_for_name[term[0]]
            for name in term[1:]:
                product = mod_mul_32(product, base_for_name[name], q)
            acc = mod_add_32(acc, product, q)

        # Sum across rows -> g_round of shape (d+1,).  This is a single fused
        # reduction and replaces the manual tree-fold of mod_adds.
        g_round = _row_sum_mod_q_32(acc, q, axis=1)
        all_round_evals.append(g_round)

        # Fold tables for the next round (skip if this was the last round).
        if rnd + 1 < num_rounds:
            r_i = jnp.asarray(challenges[rnd], dtype=jnp.uint32)
            new_tables = {}
            for name, (z, _o) in z_o.items():
                # mle_update(z, o, r_i) = z + r_i * diff.  diff is reused.
                r_diff = mod_mul_32(r_i, diff_for_name[name], q)
                new_tables[name] = mod_add_32(z, r_diff, q)
            tables = new_tables

    round_evals = jnp.stack(all_round_evals, axis=0)  # (num_rounds, degree+1)

    # claim0 = g_1(0) + g_1(1)  mod q  (verifier consistency for round 1).
    # Avoids a separate full-hypercube pass to compute the sum of f.
    claim0 = mod_add_32(round_evals[0, 0], round_evals[0, 1], q)

    return claim0, round_evals


# -----------------------------------------------------------------------------
# 64-bit / 128-bit sumcheck (optional, left for future implementation)
# -----------------------------------------------------------------------------
def sumcheck_64(eval_tables, *, q, expression, challenges, num_rounds):
    """Optional 64-bit sumcheck path."""
    raise NotImplementedError


def sumcheck_128(eval_tables, *, q, expression, challenges, num_rounds):
    """Optional 128-bit sumcheck path."""
    raise NotImplementedError


def sumcheck(eval_tables, *, q, expression, challenges, num_rounds, bit_width=32):
    """Frozen dispatcher entrypoint used by the harness."""
    if int(bit_width) == 32:
        return sumcheck_32(
            eval_tables,
            q=q,
            expression=expression,
            challenges=challenges,
            num_rounds=num_rounds,
        )
    if int(bit_width) == 64:
        return sumcheck_64(
            eval_tables,
            q=q,
            expression=expression,
            challenges=challenges,
            num_rounds=num_rounds,
        )
    if int(bit_width) == 128:
        return sumcheck_128(
            eval_tables,
            q=q,
            expression=expression,
            challenges=challenges,
            num_rounds=num_rounds,
        )
    raise ValueError(f"Unsupported bit_width={bit_width}")
