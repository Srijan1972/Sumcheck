"""
Assignment 2 student implementation reference skeleton.

This file documents the frozen student-facing API.
Only 32-bit kernels are compulsory in the base track.
64-bit and 128-bit kernels are intentionally left unimplemented here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# -----------------------------------------------------------------------------
# 32-bit primitives (compulsory)
# -----------------------------------------------------------------------------

def mod_add_32(a, b, q):
    """Return (a + b) mod q for the 32-bit track."""
    # Do arithmetic in 64 bits to avoid overflow, then reduce mod q and cast back
    a64 = jnp.asarray(a, dtype=jnp.uint64)
    b64 = jnp.asarray(b, dtype=jnp.uint64)
    q64 = jnp.asarray(q, dtype	jnp.uint64)
    return jnp.asarray((a64 + b64) % q64, dtype=jnp.uint32)


def mod_sub_32(a, b, q):
    """Return (a - b) mod q for the 32-bit track."""
    a64 = jnp.asarray(a, dtype=jnp.uint64)
    b64 = jnp.asarray(b, dtype=jnp.uint64)
    q64 = jnp.asarray(q, dtype=jnp.uint64)
    # (a - b) mod q = (a + q - b) mod q, computed in 64 bits
    return jnp.asarray((a64 + q64 - b64) % q64, dtype=jnp.uint32)


def mod_mul_32(a, b, q):
    """Return (a * b) mod q for the 32-bit track."""
    a64 = jnp.asarray(a, dtype=jnp.uint64)
    b64 = jnp.asarray(b, dtype	jnp.uint64)
    q64 = jnp.asarray(q, dtype	jnp.uint64)
    return jnp.asarray((a64 * b64) % q64, dtype=jnp.uint32)


# -----------------------------------------------------------------------------
# 64-bit primitives (optional, left for future implementation)
# -----------------------------------------------------------------------------

def mod_add_64(a, b, q):
    """Optional 64-bit modular add kernel."""
    # TODO(student): implement when enabling 64-bit track.
    raise NotImplementedError


def mod_sub_64(a, b, q):
    """Optional 64-bit modular subtract kernel."""
    # TODO(student): implement when enabling 64-bit track.
    raise NotImplementedError


def mod_mul_64(a, b, q):
    """Optional 64-bit modular multiply kernel."""
    # TODO(student): implement when enabling 64-bit track.
    raise NotImplementedError


# -----------------------------------------------------------------------------
# 128-bit primitives (optional, left for future implementation)
# -----------------------------------------------------------------------------

def mod_add_128(a, b, q):
    """Optional 128-bit modular add kernel."""
    # TODO(student): implement when enabling 128-bit track.
    raise NotImplementedError


def mod_sub_128(a, b, q):
    """Optional 128-bit modular subtract kernel."""
    # TODO(student): implement when enabling 128-bit track.
    raise NotImplementedError


def mod_mul_128(a, b, q):
    """Optional 128-bit modular multiply kernel."""
    # TODO(student): implement when enabling 128-bit track.
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


# -----------------------------------------------------------------------------
# MLE updates
# -----------------------------------------------------------------------------

def mle_update_32(zero_eval, one_eval, target_eval, *, q):
    """Compulsory 32-bit MLE update.

    Multilinear extension at arbitrary field point t:
        f(t) = zero_eval + t * (one_eval - zero_eval)  (mod q)

    Works element-wise on scalars or JAX arrays.
    """
    # diff = (one_eval - zero_eval) mod q
    diff = mod_sub_32(one_eval, zero_eval, q)
    # t_diff = t * diff mod q
    t_diff = mod_mul_32(target_eval, diff, q)
    # result = zero_eval + t_diff mod q
    return mod_add_32(zero_eval, t_diff, q)


def mle_update_64(zero_eval, one_eval, target_eval, *, q):
    """Optional 64-bit MLE update."""
    # TODO(student): implement when enabling 64-bit track.
    raise NotImplementedError


def mle_update_128(zero_eval, one_eval, target_eval, *, q):
    """Optional 128-bit MLE update."""
    # TODO(student): implement when enabling 128-bit track.
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
# Sumcheck protocol implementations
# -----------------------------------------------------------------------------

def sumcheck_32(eval_tables, *, q, expression, challenges, num_rounds):
    """Compulsory 32-bit sumcheck path.

    Parameters
    ----------
    eval_tables : dict[str, jax.Array]
        Flat evaluation tables of shape (2**num_rounds,). x1 is LSB.
    q           : JAX uint32 scalar — prime modulus.
    expression  : list[list[str]] — e.g. [["a","b"],["c"]] means a*b + c.
    challenges  : JAX array, length num_rounds — per-round field elements.
                  Does NOT include the final verifier challenge.
    num_rounds  : int — number of Boolean variables n.

    Returns
    -------
    claim0      : JAX scalar — Σ_{x∈{0,1}^n} f(x)  mod q
    round_evals : JAX array shape (num_rounds, degree+1)
                  row i = [g_i(0), g_i(1), ..., g_i(d)]
    """
    # 1) Infer polynomial degree from expression
    # Degree = length of the longest multiplicative term.
    # [["a"]] → 1,  [["a","b"],["c"]] → 2,  [["a","b","c"]] → 3
    degree = max(len(term) for term in expression)
    num_t_points = degree + 1   # evaluate at t = 0, 1, ..., degree

    # Working copy of tables (values stay as uint32 JAX arrays)
    tables = {
        name: jnp.asarray(arr, dtype=jnp.uint32)
        for name, arr in eval_tables.items()
    }

    # 2) claim0: sum f(x) over all 2^n Boolean inputs
    some_name = next(iter(tables.keys()))
    N_full = tables[some_name].shape[0]

    # Evaluate composite polynomial f pointwise on the Boolean hypercube
    flat_f = jnp.zeros(N_full, dtype=jnp.uint32)
    for term in expression:                            # additive terms
        product = jnp.ones(N_full, dtype=jnp.uint32)
        for name in term:                              # multiplicative factors
            product = mod_mul_32(product, tables[name], q)
        flat_f = mod_add_32(flat_f, product, q)

    # Reduce-sum flat_f mod q using modular add (tree fold)
    tmp = flat_f
    while tmp.shape[0] > 1:
        h = tmp.shape[0] // 2
        tmp = mod_add_32(tmp[:h], tmp[h:h * 2], q)
    claim0 = tmp[0]   # scalar

    # 3) Per-round prover loop
    all_round_evals = []

    for rnd in range(num_rounds):
        some_name = next(iter(tables.keys()))
        N = tables[some_name].shape[0]
        half = N // 2

        # Reshape (N,) → (half, 2):
        #   column 0 = x_i = 0 (even indices)
        #   column 1 = x_i = 1 (odd indices)
        paired = {name: tables[name].reshape(half, 2) for name in tables}

        # Compute g_rnd(t) for each evaluation point t
        g_vals = []
        for t in range(num_t_points):

            # For every base polynomial, produce a length-half array of
            # values at (row, t), vectorized across all rows.
            base_vecs = {}
            for name, col_pair in paired.items():
                z = col_pair[:, 0]   # (half,) — evaluations at x_i = 0
                o = col_pair[:, 1]   # (half,) — evaluations at x_i = 1
                if t == 0:
                    base_vecs[name] = z
                elif t == 1:
                    base_vecs[name] = o
                else:
                    # Evaluate at t ≥ 2 using multilinear interpolation
                    t_arr = jnp.asarray(t, dtype=jnp.uint32)
                    base_vecs[name] = mle_update_32(z, o, t_arr, q=q)

            # Evaluate composite expression for every row at this t
            f_at_t = jnp.zeros(half, dtype=jnp.uint32)
            for term in expression:
                product = jnp.ones(half, dtype	jnp.uint32)
                for name in term:
                    product = mod_mul_32(product, base_vecs[name], q)
                f_at_t = mod_add_32(f_at_t, product, q)

            # Sum f_at_t over all rows → g_rnd(t) scalar
            tmp = f_at_t
            while tmp.shape[0] > 1:
                h = tmp.shape[0] // 2
                tmp = mod_add_32(tmp[:h], tmp[h:h * 2], q)
            g_vals.append(tmp[0])

        all_round_evals.append(g_vals)

        # Fold tables with challenge r_i to eliminate one Boolean variable
        r_i = jnp.asarray(challenges[rnd], dtype=jnp.uint32)
        new_tables = {}
        for name in tables:
            z_col = paired[name][:, 0]   # (half,)
            o_col = paired[name][:, 1]   # (half,)
            new_tables[name] = mle_update_32(z_col, o_col, r_i, q=q)
        tables = new_tables

    # 4) Pack round_evals into a single 2D JAX array: (num_rounds, degree+1)
    round_evals = jnp.array(
        [[int(v) for v in row] for row in all_round_evals],
        dtype=jnp.uint32,
    )

    return claim0, round_evals


def sumcheck_64(eval_tables, *, q, expression, challenges, num_rounds):
    """Optional 64-bit sumcheck path."""
    # TODO(student): implement when enabling 64-bit track.
    raise NotImplementedError


def sumcheck_128(eval_tables, *, q, expression, challenges, num_rounds):
    """Optional 128-bit sumcheck path."""
    # TODO(student): implement when enabling 128-bit track.
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
