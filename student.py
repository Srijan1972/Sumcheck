"""
Assignment 2 student implementation.

Only 32-bit kernels are compulsory; 64/128-bit kernels are intentionally
left unimplemented in the base track.
"""

from __future__ import annotations

from functools import partial

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
    s = a64 + b64
    return jnp.where(s >= q, s - q, s).astype(jnp.uint32)


@jax.jit
def mod_sub_32(a, b, q):
    """Return (a - b) mod q for the 32-bit track."""
    a_i = a.astype(jnp.int64)
    b_i = b.astype(jnp.int64)
    d = a_i - b_i
    return jnp.where(d < 0, d + q.astype(jnp.int64), d).astype(jnp.uint32)


@jax.jit
def mod_mul_32(a, b, q):
    """Return (a * b) mod q for the 32-bit track."""
    a64 = a.astype(jnp.uint64)
    b64 = b.astype(jnp.uint64)
    return ((a64 * b64) % q).astype(jnp.uint32)


# -----------------------------------------------------------------------------
# 64-bit primitives (optional)
# -----------------------------------------------------------------------------

def mod_add_64(a, b, q):
    raise NotImplementedError


def mod_sub_64(a, b, q):
    raise NotImplementedError


def mod_mul_64(a, b, q):
    raise NotImplementedError


# -----------------------------------------------------------------------------
# 128-bit primitives (optional)
# -----------------------------------------------------------------------------

def mod_add_128(a, b, q):
    raise NotImplementedError


def mod_sub_128(a, b, q):
    raise NotImplementedError


def mod_mul_128(a, b, q):
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
# MLE update: (o - z) * t + z mod q
# -----------------------------------------------------------------------------
@jax.jit
def mle_update_32(zero_eval, one_eval, target_eval, *, q):
    """Compulsory 32-bit MLE update."""
    diff = mod_sub_32(one_eval, zero_eval, q)
    prod = mod_mul_32(target_eval, diff, q)
    return mod_add_32(zero_eval, prod, q)


def mle_update_64(zero_eval, one_eval, target_eval, *, q):
    raise NotImplementedError


def mle_update_128(zero_eval, one_eval, target_eval, *, q):
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
# 32-bit sumcheck
# -----------------------------------------------------------------------------
@partial(jax.jit, static_argnames=("expr_idx",))
def _claim0_32(tables, expr_idx, q):
    N = tables[0].shape[0]
    flat = jnp.zeros(N, dtype=jnp.uint32)
    for term in expr_idx:
        prod = jnp.ones(N, dtype=jnp.uint32)
        for vi in term:
            prod = mod_mul_32(prod, tables[vi], q)
        flat = mod_add_32(flat, prod, q)
    s64 = flat.astype(jnp.uint64).sum()
    return (s64 % q.astype(jnp.uint64)).astype(jnp.uint32)


@partial(jax.jit, static_argnames=("expr_idx", "num_t_points"))
def _round_step_32(tables, expr_idx, num_t_points, r_i, q):
    half = tables[0].shape[0] // 2
    paired = tuple(t.reshape(half, 2) for t in tables)
    z = tuple(p[:, 0] for p in paired)
    o = tuple(p[:, 1] for p in paired)

    g_vals = []
    for t in range(num_t_points):
        if t == 0:
            base = z
        elif t == 1:
            base = o
        else:
            t_arr = jnp.uint32(t)
            base = tuple(
                mle_update_32(zi, oi, t_arr, q=q) for zi, oi in zip(z, o)
            )

        f_at_t = jnp.zeros(half, dtype=jnp.uint32)
        for term in expr_idx:
            prod = jnp.ones(half, dtype=jnp.uint32)
            for vi in term:
                prod = mod_mul_32(prod, base[vi], q)
            f_at_t = mod_add_32(f_at_t, prod, q)

        s64 = f_at_t.astype(jnp.uint64).sum()
        g_vals.append((s64 % q.astype(jnp.uint64)).astype(jnp.uint32))

    g_arr = jnp.stack(g_vals)
    new_tables = tuple(mle_update_32(zi, oi, r_i, q=q) for zi, oi in zip(z, o))
    return g_arr, new_tables


def sumcheck_32(eval_tables, *, q, expression, challenges, num_rounds):
    """Compulsory 32-bit sumcheck.

    Returns
    -------
    claim0      : JAX uint32 scalar — Σ_{x∈{0,1}^n} f(x) mod q
    round_evals : JAX uint32 array shape (num_rounds, degree+1)
    """
    var_seen = []
    for term in expression:
        for v in term:
            if v not in var_seen:
                var_seen.append(v)
    var_idx = {n: i for i, n in enumerate(var_seen)}
    expr_idx = tuple(tuple(var_idx[v] for v in term) for term in expression)
    degree = max(len(term) for term in expression)
    num_t_points = degree + 1

    tables = tuple(
        jnp.asarray(eval_tables[name], dtype=jnp.uint32) for name in var_seen
    )
    q32 = jnp.asarray(q, dtype=jnp.uint32)

    claim0 = _claim0_32(tables, expr_idx, q32)

    round_evals_list = []
    for rnd in range(int(num_rounds)):
        r_i = jnp.asarray(challenges[rnd], dtype=jnp.uint32)
        g_arr, tables = _round_step_32(tables, expr_idx, num_t_points, r_i, q32)
        round_evals_list.append(g_arr)

    round_evals = jnp.stack(round_evals_list)
    return claim0, round_evals


def sumcheck_64(eval_tables, *, q, expression, challenges, num_rounds):
    raise NotImplementedError


def sumcheck_128(eval_tables, *, q, expression, challenges, num_rounds):
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
