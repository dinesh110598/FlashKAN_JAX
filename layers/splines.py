import jax
import jax.numpy as jnp
import numpy as np

def b_spline(x: jax.Array, i: jax.Array, t: jax.Array, k: int):
    
    x = jnp.expand_dims(x, -1) # (in_dim, 1)
    
    t2 = t[i] # (in_dim, 2k)
    bases = jnp.logical_and((t2[:, :-1] <= x), (x < t2[:, 1:]))
    
    for j in range(1,k):
        bases = jnp.nan_to_num(
            (x - t2[:, :-(j+1)])
            / (t2[:, j:-1] - t2[:, :-(j+1)])
            * bases[:, :-1], False, 0, 0, 0
        ) + jnp.nan_to_num(
            (t2[:, j+1:] - x)
            / (t2[:, j+1:] - t2[:, 1:-j])
            * bases[:, 1:], False, 0, 0, 0
        )
    
    return bases # (in_dim, k)

def b_spline_diff(x, i, t, k):
    return jnp.nan_to_num(
        b_spline(x, i[:, :-1], t, k-1)/
        (t[i[:, k-1:-1]] - t[i[:, :k]]), False, 0, 0, 0
    ) + jnp.nan_to_num(
        b_spline(x, i[:, 1:], t, k-1)/
        (t[i[:, k:]] - t[i[:, 1:k+1]]), False, 0, 0, 0
    )

def _evaluate_idx(k: int, x: jax.Array, t0: jax.Array, 
                  dt: jax.Array, t_end:jax.Array):
    f = lambda x: (x - t0)/dt + (k-1)
    idx = jnp.maximum(jnp.minimum(f(x), f(t_end-dt)), f(t0))
    return jnp.int32(jnp.floor(idx))

def evaluate_all(k: int, t: jax.Array, x: jax.Array):
    i_max = _evaluate_idx(k, x, t[:1], t[k:k+1]-t[k-1:k], t[-1:])
    i_full = jnp.stack([i_max+j for j in range(1-k, k+1)], -1)
    y = b_spline(x, i_full, t, k)
    
    return i_full, y
    