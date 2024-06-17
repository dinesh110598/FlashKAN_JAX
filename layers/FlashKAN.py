# %%
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from jax import nn, lax, custom_vjp, vmap, grad
from jax import random as jrandom
from functools import partial
import jaxlib

from .splines import evaluate_all, b_spline_diff
# %%
class FlashKAN(eqx.Module):
    order: int
    w: jax.Array
    knots: jax.Array
    act: jaxlib.xla_extension.PjitFunction
    
    def __init__(self, in_dim, out_dim, G,
                 t0=-1., t1=1., k=4, act=nn.silu, *, key):
        super().__init__()
        self.order = k
        
        self.w = nn.initializers.xavier_uniform([0,1], 2)(
            key, [G+k, in_dim, out_dim])
        
        t = jnp.linspace(t0, t1, G+1)
        self.knots = jnp.concat([jnp.full([k-1], t0), t,
                             jnp.full([k-1], t1)])
        self.act = act
        
    def __call__(self, x, key=None):
        return flash_kan(x, self.w, self.knots, self.order, self.act)

@partial(custom_vjp, nondiff_argnums=(3,4))
def flash_kan(x, w, t, k, act):
    in_dim = x.shape[0]
    
    i_full, y1 = evaluate_all(k, t, x)
    # (in_dim, 2k), (in_dim, k)
    
    slice1 = jnp.concat([i_full[:, :k],
                        jnp.full_like(i_full[:, :1], -1)], axis=1) #(in_dim, k+1)
    slice2 = jnp.expand_dims(jnp.arange(in_dim), 1) #(in_dim, 1)
    
    w2 = w[slice1, slice2, :] #(in_dim, k+1, out_dim)
    
    y2 = jnp.concat([y1, jnp.expand_dims(act(x), 1)],
                    axis=1)
    y2 = jnp.expand_dims(y2, 2) #(in_dim, k+1, 1)
    
    return jnp.sum(w2 * y2, axis=(0,1))

def flash_kan_fwd(x, w, t, k, act):
    # Make sure that all arrays below are statically shaped
    i_full, y1 = evaluate_all(k, t, x)
    # (in_dim, 2k), (in_dim, k)
    st_pos = i_full[:, 0]
    
    w1 = vmap(lax.dynamic_slice_in_dim, (1, 0, None, None))(
        w, st_pos, k, 0) #(in, k, out)
    w2 = jnp.transpose(w[-1:], (1, 0, 2)) #(in, 1, out)
    w0 = jnp.concat([w1, w2], axis=1) #(in, k+1, out)
    
    y1 = jnp.expand_dims(y1, 2) #(in, k, 1)
    y2 = jnp.expand_dims(act(x), (1,2)) #(in, 1, 1)
    y = jnp.concat([y1, y2], axis=1) #(in, k+1, 1)
    
    return jnp.sum(w0 * y, axis=(0,1)), (x, w, t, i_full, w0, y1, y2)
    
def flash_kan_bwd(k, act, res, d_out):
    x, w, t, i_full, w0, y1, y2 = res
    
    st_pos = i_full[:, 0]
    Dw = jnp.zeros_like(w)
    d_out = jnp.expand_dims(d_out, (0,1)) #(1, 1, out)
    Dw = vmap(lax.dynamic_update_slice_in_dim, (1, 0, 0, None), 1)(
        Dw, y1*d_out, st_pos, 0)
    Dw = Dw.at[-1].set(y2[:, 0] * d_out[0])
    
    Dy1 = b_spline_diff(x, i_full, t, k) # (in, k)
    Dy2 = jnp.expand_dims(vmap(grad(act))(x), 1)
    Dy = jnp.expand_dims(jnp.concatenate([Dy1, Dy2], 1), 2) # (in, k_1, 1)
    
    Dx = jnp.sum(d_out * w0 * Dy, axis=(1,2)) #(in,)
    
    return (Dx, Dw, None)

flash_kan.defvjp(flash_kan_fwd, flash_kan_bwd)