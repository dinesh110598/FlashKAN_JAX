from jax import numpy as jnp, random as jrandom
from jax import nn

def test_initialization():
    key = jrandom.key(27)
    skey, key = jrandom.split(key)
    
    batch_dim, in_dim, out_dim = 50, 8, 16
    x = jrandom.uniform(skey, [batch_dim, in_dim])
    t0, t1, k, G = 0., 5., 4, 20
    t = jnp.linspace(t0, t1, G+1)
    t = jnp.concat([jnp.full([k-1], t0), t,
                    jnp.full([k-1], t1)])
    skey, key = jrandom.split(key)
    w = jrandom.normal(skey, [G+k, in_dim, out_dim])
    act = nn.silu
    
    return batch_dim, in_dim, out_dim, t0, t1, k, G, x, t, w, act

batch_dim, in_dim, out_dim, t0, t1, k, G, x, t, w, act = test_initialization()