import jax.numpy as jnp
from jax import random


def generate_sim(seed, l_dim, n_dim, p_dim, z_dim, effect_size=1):
    rng_key = random.PRNGKey(seed)
    rng_key, z_key, b_key, obs_key = random.split(rng_key, 4)

    Z = random.normal(z_key, shape=(n_dim, z_dim))
    W = jnp.zeros(shape=(z_dim, p_dim))

    for k in range(z_dim):

        W = W.at[k, (k * l_dim) : ((k + 1) * l_dim)].set(
            effect_size * random.normal(b_key, shape=(l_dim,))
        )

    m = Z @ W

    X = m + random.normal(obs_key, shape=(n_dim, p_dim))

    return Z, W, X
