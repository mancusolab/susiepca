from typing import NamedTuple

import jax.numpy as jnp
from jax import random


class SimulateData(NamedTuple):
    Z: jnp.ndarray
    W: jnp.ndarray
    X: jnp.ndarray


def generate_sim(
    seed: int,
    l_dim: int,
    n_dim: int,
    p_dim: int,
    z_dim: int,
    effect_size: int = 1,
) -> SimulateData:
    """


    Args:
        seed : Seed for "random" initialization (int)
        l_dim : Number of single effects in each factor (L)
        n_dim : Number of sample in the data (N)
        p_dim : Number of feature in the data (P)
        z_dim : Latent dimensions (K)
        effect_size : The effect size of features contributing to the factor.
                      The default is 1.

    Returns
    -------
    Z : Simulated factors (N by K)
    W : Simulated factor loadings (K by P)
    X : Simulated data (N by P)

    """
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

    return SimulateData(Z, W, X)
