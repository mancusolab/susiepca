from typing import NamedTuple

import jax.numpy as jnp
from jax import random

__all__ = [
    "SimulatedData",
    "generate_sim",
]


class SimulatedData(NamedTuple):
    """the object contain simulated data components.

    Args:
        Z: simulated factor
        W: simulated loadings
        X: simulated data set

    """

    Z: jnp.ndarray
    W: jnp.ndarray
    X: jnp.ndarray


def generate_sim(
    seed: int,
    l_dim: int,
    n_dim: int,
    p_dim: int,
    z_dim: int,
    effect_size: float = 1.0,
) -> SimulatedData:
    """Create the function to generate a sparse data for PCA.

    Args:
        seed: Seed for "random" initialization
        l_dim: Number of single effects in each factor
        n_dim: Number of sample in the data
        p_dim: Number of feature in the data
        z_dim: Number of Latent dimensions
        effect_size: The effect size of features contributing to the factor.
                      The default is 1.

    Returns:
        Z: Simulated factors (N by K)
        W: Simulated factor loadings (K by P)
        X: Simulated data (N by P)

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

    return SimulatedData(Z, W, X)
