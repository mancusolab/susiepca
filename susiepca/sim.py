from typing import NamedTuple

import jax.numpy as jnp
from jax import random

__all__ = [
    "SimulatedData",
    "generate_sim",
]


class SimulatedData(NamedTuple):
    """the object contain simulated data components.

    Attributes:
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
    """Create the function to generate a sparse data for PCA. Please note that to
        illustrate how SuSiE PCA work, we build this simple and
        straitforward simulation where each latent component have exact l_dim number
        of non-overlapped single effects. Please make sure l_dim < p_dim/z_dim
        when generate simulation data using this function.

    Args:
        seed: Seed for "random" initialization
        l_dim: Number of single effects in each factor
        n_dim: Number of sample in the data
        p_dim: Number of feature in the data
        z_dim: Number of Latent dimensions
        effect_size: The effect size of features contributing to the factor.
                      (default = 1).

    Returns:
        SimulatedData: Tuple that contains simulated factors (`N x K`),
        W (factor loadings (`K x P`), and data X (data (`N x P`).
    """

    # interger seed
    if isinstance(seed, int) is False:
        raise ValueError(f"seed should be an interger: received seed = {seed}")

    rng_key = random.PRNGKey(seed)
    rng_key, z_key, b_key, obs_key = random.split(rng_key, 4)

    # dimension check
    if l_dim > p_dim:
        raise ValueError(
            f"l_dim should be less than p: received l_dim = {l_dim}, p = {p_dim}"
        )
    if l_dim > p_dim / z_dim:
        raise ValueError(
            f"""l_dim is smaller than p_dim/z_dim,
            please make sure each component has {l_dim} single effects"""
        )

    if l_dim <= 0:
        raise ValueError(f"l_dim should be positive: received l_dim = {l_dim}")

    if z_dim > p_dim:
        raise ValueError(
            f"z_dim should be less than p: received z_dim = {z_dim}, p = {p_dim}"
        )
    if z_dim > n_dim:
        raise ValueError(
            f"z_dim should be less than n: received z_dim = {z_dim}, n = {n_dim}"
        )
    if z_dim <= 0:
        raise ValueError(f"z_dim should be positive: received z_dim = {z_dim}")

    if effect_size <= 0:
        raise ValueError(
            f"effect size should be positive: received effect_size = {effect_size}"
        )

    Z = random.normal(z_key, shape=(n_dim, z_dim))
    W = jnp.zeros(shape=(z_dim, p_dim))

    for k in range(z_dim):

        W = W.at[k, (k * l_dim) : ((k + 1) * l_dim)].set(
            effect_size * random.normal(b_key, shape=(l_dim,))
        )

    m = Z @ W

    X = m + random.normal(obs_key, shape=(n_dim, p_dim))

    return SimulatedData(Z, W, X)
