import jax.numpy as jnp
from jax import random


def simulation_order(rng_key, n_dim=200, p_dim=320, z_dim=4):
    rng_key, z_key, b_key, obs_key = random.split(rng_key, 4)

    Z = random.normal(z_key, shape=(n_dim, z_dim))

    l_dim = int(p_dim / z_dim)

    # effects
    w1_nonzero = 1 * random.normal(b_key, shape=(l_dim,))
    w1_rest = jnp.zeros(shape=(p_dim - l_dim,))
    w1 = jnp.concatenate((w1_nonzero, w1_rest))

    w2_nonzero = 1 * random.normal(b_key, shape=(l_dim,))
    w2_rest = jnp.zeros(shape=(p_dim - 2 * l_dim,))
    w2 = jnp.concatenate((jnp.zeros(shape=(l_dim,)), w2_nonzero, w2_rest))

    w3_nonzero = 2 * random.normal(b_key, shape=(l_dim,))
    w3_rest = jnp.zeros(shape=(p_dim - 3 * l_dim,))
    w3 = jnp.concatenate((jnp.zeros(shape=(2 * l_dim,)), w3_nonzero, w3_rest))

    w4_nonzero = 1 * random.normal(b_key, shape=(l_dim,))
    w4_rest = jnp.zeros(shape=(p_dim - l_dim,))
    w4 = jnp.concatenate((w4_rest, w4_nonzero))

    W = jnp.vstack((w1, w2, w3, w4))

    m = Z @ W

    X = m + random.normal(obs_key, shape=(n_dim, p_dim))

    return Z, W, X, m
