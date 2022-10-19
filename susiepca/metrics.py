import jax.numpy as jnp


def mse(X, Xhat):
    return jnp.sum((X - Xhat) ** 2) / jnp.sum(X ** 2)


def compute_pip(params):

    pip = 1 - jnp.prod(1 - params.alpha, axis=0)

    return pip


def compute_pve(params):
    n_dim, z_dim = params.mu_z.shape
    W = jnp.sum(params.mu_w * params.alpha, axis=0)

    z_dim, p_dim = W.shape

    sk = jnp.zeros(z_dim)
    for k in range(z_dim):
        sk = sk.at[k].set(jnp.sum((params.mu_z[:, k, jnp.newaxis] * W[k, :]) ** 2))

    s = jnp.sum(sk)
    pve = sk / (s + p_dim * n_dim * (1 / params.tau))

    return pve
