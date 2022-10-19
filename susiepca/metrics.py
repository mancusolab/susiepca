import jax.numpy as jnp
import pandas as pd


def mse(X, Xhat):
    return jnp.sum((X - Xhat) ** 2) / jnp.sum(X ** 2)


def compute_largest_pip(pip, n, absol=False):
    z_dim, n_dim = pip.shape
    pip_df = pd.DataFrame(pip.T)

    credible = {}
    credible_index = []

    for k in range(z_dim):
        if absol:
            credible["z" + str(k)] = pip_df.abs().nlargest(n, k)[k]
            credible_index.append(pip_df.abs().nlargest(n, k)[k].index)
        else:
            credible["z" + str(k)] = pip_df.nlargest(n, k)[k]
            credible_index.append(pip_df.nlargest(n, k)[k].index)

    return credible, credible_index


def compute_pve(mu_z, W, tau):
    n_dim, z_dim = mu_z.shape
    z_dim, p_dim = W.shape

    sk = jnp.zeros(z_dim)
    for k in range(z_dim):
        sk = sk.at[k].set(jnp.sum((mu_z[:, k, jnp.newaxis] * W[k, :]) ** 2))

    s = jnp.sum(sk)
    pve = sk / (s + p_dim * n_dim * (1 / tau))

    return pve
