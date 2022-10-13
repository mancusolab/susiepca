import jax.numpy as jnp
import numpy as np
import pandas as pd
import procrustes


def mse(X, Xhat):
    return ((X - Xhat) ** 2).sum() / (X**2).sum()


def procrustes_norm(W, What):

    # Procruste Transformation
    proc_trans_susie = procrustes.orthogonal(
        np.asarray(What.T), np.asarray(W.T), scale=True
    )

    err = proc_trans_susie.error

    return err


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
