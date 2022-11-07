import jax.numpy as jnp

# from jax import jit

__all__ = [
    "mse",
    "get_credset",
]


def mse(X: jnp.ndarray, Xhat: jnp.ndarray) -> float:
    """Create a function to compute relative root mean square error.

    Args:
        X : Input data. Should be a array-like
        Xhat : Predicted data. Should be in the same shape with X

    Returns:
        RRMSE: relative root mean square error

    """
    if X.shape != Xhat.shape:
        raise ValueError("Predicted data shape doesn't match, please check")

    mse = jnp.sum((X - Xhat) ** 2) / jnp.sum(X ** 2)
    return mse


def get_credset(params, rho=0.9):

    """Creat a function to compute the rho-level credible set

    Args:
        params: the dictionary return from the function susie_pca
        rho: the level from credible set, should ranged in (0,1)

    Returns:
        cs: credible set, which is a dictionary contain K*P credible sets

    """

    l_dim, z_dim, p_dim = params.alpha.shape
    idxs = jnp.argsort(-params.alpha, axis=-1)
    cs = {}
    for zdx in range(z_dim):
        cs_s = []
        for ldx in range(l_dim):
            cs_s.append([])
            local = 0.0
            for pdx in range(p_dim):
                if local >= rho:
                    break
                idx = idxs[ldx][zdx][pdx]
                cs_s[ldx].append(idx)
                local += params.alpha[ldx, zdx, idx]
        cs["z" + str(zdx)] = cs_s

    return cs


def get_credset_v2(params, rho=0.9):
    l_dim, z_dim, p_dim = params.alpha.shape
    idxs = jnp.argsort(-params.alpha, axis=-1)
    cs = {}
    for zdx in range(z_dim):
        cs_s = []
        for ldx in range(l_dim):
            cs_s.append([])

            # idxs for all feature at this zdx and ldx
            p_idxs = idxs[ldx, zdx, :]
            # compute the cumulative sum
            p_sums = jnp.cumsum(params.alpha[ldx, zdx, p_idxs])
            # find all the index where the cumsum>rho
            p_gts = jnp.where(p_sums >= rho)[0]
            # get the minimum value that satisfy the above criterion
            min_p_gts = p_gts[0]
            # form the cs. note it's possible that min_p_gets is 0
            idx = p_idxs[0 : max(min_p_gts, 1)]
            cs_s[ldx].append(idx)

        cs["z" + str(zdx)] = cs_s

    return cs
