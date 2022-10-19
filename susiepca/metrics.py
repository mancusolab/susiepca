import jax.numpy as jnp


def mse(X, Xhat):
    return jnp.sum((X - Xhat) ** 2) / jnp.sum(X ** 2)


def get_credset(params, rho=0.9):
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
