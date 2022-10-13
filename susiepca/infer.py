from typing import NamedTuple, Union

import jax.numpy as jnp
import jax.scipy.special as spec
import scipy.optimize as sopt
from jax import jit, nn, random


def logdet(A):
    sign, ldet = jnp.linalg.slogdet(A)
    return ldet


class ModelParams(NamedTuple):
    # variational params for Z
    mu_z: jnp.ndarray
    var_z: jnp.ndarray

    # variational params for W given Gamma
    mu_w: jnp.ndarray
    var_w: jnp.ndarray

    # variational params for Gamma
    alpha: jnp.ndarray

    # residual precision param
    tau: Union[float, jnp.ndarray]
    tau_0: jnp.ndarray

    # prior probability for gamma
    pi: jnp.ndarray


class ELBOResults(NamedTuple):
    elbo: Union[float, jnp.ndarray]
    E_ll: Union[float, jnp.ndarray]
    negKL_z: Union[float, jnp.ndarray]
    negKL_w: Union[float, jnp.ndarray]
    negKL_gamma: Union[float, jnp.ndarray]

    def __str__(self):
        return f"ELBO = {self.elbo} | E_ll = {self.E_ll} | -KL[Z] = {self.negKL_z} | -KL[W] = {self.negKL_w} | -KL[G] = {self.negKL_gamma}"


class SuSiEPCAResults(NamedTuple):
    params: ModelParams
    elbo: ELBOResults


def init_params(rng_key, n_dim, p_dim, z_dim, l_dim):
    tau = 10.0
    tau_0 = jnp.ones((l_dim, z_dim))

    rng_key, mu_key, var_key = random.split(rng_key, 3)
    init_mu_z = random.normal(mu_key, shape=(n_dim, z_dim))
    init_var_z = jnp.diag(random.normal(var_key, shape=(z_dim,)) ** 2)

    rng_key, mu_key, var_key = random.split(rng_key, 3)
    init_mu_w = random.normal(mu_key, shape=(l_dim, z_dim, p_dim)) * 1e-3

    # suppose each w_kl has a specific variance term
    init_var_w = (1 / tau_0) * (random.normal(var_key, shape=(l_dim, z_dim))) ** 2

    rng_key, alpha_key = random.split(rng_key, 2)
    init_alpha = random.dirichlet(
        alpha_key, alpha=jnp.ones(p_dim), shape=(l_dim, z_dim)
    )

    pi = jnp.ones(p_dim) / p_dim

    return ModelParams(
        init_mu_z,
        init_var_z,
        init_mu_w,
        init_var_w,
        init_alpha,
        tau,
        tau_0,
        pi=pi,
    )


def compute_W_moment(params):
    # l_dim, z_dim, p_dim = params.mu_w.shape

    # tr(V[w_k]) = sum_l tr(V[w_kl]) =
    #   sum_l sum_i [(E[w_kl^2 | gamma_kli = 1] * E[gamma_kl]) + (E[w_kl | gamma_kli = 1] * E[gamma_kl]) ** 2
    # mu**2 * a - mu**2 * a * a = mu**2 * a * (1 - a) = mu**2 * var[a]
    # V[w_kl] = var[w] * E[a] + E[w]*var[a]
    # V[Y] = E_X[V[Y | X]] + V_X[E[Y | X]]
    trace_var = jnp.sum(
        params.var_w[:, :, jnp.newaxis] * params.alpha
        + (params.mu_w**2 * params.alpha * (1 - params.alpha)),
        axis=(-1, 0),
    )

    E_W = jnp.sum(params.mu_w * params.alpha, axis=0)
    E_WW = E_W @ E_W.T + jnp.diag(trace_var)

    return E_W, E_WW


# update w
def update_w(RtZk, E_zzk, params, k, l):
    # n_dim, z_dim = params.mu_z.shape

    # calculate update_var_w as the new V[w | gamma]
    # suppose indep between w_k
    update_var_wkl = jnp.reciprocal(params.tau * E_zzk + params.tau_0[l, k])

    # calculate update_mu_w as the new E[w | gamma]
    update_mu_wkl = params.tau * update_var_wkl * RtZk

    return params._replace(
        mu_w=params.mu_w.at[l, k].set(update_mu_wkl),
        var_w=params.var_w.at[l, k].set(update_var_wkl),
    )


def log_bf_np(z, s2, s0):
    return 0.5 * (jnp.log(s2) - jnp.log(s2 + 1 / s0)) + 0.5 * z**2 * (
        (1 / s0) / (s2 + 1 / s0)
    )


def update_tau0(RtZk, E_zzk, params, k, l):
    Z_s = (RtZk / E_zzk) * jnp.sqrt(E_zzk * params.tau)
    s2_s = 1 / (E_zzk * params.tau)

    def min_obj(log_s20_s):
        return -spec.logsumexp(
            jnp.log(params.pi) + log_bf_np(Z_s, s2_s, jnp.exp(log_s20_s))
        )

    res = sopt.minimize_scalar(min_obj, method="bounded", bounds=(-30, 10))
    new_s20_s = jnp.exp(res.x)
    params = params._replace(tau_0=params.tau_0.at[l, k].set(new_s20_s))

    return params


def update_tau0_mle(params):
    # l_dim, z_dim, p_dim = params.mu_w.shape

    est_varw = params.mu_w**2 + params.var_w[:, :, jnp.newaxis]

    u_tau_0 = jnp.sum(params.alpha, axis=-1) / jnp.sum(est_varw * params.alpha, axis=-1)

    return params._replace(tau_0=u_tau_0)


def update_alpha_bf(RtZk, E_zzk, params, k, l):
    Z_s = (RtZk / E_zzk) * jnp.sqrt(E_zzk * params.tau)
    s2_s = 1 / (E_zzk * params.tau)
    s20_s = params.tau_0[l, k]

    log_bf = log_bf_np(Z_s, s2_s, s20_s)
    log_alpha = jnp.log(params.pi) + log_bf
    alpha_kl = nn.softmax(log_alpha)

    params = params._replace(
        alpha=params.alpha.at[l, k].set(alpha_kl),
    )

    return params


def update_z(X, params):
    E_W, E_WW = compute_W_moment(params)
    z_dim, p_dim = E_W.shape

    update_var_z = jnp.linalg.inv(params.tau * E_WW + jnp.identity(z_dim))
    update_mu_z = params.tau * X @ E_W.T @ update_var_z

    return params._replace(mu_z=update_mu_z, var_z=update_var_z)


def update_tau(X, params):
    n_dim, z_dim = params.mu_z.shape
    l_dim, z_dim, p_dim = params.mu_w.shape

    # calculate second moment of Z; (k x k) matrix
    E_ZZ = params.mu_z.T @ params.mu_z + n_dim * params.var_z

    # calculate moment of W
    E_W, E_WW = compute_W_moment(params)

    # expectation of log likelihood
    E_ss = (
        jnp.sum(X**2)
        - 2 * jnp.trace(E_W @ X.T @ params.mu_z)
        + jnp.trace(E_ZZ @ E_WW)
    )
    u_tau = (n_dim * p_dim) / E_ss

    return params._replace(tau=u_tau)


def compute_elbo(X, params) -> ELBOResults:
    n_dim, z_dim = params.mu_z.shape
    l_dim, z_dim, p_dim = params.mu_w.shape

    # calculate second moment of Z along k, (k x k) matrix
    # E[Z'Z] = V_k[Z] * tr(I_n) + E[Z]'E[Z] = V_k[Z] * n + E[Z]'E[Z]
    E_ZZ = n_dim * params.var_z + params.mu_z.T @ params.mu_z

    # calculate moment of W
    E_W, E_WW = compute_W_moment(params)

    # expectation of log likelihood
    # calculation tip: tr(A @ A.T) = tr(A.T @ A) = sum(A ** 2)
    # (X.T @ E[Z] @ E[W]) is p x p (big!); compute (E[W] @ X.T @ E[Z]) (k x k)
    E_ll = (-0.5 * params.tau) * (
        jnp.sum(X**2)  # tr(X.T @ X)
        - 2 * jnp.trace(E_W @ X.T @ params.mu_z)  # tr(E[W] @ X.T @ E[Z])
        + jnp.trace(E_ZZ @ E_WW)  # tr(E[Z.T @ Z] @ E[W @ W.T])
    ) + 0.5 * n_dim * p_dim * jnp.log(
        params.tau
    )  # -0.5 * n * p * log(1 / tau) = 0.5 * n * p * log(tau)

    # neg-KL for Z
    negKL_z = 0.5 * (-jnp.trace(E_ZZ) + n_dim * z_dim + n_dim * logdet(params.var_z))

    # neg-KL for w
    # awkward indexing to get broadcast working
    klw_term1 = params.tau_0[:, :, jnp.newaxis] * (
        params.var_w[:, :, jnp.newaxis] + params.mu_w**2
    )
    klw_term2 = (
        klw_term1
        - 1.0
        - (jnp.log(params.tau_0) + jnp.log(params.var_w))[:, :, jnp.newaxis]
    )
    negKL_w = -0.5 * jnp.sum(params.alpha * klw_term2)

    # neg-KL for gamma
    negKL_gamma = -jnp.sum(
        params.alpha * (jnp.log(params.alpha + 1e-10) - jnp.log(params.pi))
    )

    elbo = E_ll + negKL_z + negKL_w + negKL_gamma

    result = ELBOResults(elbo, E_ll, negKL_z, negKL_w, negKL_gamma)

    return result


def compute_pip(params):
    pip = 1 - jnp.prod(1 - params.alpha, axis=0)

    return pip


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


@jit
def _inner_loop(X, params):
    n_dim, z_dim = params.mu_z.shape
    l_dim, _, _ = params.mu_w.shape

    # compute expected residuals
    # use posterior mean of Z, W, and Alpha to calculate residuals
    W = jnp.sum(params.mu_w * params.alpha, axis=0)
    E_ZZ = params.mu_z.T @ params.mu_z + n_dim * params.var_z
    params = update_tau0_mle(params)

    # update locals (W, Gamma)
    for k in range(z_dim):
        E_zpzk = jnp.delete(E_ZZ[k], k)
        E_zzk = E_ZZ[k, k]
        Wk = W[k, :]
        Wnk = jnp.delete(W, k, axis=0)
        RtZk = params.mu_z[:, k] @ X - Wnk.T @ E_zpzk

        for l in range(l_dim):
            Wkl = Wk - (params.mu_w[l, k] * params.alpha[l, k])

            E_RtZk = RtZk - E_zzk * Wkl
            params = update_w(E_RtZk, E_zzk, params, k, l)
            params = update_alpha_bf(E_RtZk, E_zzk, params, k, l)

            Wk = Wkl + (params.mu_w[l, k] * params.alpha[l, k])

    params = update_z(X, params)
    params = update_tau(X, params)

    # compute elbo
    elbo_tmp = compute_elbo(X, params)

    return W, elbo_tmp, params


def susie_pca(
    X: jnp.ndarray,
    z_dim: int,
    l_dim: int,
    seed: int = 0,
    max_iter: int = 100,
    tol: float = 1e-3,
) -> SuSiEPCAResults:
    """

    Args:
        X:
        z_dim:
        l_dim:
        seed:
        max_iter:
        tol:

    Returns:

    """
    n_dim, p_dim = X.shape
    # TODO: error checking on z_dim, l_dim wrt n_dim, p_dim, positive, etc

    # initialize PRNGkey and params
    rng_key = random.PRNGKey(seed)
    params = init_params(rng_key, n_dim, p_dim, z_dim, l_dim)

    # run inference
    elbo = -5e25
    for idx in range(1, max_iter + 1):

        W, elbo_res, params = _inner_loop(X, params)

        print(f"Iter [{idx}] | {elbo_res}")

        diff = elbo_res.elbo - elbo
        if diff < 0:
            print(f"Bug alert! Diff between elbo[{idx - 1}] and elbo[{idx}] = {diff}")
        if jnp.fabs(diff) < tol:
            print(f"Elbo diff tolerance reached at iteration {idx}")
            break
        if idx == max_iter:
            print("Maximum Iteration reached")
            break

        elbo = elbo_res.elbo

    return SuSiEPCAResults(params, elbo_res)
