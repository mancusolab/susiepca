from typing import Literal, NamedTuple, Union, get_args

import jax.numpy as jnp
from jax import jit, lax, nn, random

# TODO: append internal functions to have '_'


__all__ = [
    "ModelParams",
    "ELBOResults",
    "SuSiEPCAResults",
    "compute_elbo",
    "compute_pip",
    "compute_pve",
    "susie_pca",
]


_init_type = Literal["pca", "random"]


def logdet(A):
    sign, ldet = jnp.linalg.slogdet(A)
    return ldet


class ModelParams(NamedTuple):
    """
    Define the class for variational parameters of all the variable we need
    to infer from the SuSiE PCA.

    Attributes:
        mu_z: mean parameter for factor Z
        var_z: variance parameter for factor Z
        mu_w: conditional mean parameter for loadings W
        var_w: conditional variance parameter for loading W
        alpha: parameter for the gamma that follows multinomial
                distribution
        tau: inverse variance parameter of observed data X
        tau_0: inverse variance parameter of single effect w_kl
        pi: prior probability for gamma

    """

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


#
class ELBOResults(NamedTuple):

    """Define the class of all components in ELBO.

    Attributes:
        elbo: the value of ELBO
        E_ll: Expectation of log-likelihood
        negKL_z: -KL divergence of Z
        negKL_w: -KL divergence of W
        negKL_gamma: -KL divergence of gamma

    """

    elbo: Union[float, jnp.ndarray]
    E_ll: Union[float, jnp.ndarray]
    negKL_z: Union[float, jnp.ndarray]
    negKL_w: Union[float, jnp.ndarray]
    negKL_gamma: Union[float, jnp.ndarray]

    def __str__(self):
        return (
            f"ELBO = {self.elbo:.3f} | E[logl] = {self.E_ll:.3f} | "
            f"-KL[Z] = {self.negKL_z:.3f} | -KL[W] = {self.negKL_w:.3f} | "
            f"-KL[G] = {self.negKL_gamma:.3f}"
        )


class SuSiEPCAResults(NamedTuple):
    """Define the results object returned by function :py:obj:`susie_pca`.

    Attributes:
        params: the dictionary contain all the infered parameters
        elbo: the value of ELBO
        pve: the ndarray of percent of variance explained
        pip: the ndarray of posterior inclusion probabilities
        W: the posterior mean parameter for loadings

    """

    params: ModelParams
    elbo: ELBOResults
    pve: jnp.ndarray
    pip: jnp.ndarray
    W: jnp.ndarray


def init_params(
    rng_key: random.PRNGKey,
    X: jnp.ndarray,
    z_dim: int,
    l_dim: int,
    tau: float,
    init: _init_type = "pca",
) -> ModelParams:
    """Initialize parameters for SuSiE PCA.

    Args:
        rng_key: Random number generator seed
        X: Input data. Should be a array-like
        z_dim: Latent factor dimension (K)
        l_dim: Number of single-effects comprising each factor ( L)
        init: How to initialize the variational mean parameters for latent factors.
            Either "pca" or "random" (default = "pca")
        tau: initial value of residual precision

    Returns:
        ModelParams: initialized set of model parameters.

    Raises:
        ValueError: Invalid initialization scheme.
    """

    tau_0 = jnp.ones((l_dim, z_dim))

    n_dim, p_dim = X.shape

    rng_key, mu_key, var_key, muw_key, varw_key = random.split(rng_key, 5)

    # pull type options for init
    type_options = get_args(_init_type)

    if init == "pca":
        # run PCA and extract weights and latent
        svd_result = jnp.linalg.svd(X - jnp.mean(X, axis=0))
        init_mu_z = svd_result[0] @ jnp.diag(svd_result[1])[:, 0:z_dim]
    elif init == "random":
        # random initialization
        init_mu_z = random.normal(mu_key, shape=(n_dim, z_dim))
    else:
        raise ValueError(
            f"Unknown initialization provided '{init}'; Choices: {type_options}"
        )

    init_var_z = jnp.diag(random.normal(var_key, shape=(z_dim,)) ** 2)

    # each w_kl has a specific variance term
    init_mu_w = random.normal(muw_key, shape=(l_dim, z_dim, p_dim)) * 1e-3
    init_var_w = (1 / tau_0) * (random.normal(varw_key, shape=(l_dim, z_dim))) ** 2

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


# Create a function to compute the first and second moment of W
def compute_W_moment(params):

    trace_var = jnp.sum(
        params.var_w[:, :, jnp.newaxis] * params.alpha
        + (params.mu_w ** 2 * params.alpha * (1 - params.alpha)),
        axis=(-1, 0),
    )

    E_W = jnp.sum(params.mu_w * params.alpha, axis=0)
    E_WW = E_W @ E_W.T + jnp.diag(trace_var)

    return E_W, E_WW


# Update posterior mean and variance W
def update_w(
    RtZk: jnp.ndarray, E_zzk: jnp.ndarray, params: ModelParams, kdx: int, ldx: int
) -> ModelParams:
    # n_dim, z_dim = params.mu_z.shape

    # calculate update_var_w as the new V[w | gamma]
    # suppose indep between w_k
    update_var_wkl = jnp.reciprocal(params.tau * E_zzk + params.tau_0[ldx, kdx])

    # calculate update_mu_w as the new E[w | gamma]
    update_mu_wkl = params.tau * update_var_wkl * RtZk

    return params._replace(
        mu_w=params.mu_w.at[ldx, kdx].set(update_mu_wkl),
        var_w=params.var_w.at[ldx, kdx].set(update_var_wkl),
    )


# Compute log of Bayes factor
def log_bf_np(z, s2, s0):
    return 0.5 * (jnp.log(s2) - jnp.log(s2 + 1 / s0)) + 0.5 * z ** 2 * (
        (1 / s0) / (s2 + 1 / s0)
    )


# Update tau_0 based on MLE
def update_tau0_mle(params):
    # l_dim, z_dim, p_dim = params.mu_w.shape

    est_varw = params.mu_w ** 2 + params.var_w[:, :, jnp.newaxis]

    u_tau_0 = jnp.sum(params.alpha, axis=-1) / jnp.sum(est_varw * params.alpha, axis=-1)

    return params._replace(tau_0=u_tau_0)


# Update posterior of alpha using Bayes factor
def update_alpha_bf(RtZk, E_zzk, params, kdx, ldx):
    Z_s = (RtZk / E_zzk) * jnp.sqrt(E_zzk * params.tau)
    s2_s = 1 / (E_zzk * params.tau)
    s20_s = params.tau_0[ldx, kdx]

    log_bf = log_bf_np(Z_s, s2_s, s20_s)
    log_alpha = jnp.log(params.pi) + log_bf
    alpha_kl = nn.softmax(log_alpha)

    params = params._replace(
        alpha=params.alpha.at[ldx, kdx].set(alpha_kl),
    )

    return params


# Update Posterior mean and variance of Z
def update_z(X, params):
    E_W, E_WW = compute_W_moment(params)
    z_dim, p_dim = E_W.shape

    update_var_z = jnp.linalg.inv(params.tau * E_WW + jnp.identity(z_dim))
    update_mu_z = params.tau * X @ E_W.T @ update_var_z

    return params._replace(mu_z=update_mu_z, var_z=update_var_z)


# Update tau based on MLE
def update_tau(X, params):
    n_dim, z_dim = params.mu_z.shape
    l_dim, z_dim, p_dim = params.mu_w.shape

    # calculate second moment of Z; (k x k) matrix
    E_ZZ = params.mu_z.T @ params.mu_z + n_dim * params.var_z

    # calculate moment of W
    E_W, E_WW = compute_W_moment(params)

    # expectation of log likelihood
    E_ss = (
        jnp.sum(X ** 2)
        - 2 * jnp.trace(E_W @ X.T @ params.mu_z)
        + jnp.trace(E_ZZ @ E_WW)
    )
    u_tau = (n_dim * p_dim) / E_ss

    return params._replace(tau=u_tau)


#
def compute_elbo(X: jnp.ndarray, params: ModelParams) -> ELBOResults:
    """Create function to compute evidence lower bound (ELBO)

    Args:
        X: the observed data, a N by P ndarray
        params: the dictionary contains all the infered parameters

    Returns:
        ELBOResults: the object contains all components in ELBO

    """
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
        jnp.sum(X ** 2)  # tr(X.T @ X)
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
        params.var_w[:, :, jnp.newaxis] + params.mu_w ** 2
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


def compute_pip(params: ModelParams) -> jnp.ndarray:
    """Create a function to compute the posterior inclusion probabilities (PIPs).

    Args:
        params: instance of infered parameters

    Returns:
        jnp.ndarray: Array of posterior inclusion probabilities (PIPs) for each of
        `K x P` factor, feature combinations
    """

    pip = 1 - jnp.prod(1 - params.alpha, axis=0)

    return pip


def compute_pve(params: ModelParams) -> jnp.ndarray:
    """Create a function to compute the percent of variance explained (PVE).

    Args:
        params: instance of infered parameters

    Returns:
        jnp.ndarray: Array of length `K` that contains percent of variance
        explained by each factor (PVE)
    """

    n_dim, z_dim = params.mu_z.shape
    W = jnp.sum(params.mu_w * params.alpha, axis=0)

    z_dim, p_dim = W.shape

    sk = jnp.zeros(z_dim)
    for k in range(z_dim):
        sk = sk.at[k].set(jnp.sum((params.mu_z[:, k, jnp.newaxis] * W[k, :]) ** 2))

    s = jnp.sum(sk)
    pve = sk / (s + p_dim * n_dim * (1 / params.tau))

    return pve


class _FactorLoopResults(NamedTuple):
    X: jnp.ndarray
    W: jnp.ndarray
    EZZ: jnp.ndarray
    params: ModelParams


class _EffectLoopResults(NamedTuple):
    E_zzk: jnp.ndarray
    RtZk: jnp.ndarray
    Wk: jnp.ndarray
    k: int
    params: ModelParams


def _factor_loop(kdx: int, loop_params: _FactorLoopResults) -> _FactorLoopResults:
    X, W, E_ZZ, params = loop_params

    l_dim, z_dim, p_dim = params.mu_w.shape

    # sufficient stats for inferring downstream w_kl/alpha_kl
    not_kdx = jnp.where(jnp.arange(z_dim) != kdx, size=z_dim - 1)
    E_zpzk = E_ZZ[kdx][not_kdx]
    E_zzk = E_ZZ[kdx, kdx]
    Wk = W[kdx, :]
    Wnk = W[not_kdx]
    RtZk = params.mu_z[:, kdx] @ X - Wnk.T @ E_zpzk

    # update over each of L effects
    init_loop_param = _EffectLoopResults(E_zzk, RtZk, Wk, kdx, params)
    _, _, Wk, _, params = lax.fori_loop(
        0,
        l_dim,
        _effect_loop,
        init_loop_param,
    )

    return loop_params._replace(W=W.at[kdx].set(Wk), params=params)


def _effect_loop(ldx: int, effect_params: _EffectLoopResults) -> _EffectLoopResults:
    E_zzk, RtZk, Wk, kdx, params = effect_params

    # remove current kl'th effect and update its expected residual
    Wkl = Wk - (params.mu_w[ldx, kdx] * params.alpha[ldx, kdx])
    E_RtZk = RtZk - E_zzk * Wkl

    # update conditional w_kl and alpha_kl based on current residual
    params = update_w(E_RtZk, E_zzk, params, kdx, ldx)
    params = update_alpha_bf(E_RtZk, E_zzk, params, kdx, ldx)

    # update marginal w_kl
    Wk = Wkl + (params.mu_w[ldx, kdx] * params.alpha[ldx, kdx])

    return effect_params._replace(Wk=Wk, params=params)


@jit
def _inner_loop(X: jnp.ndarray, params: ModelParams):
    n_dim, z_dim = params.mu_z.shape
    l_dim, _, _ = params.mu_w.shape

    # compute expected residuals
    # use posterior mean of Z, W, and Alpha to calculate residuals
    W = jnp.sum(params.mu_w * params.alpha, axis=0)
    E_ZZ = params.mu_z.T @ params.mu_z + n_dim * params.var_z

    # update effect precision via MLE
    params = update_tau0_mle(params)

    # update locals (W, alpha)
    init_loop_param = _FactorLoopResults(X, W, E_ZZ, params)
    _, W, _, params = lax.fori_loop(0, z_dim, _factor_loop, init_loop_param)

    # update factor parameters
    params = update_z(X, params)

    # update precision parameters via MLE
    params = update_tau(X, params)

    # compute elbo
    elbo_res = compute_elbo(X, params)

    return W, elbo_res, params


def susie_pca(
    X: jnp.ndarray,
    z_dim: int,
    l_dim: int,
    tau: float = 1.0,
    center: bool = False,
    init: _init_type = "pca",
    seed: int = 0,
    max_iter: int = 200,
    tol: float = 1e-3,
    verbose: bool = True,
) -> SuSiEPCAResults:
    """The main inference function for SuSiE PCA.

    Args:
        X: Input data. Should be a array-like
        z_dim: Latent factor dimension (int; K)
        l_dim: Number of single-effects comprising each factor (int; L)
        tau: initial value of residual precision (default = 1)
        center: Whether to center and scale the input data with mean 0
                and variance 1 (default = False)
        init: How to initialize the variational mean parameters for latent factors.
            Either "pca" or "random" (default = "pca")
        seed: Seed for "random" initialization (int)
        max_iter: Maximum number of iterations for inference (int)
        tol: Numerical tolerance for ELBO convergence (float)
        verbose: Flag to indicate displaying log information (ELBO value) in each
            iteration

    Returns:
        :py:obj:`SuSiEPCAResults`: tuple that has member variables for learned
        parameters (:py:obj:`ModelParams`), evidence lower bound (ELBO) results
        (:py:obj:`ELBOResults`) from the last iteration, the percent of variance
        explained (PVE) for each of the `K` factors (:py:obj:`jax.numpy.ndarray`),
        the posterior inclusion probabilities (PIPs) for each of the `K` factors
        and `P` features (:py:obj:`jax.numpy.ndarray`), and the posterior mean of
        loading matrix of shape `K x P` (:py:obj:`jax.numpy.ndarray`).

    Raises:
        ValueError: Invalid `l_dim` or `z_dim` values. Invalid initialization scheme.
        Data `X` contains `inf` or `nan`.
    """

    # pull type options for init
    type_options = get_args(_init_type)

    # cast to jax array
    X = jnp.asarray(X)
    if len(X.shape) != 2:
        raise ValueError(f"Shape of X = {X.shape}; Expected 2-dim matrix")

    # should we check for n < p?
    n_dim, p_dim = X.shape

    # dim checks
    if l_dim > p_dim:
        raise ValueError(
            f"l_dim should be less than p: received l_dim = {l_dim}, p = {p_dim}"
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

    # quality checks
    if jnp.any(jnp.isnan(X)):
        raise ValueError(
            "X contains 'nan'. Please check input data for correctness or missingness"
        )
    if jnp.any(jnp.isinf(X)):
        raise ValueError(
            "X contains 'inf'. Please check input data for correctness or missingness"
        )

    # option to center the data
    if center:
        X -= jnp.mean(X, axis=0)
        X /= jnp.std(X, axis=0)

    # type check for init
    if init not in type_options:
        raise ValueError(
            f"Unknown initialization provided '{init}'; Choices: {type_options}"
        )

    # initialize PRNGkey and params
    rng_key = random.PRNGKey(seed)
    params = init_params(rng_key, X, z_dim, l_dim, tau, init)

    # run inference
    elbo = -5e25
    for idx in range(1, max_iter + 1):

        #  core loop for inference
        W, elbo_res, params = _inner_loop(X, params)

        if verbose:
            print(f"Iter [{idx}] | {elbo_res}")

        diff = elbo_res.elbo - elbo
        if diff < 0 and verbose:
            print(f"Alert! Diff between elbo[{idx - 1}] and elbo[{idx}] = {diff}")
        if jnp.fabs(diff) < tol:
            if verbose:
                print(f"Elbo diff tolerance reached at iteration {idx}")
            break

        elbo = elbo_res.elbo

    # compute PVE
    pve = compute_pve(params)

    # compute PIPs
    pip = compute_pip(params)

    return SuSiEPCAResults(params, elbo_res, pve, pip, W)
