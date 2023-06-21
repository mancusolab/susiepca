from typing import Literal, NamedTuple, Optional, Tuple, get_args

import optax

from jax import Array, grad, jit, lax, nn, numpy as jnp, random
from jax.scipy import special as spec
from jax.typing import ArrayLike

__all__ = [
    "compute_elbo",
    "susie_pca",
]

from susiepca.common import (
    ELBOResults,
    ModelParams,
    SuSiEPCAResults,
    compute_pip,
    compute_pve,
)

_init_type = Literal["pca", "random"]


def logdet(A):
    sign, ldet = jnp.linalg.slogdet(A)
    return ldet


def _compute_pi(A: ArrayLike, theta: ArrayLike) -> Array:
    return nn.softmax(A @ theta, axis=0)


def _kl_gamma(alpha: ArrayLike, pi: ArrayLike) -> float:
    return jnp.sum(spec.xlogy(alpha, alpha) - spec.xlogy(alpha, pi.T))


def _compute_w_moment(params: ModelParams) -> Tuple[Array, Array]:
    trace_var = jnp.sum(
        params.var_w[:, :, jnp.newaxis] * params.alpha
        + (params.mu_w**2 * params.alpha * (1 - params.alpha)),
        axis=(-1, 0),
    )

    E_W = params.W
    E_WW = E_W @ E_W.T + jnp.diag(trace_var)

    return E_W, E_WW


# Update posterior mean and variance W
def _update_w(
    RtZk: ArrayLike, E_zzk: ArrayLike, params: ModelParams, kdx: int, ldx: int
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
def _log_bf_np(z: ArrayLike, s2: ArrayLike, s0: ArrayLike):
    return 0.5 * (jnp.log(s2) - jnp.log(s2 + 1 / s0)) + 0.5 * z**2 * (
        (1 / s0) / (s2 + 1 / s0)
    )


# Update tau_0 based on MLE
def _update_tau0_mle(params: ModelParams) -> ModelParams:
    # l_dim, z_dim, p_dim = params.mu_w.shape

    est_varw = params.mu_w**2 + params.var_w[:, :, jnp.newaxis]

    u_tau_0 = jnp.sum(params.alpha, axis=-1) / jnp.sum(est_varw * params.alpha, axis=-1)

    return params._replace(tau_0=u_tau_0)


# Update posterior of alpha using Bayes factor
def _update_alpha_bf(
    RtZk: ArrayLike, E_zzk: ArrayLike, params: ModelParams, kdx: int, ldx: int
) -> ModelParams:
    Z_s = (RtZk / E_zzk) * jnp.sqrt(E_zzk * params.tau)
    s2_s = 1 / (E_zzk * params.tau)
    s20_s = params.tau_0[ldx, kdx]

    log_bf = _log_bf_np(Z_s, s2_s, s20_s)
    log_alpha = jnp.log(params.pi) + log_bf
    alpha_kl = nn.softmax(log_alpha)

    params = params._replace(
        alpha=params.alpha.at[ldx, kdx].set(alpha_kl),
    )

    return params


# Update Posterior mean and variance of Z
def _update_z(X: ArrayLike, params: ModelParams) -> ModelParams:
    E_W, E_WW = _compute_w_moment(params)
    z_dim, p_dim = E_W.shape

    update_var_z = jnp.linalg.inv(params.tau * E_WW + jnp.identity(z_dim))
    update_mu_z = params.tau * X @ E_W.T @ update_var_z

    return params._replace(mu_z=update_mu_z, var_z=update_var_z)


# Update tau based on MLE
def _update_tau(X: ArrayLike, params: ModelParams) -> ModelParams:
    n_dim, z_dim = params.mu_z.shape
    l_dim, z_dim, p_dim = params.mu_w.shape

    # calculate second moment of Z; (k x k) matrix
    E_ZZ = params.mu_z.T @ params.mu_z + n_dim * params.var_z

    # calculate moment of W
    E_W, E_WW = _compute_w_moment(params)

    # expectation of log likelihood
    E_ss = (
        jnp.sum(X**2)
        - 2 * jnp.trace(E_W @ X.T @ params.mu_z)
        + jnp.trace(E_ZZ @ E_WW)
    )
    u_tau = (n_dim * p_dim) / E_ss

    return params._replace(tau=u_tau)


def _update_theta(
    params: ModelParams,
    A: ArrayLike,
    lr: float = 1e-2,
    tol: float = 1e-3,
    max_iter: int = 100,
) -> ModelParams:
    optimizer = optax.adam(lr)
    init_opt_state = optimizer.init(params.theta)

    def _loss(theta_i: ArrayLike) -> float:
        pi = _compute_pi(A, theta_i)
        return _kl_gamma(params.alpha, pi)

    def body_fun(inputs):
        old_theta, theta, idx, opt_state = inputs
        grads = grad(_loss)(theta)
        updates, new_optimizer_state = optimizer.update(grads, opt_state)
        new_theta = optax.apply_updates(theta, updates)
        old_theta = theta
        return old_theta, new_theta, idx + 1, new_optimizer_state

    # define a function to check the stopping criterion
    def cond_fn(inputs):
        old_theta, theta, idx, _ = inputs
        tol_check = jnp.linalg.norm(theta - old_theta) > tol
        iter_check = idx > max_iter
        return jnp.logical_and(tol_check, iter_check)

    # use jax.lax.while_loop until the change in parameters is less than a given tolerance
    old_theta, theta, idx_count, opt_state = lax.while_loop(
        cond_fn,
        body_fun,
        (jnp.zeros_like(params.theta), params.theta, 0, init_opt_state),
    )

    return params._replace(theta=theta, pi=_compute_pi(A, theta))


def compute_elbo(X: ArrayLike, params: ModelParams) -> ELBOResults:
    """Create function to compute evidence lower bound (ELBO)

    Args:
        X: the observed data, an N by P ndarray
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
    E_W, E_WW = _compute_w_moment(params)

    # expectation of log likelihood
    # calculation tip: tr(A @ A.T) = tr(A.T @ A) = sum(A ** 2)
    # (X.T @ E[Z] @ E[W]) is p x p (big!); compute (E[W] @ X.T @ E[Z]) (k x k)
    E_ll = (-0.5 * params.tau) * (
        jnp.sum(X**2)  # tr(X.T @ X)
        - 2 * jnp.einsum("kp,np,nk->", E_W, X, params.mu_z)  # tr(E[W] @ X.T @ E[Z])
        + jnp.einsum("ij,ji->", E_ZZ, E_WW)  # tr(E[Z.T @ Z] @ E[W @ W.T])
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
    negKL_gamma = -_kl_gamma(params.alpha, params.pi)

    elbo = E_ll + negKL_z + negKL_w + negKL_gamma

    result = ELBOResults(elbo, E_ll, negKL_z, negKL_w, negKL_gamma)

    return result


class _FactorLoopResults(NamedTuple):
    X: Array
    W: Array
    EZZ: Array
    params: ModelParams


class _EffectLoopResults(NamedTuple):
    E_zzk: Array
    RtZk: Array
    Wk: Array
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
    params = _update_w(E_RtZk, E_zzk, params, kdx, ldx)
    params = _update_alpha_bf(E_RtZk, E_zzk, params, kdx, ldx)

    # update marginal w_kl
    Wk = Wkl + (params.mu_w[ldx, kdx] * params.alpha[ldx, kdx])

    return effect_params._replace(Wk=Wk, params=params)


@jit
def _inner_loop(X: ArrayLike, params: ModelParams):
    n_dim, z_dim = params.mu_z.shape
    l_dim, _, _ = params.mu_w.shape

    # compute expected residuals
    # use posterior mean of Z, W, and Alpha to calculate residuals
    W = params.W
    E_ZZ = params.mu_z.T @ params.mu_z + n_dim * params.var_z

    # update effect precision via MLE
    params = _update_tau0_mle(params)

    # update locals (W, alpha)
    init_loop_param = _FactorLoopResults(X, W, E_ZZ, params)
    _, W, _, params = lax.fori_loop(0, z_dim, _factor_loop, init_loop_param)

    # update factor parameters
    params = _update_z(X, params)

    # update precision parameters via MLE
    params = _update_tau(X, params)

    # compute elbo
    elbo_res = compute_elbo(X, params)

    return elbo_res, params


@jit
def _annotation_inner_loop(X: jnp.ndarray, A: jnp.ndarray, params: ModelParams):
    n_dim, z_dim = params.mu_z.shape
    l_dim, _, _ = params.mu_w.shape

    # compute expected residuals
    # use posterior mean of Z, W, and Alpha to calculate residuals
    W = params.W
    E_ZZ = params.mu_z.T @ params.mu_z + n_dim * params.var_z

    # perform MLE inference before variational inference
    # update effect precision via MLE
    params = _update_tau0_mle(params)

    # update theta via MLE
    # jax.debug.print("update theta")
    params = _update_theta(params, A)

    # update locals (W, alpha)
    init_loop_param = _FactorLoopResults(X, W, E_ZZ, params)
    _, W, _, params = lax.fori_loop(0, z_dim, _factor_loop, init_loop_param)

    # update factor parameters
    params = _update_z(X, params)

    # update precision parameters via MLE
    params = _update_tau(X, params)

    # compute elbo
    elbo_res = compute_elbo(X, params)

    return W, elbo_res, params


def _reorder_factors_by_pve(
    A: ArrayLike, params: ModelParams, pve: ArrayLike
) -> Tuple[ModelParams, Array]:
    sorted_indices = jnp.argsort(pve)[::-1]
    pve = pve[sorted_indices]
    sorted_mu_z = params.mu_z[:, sorted_indices]
    sorted_var_z = params.var_z[sorted_indices, sorted_indices]
    sorted_mu_w = params.mu_w[:, sorted_indices, :]
    sorted_var_w = params.var_w[:, sorted_indices]
    sorted_alpha = params.alpha[:, sorted_indices, :]
    sorted_tau_0 = params.tau_0[:, sorted_indices]
    if A is not None:
        sorted_theta = params.theta[:, sorted_indices]
        sorted_pi = _compute_pi(A, sorted_theta)
    else:
        sorted_theta = None
        sorted_pi = params.pi

    params = ModelParams(
        sorted_mu_z,
        sorted_var_z,
        sorted_mu_w,
        sorted_var_w,
        sorted_alpha,
        params.tau,
        sorted_tau_0,
        sorted_theta,
        sorted_pi,
    )

    return params, pve


def _init_params(
    rng_key: random.PRNGKey,
    X: ArrayLike,
    A: ArrayLike,
    z_dim: int,
    l_dim: int,
    tau: float,
    init: _init_type = "pca",
) -> ModelParams:
    """Initialize parameters for SuSiE PCA.

    Args:
        rng_key: Random number generator seed
        X: Input data. Should be an array-like
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
    p_dim, m = A.shape

    rng_key, mu_key, var_key, muw_key, varw_key, alpha_key, theta_key = random.split(
        rng_key, 7
    )

    # pull type options for init
    type_options = get_args(_init_type)

    if init == "pca":
        # run PCA and extract weights and latent
        svd_result = jnp.linalg.svd(X - jnp.mean(X, axis=0), full_matrices=False)
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

    init_alpha = random.dirichlet(
        alpha_key, alpha=jnp.ones(p_dim), shape=(l_dim, z_dim)
    )
    if A is not None:
        theta = random.normal(theta_key, shape=(m, z_dim))
        pi = _compute_pi(A, theta)
    else:
        theta = None
        pi = jnp.ones(p_dim) / p_dim

    return ModelParams(
        init_mu_z,
        init_var_z,
        init_mu_w,
        init_var_w,
        init_alpha,
        tau,
        tau_0,
        theta=theta,
        pi=pi,
    )


def _check_args(X: ArrayLike, A: ArrayLike, z_dim: int, l_dim: int, init: _init_type):
    # pull type options for init
    type_options = get_args(_init_type)

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
    if A is not None:
        if len(A.shape) != 2:
            raise ValueError(
                f"Dimension of annotation matrix A should be 2: received {len(A.shape)}"
            )
        a_p_dim, _ = A.shape
        if a_p_dim != p_dim:
            raise ValueError(
                f"Leading dimension of annotation matrix A should match feature dimension {p_dim}: received {a_p_dim}"
            )
        if jnp.any(jnp.isnan(A)):
            raise ValueError(
                "A contains 'nan'. Please check input data for correctness or missingness"
            )
        if jnp.any(jnp.isinf(A)):
            raise ValueError(
                "A contains 'inf'. Please check input data for correctness or missingness"
            )
    # type check for init
    if init not in type_options:
        raise ValueError(
            f"Unknown initialization provided '{init}'; Choices: {type_options}"
        )

    return


def susie_pca(
    X: ArrayLike,
    z_dim: int,
    l_dim: int,
    A: Optional[ArrayLike] = None,
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
        X: Input data. Should be an array-like
        z_dim: Latent factor dimension (int; K)
        l_dim: Number of single-effects comprising each factor (int; L)
        A: Annotation matrix to use in parameterized-prior mode. If not `None`, leading dimension
            should match the feature dimension of X.
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
        and `P` features (:py:obj:`jax.numpy.ndarray`).

    Raises:
        ValueError: Invalid `l_dim` or `z_dim` values. Invalid initialization scheme.
        Data `X` contains `inf` or `nan`. If annotation matrix `A` is not `None`, raises
        if `A` contains `inf`, `nan` or does not match feature dimension with `X`.
    """

    # cast to jax array
    X = jnp.asarray(X)

    # sanity check arguments
    _check_args(X, A, z_dim, l_dim, init)

    # option to center the data
    if center:
        X -= jnp.mean(X, axis=0)
        X /= jnp.std(X, axis=0)

    # initialize PRNGkey and params
    rng_key = random.PRNGKey(seed)
    params = _init_params(rng_key, X, A, z_dim, l_dim, tau, init)

    #  core loop for inference
    elbo = -5e25
    for idx in range(1, max_iter + 1):
        if A is not None:
            elbo_res, params = _annotation_inner_loop(X, A, params)
        else:
            elbo_res, params = _inner_loop(X, params)

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

    # compute PVE and reorder in descending value
    pve = compute_pve(params)
    params, pve = _reorder_factors_by_pve(A, params, pve)

    # compute PIPs
    pip = compute_pip(params)

    return SuSiEPCAResults(params, elbo_res, pve, pip)
