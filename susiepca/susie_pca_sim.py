#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import argparse as ap
import logging
import sys
import typing

import jax
import jax.numpy as jnp
import numpy as np
import jax.scipy.special as spec

from jax.config import config
from jax import jit, nn, random
from jax.random import PRNGKey

import pandas as pd
import scipy
import scipy.optimize as sopt
from scipy.stats import rankdata
from csv import writer
from sklearn.decomposition import SparsePCA
import procrustes 

def logdet(A):
    sign, ldet = jnp.linalg.slogdet(A)
    return ldet

def compute_effect(S, B):
    # S.shape = (p_dim, l_dim, k_dim); B.shape = (p_dim, l_dim);
    # but we only want to sum over l_dim
    W = jnp.einsum("klp,kl->kp", S, B)
    return W

#simulation data
def simulation_order(rng_key, n_dim =200, p_dim =320, z_dim = 4):
    rng_key, z_key, b_key, obs_key = random.split(rng_key, 4)

    
    Z = random.normal(z_key, shape=(n_dim, z_dim))
    l_dim=int(p_dim/z_dim)
    #effects
    w1_nonzero = 1 * random.normal(b_key, shape=(l_dim,))
    w1_rest = jnp.zeros(shape=(p_dim-l_dim,))
    w1 = jnp.concatenate((w1_nonzero,w1_rest))

    w2_nonzero = 1 * random.normal(b_key, shape=(l_dim,))
    w2_rest = jnp.zeros(shape=(p_dim-2*l_dim,))
    w2 = jnp.concatenate((jnp.zeros(shape=(l_dim,)),w2_nonzero,w2_rest))


    w3_nonzero = 2 * random.normal(b_key, shape=(l_dim,))
    w3_rest = jnp.zeros(shape=(p_dim-3*l_dim,))
    w3 = jnp.concatenate((jnp.zeros(shape=(2*l_dim,)),w3_nonzero,w3_rest))

    w4_nonzero = 1 * random.normal(b_key, shape=(l_dim,))
    w4_rest = jnp.zeros(shape=(p_dim-l_dim,))
    w4 = jnp.concatenate((w4_rest,w4_nonzero))

    W = jnp.vstack((w1,w2,w3,w4))

    m = Z @ W

    X = m + random.normal(obs_key, shape=(n_dim, p_dim))

    return Z, W, X, m



class ModelParams(typing.NamedTuple):
    # variational params for Z
    mu_z: jnp.ndarray
    var_z: jnp.ndarray

    # variational params for W given Gamma
    mu_w: jnp.ndarray
    var_w: jnp.ndarray

    # variational params for Gamma
    alpha: jnp.ndarray

    # residual precision param
    tau: jnp.ndarray
    tau_0: jnp.ndarray

    # prior probability for gamma
    pi: jnp.ndarray


def init_params(rng_key, n_dim, p_dim, z_dim, l_dim):

    tau = 10
    tau_0 = jnp.ones((l_dim,z_dim))

    rng_key, mu_key, var_key = random.split(rng_key, 3)
    init_mu_z = random.normal(mu_key, shape=(n_dim, z_dim))
    init_var_z = jnp.diag(random.normal(var_key, shape=(z_dim,)) ** 2)

    rng_key, mu_key, var_key = random.split(rng_key, 3)
    init_mu_w = random.normal(mu_key, shape=(l_dim, z_dim, p_dim))*1e-3

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
    l_dim, z_dim, p_dim = params.mu_w.shape

    # tr(V[w_k]) = sum_l tr(V[w_kl]) = sum_l sum_i [(E[w_kl^2 | gamma_kli = 1] * E[gamma_kl]) + (E[w_kl | gamma_kli = 1] * E[gamma_kl]) ** 2
    # mu**2 * a - mu**2 * a * a = mu**2 * a * (1 - a) = mu**2 * var[a]
    # V[w_kl] = var[w] * E[a] + E[w]*var[a]
    # V[Y] = E_X[V[Y | X]] + V_X[E[Y | X]]
    trace_var = jnp.sum(
            params.var_w[:,:,jnp.newaxis] * params.alpha
            + (params.mu_w ** 2 * params.alpha * (1 - params.alpha)),
            axis=(-1, 0))

    E_W = jnp.sum(params.mu_w * params.alpha, axis=0)
    E_WW = E_W @ E_W.T + jnp.diag(trace_var)

    return E_W, E_WW


# update w
def update_w(RtZk, E_zzk, params, k, l):
    n_dim, z_dim = params.mu_z.shape

    # calculate update_var_w as the new V[w | gamma]
    # suppose indep between w_k
    update_var_wkl = jnp.reciprocal(params.tau * E_zzk + params.tau_0[l,k])

    # calculate update_mu_w as the new E[w | gamma]
    update_mu_wkl = params.tau * update_var_wkl * RtZk

    return params._replace(
        mu_w=params.mu_w.at[l, k].set(update_mu_wkl),
        var_w=params.var_w.at[l, k].set(update_var_wkl),
    )


def update_alpha(X, params, k, l):

    eps = 1e-8
    """
    update_alpha_kl = nn.softmax(
        jnp.log(params.pi) +
        0.5 * jnp.log(update_var_wkl) +
        0.5 * (update_mu_wkl ** 2) / update_var_wkl
    )
    """
    # help with numerical issues that forced 0/1 entries...
    log_alpha = jnp.log(params.pi) + 0.5 * jnp.log(params.var_w[l, k]) + 0.5 * (params.mu_w[l, k] ** 2) / params.var_w[l, k]
    alpha_kl = jnp.exp(log_alpha - spec.logsumexp(log_alpha) - eps)
    #alpha_kl = nn.softmax(log_alpha)

    # update into correspond location
    params = params._replace(
            alpha=params.alpha.at[l, k].set(alpha_kl),
            )

    return params

def log_bf_np(z, s2, s0):
    return 0.5 * (jnp.log(s2) - jnp.log(s2 + 1 / s0)) + 0.5 * z ** 2 * (
        (1 / s0) / (s2 + 1 / s0)
    )


def update_tau0(RtZk, E_zzk, params, k, l):
    Z_s = (RtZk / E_zzk) * jnp.sqrt(E_zzk * params.tau)
    s2_s = 1 / (E_zzk * params.tau)
    s20_s = params.tau_0[l, k]

    def min_obj(log_s20_s):
        return -spec.logsumexp(
            jnp.log(params.pi) + log_bf_np(Z_s, s2_s, jnp.exp(log_s20_s))
        )

    res = sopt.minimize_scalar(min_obj, method="bounded", bounds=(-30, 10))
    new_s20_s = jnp.exp(res.x)
    params = params._replace(tau_0=params.tau_0.at[l, k].set(new_s20_s))

    return params


def update_tau0_mle(params):
    l_dim, z_dim, p_dim = params.mu_w.shape
    
    est_varw = params.mu_w ** 2 + params.var_w[:,:,jnp.newaxis]
    
    u_tau_0 = jnp.sum(params.alpha,axis = -1)/ jnp.sum(est_varw * params.alpha,axis = -1)
    
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
    E_ss = jnp.sum(X ** 2) - 2 * jnp.trace(E_W @ X.T @ params.mu_z) + jnp.trace(E_ZZ @ E_WW)
    u_tau = (n_dim * p_dim) / E_ss

    return params._replace(tau=u_tau)




def compute_elbo(X, params, idx):
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
    klw_term1 = params.tau_0[:, :, jnp.newaxis] * (params.var_w[:, :, jnp.newaxis] + params.mu_w ** 2)
    klw_term2 = klw_term1 - 1.0 - (jnp.log(params.tau_0) + jnp.log(params.var_w))[:,:,jnp.newaxis]
    negKL_w = -0.5 * jnp.sum(params.alpha * klw_term2)

    # neg-KL for gamma
    negKL_gamma = -jnp.sum(
        params.alpha * (jnp.log(params.alpha + 1e-10) - jnp.log(params.pi))
    )

    elbo = E_ll + negKL_z + negKL_w + negKL_gamma

    print(
        f"Iter={idx},ELBO = {elbo} | E_ll = {E_ll} | -KL[Z] = {negKL_z} | -KL[W] = {negKL_w} | -KL[G] = {negKL_gamma}"
    )

    return elbo

def compute_pip(params):

    pip = 1-jnp.prod(1-params.alpha,axis = 0)

    return pip

def get_credset(params, rho=0.9):
    l_dim,z_dim, p_dim = params.alpha.shape
    idxs = jnp.argsort(-params.alpha,axis=-1)
    cs={}
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
                local += params.alpha[ldx,zdx,idx]
        cs["z"+str(zdx)] = cs_s

    return cs



def compute_largest_pip(pip,n,absol = False):
    z_dim,n_dim = pip.shape
    pip_df = pd.DataFrame(pip.T)
    
    credible = {}
    credible_index = []
    
    for k in range(z_dim):
        if absol == True:
            credible["z"+str(k)] = pip_df.abs().nlargest(n,k)[k]
            credible_index.append(pip_df.abs().nlargest(n,k)[k].index)
        else:
            credible["z"+str(k)] = pip_df.nlargest(n,k)[k]
            credible_index.append(pip_df.nlargest(n,k)[k].index)
        
    return credible,credible_index


def find_match2(params,Z_real,pip,proc_trans,spca_trans):
    p_dim,real_z_dim = Z_real.shape
    l_dim,z_dim, p_dim = params.alpha.shape
    
    pip_trans = jnp.abs((np.asarray(pip.T) @ proc_trans.t).T)
    spca_set,spca_index = compute_largest_pip(spca_trans, n=l_dim,absol=True)
    
    count8 = 0
    count9 = 0
    count_spca8=0
    count_spca9=0
    
    spca_mean=jnp.mean(spca_trans,axis=1)
    spca_std = jnp.std(spca_trans,axis=1)
    
    for real_factor_idx in range(real_z_dim):
        set8 = jnp.where(pip_trans[real_factor_idx,:] > 0.90)
        set9 = jnp.where(pip_trans[real_factor_idx,:] > 0.95)
        spca_set8 = jnp.where(jnp.abs(spca_trans[real_factor_idx,:]) > spca_mean[real_factor_idx]+ 1.64*spca_std[real_factor_idx])
        spca_set9 = jnp.where(jnp.abs(spca_trans[real_factor_idx,:]) > spca_mean[real_factor_idx]+ 1.96*spca_std[real_factor_idx])
        
        real = jnp.array(range(real_factor_idx*50,(1+real_factor_idx)*50,1))

        count8 += len(jnp.intersect1d(set8,real))
        count9 += len(jnp.intersect1d(set9,real))
        
        
        count_spca8 += len(jnp.intersect1d(spca_set8,real))
        count_spca9 += len(jnp.intersect1d(spca_set9,real))
    
    return count8,count9,count_spca8,count_spca9

def compute_pve(mu_z,W,tau):
    n_dim, z_dim = mu_z.shape
    z_dim, p_dim = W.shape    
    
    sk = jnp.zeros(z_dim)
    for k in range(z_dim):
        sk = sk.at[k].set(jnp.sum((mu_z[:,k,jnp.newaxis] * W[k,:]) ** 2))

    s = jnp.sum(sk)       
    pve = sk/(s + p_dim * n_dim * (1/tau))
    return pve

def mySparsePCA(X,n_components,seed):
    
    spca = SparsePCA(n_components=n_components,random_state=seed)
    spca_z = spca.fit_transform(X)
    spca_weights = spca.components_
    
    return spca_z,spca_weights




def main(args):
    argp = ap.ArgumentParser(description="")
    argp.add_argument("--seed", default=1, type=int,help="Random seed")
    argp.add_argument(
        "--n-dim", default=180, type=int, help="Number of samples"
    )
    argp.add_argument(
        "--p-dim", default=200, type=int, help="Number of features"
    )
    argp.add_argument(
        "--max-iter", default=60, type=int, help="Maximum number of iterations for VI"
    )
    argp.add_argument(
        "--z-dim", default=4, type=int, help="Number of latent factors"
    )
    argp.add_argument(
        "--real-z-dim", default=4, type=int, help="Number of real latent factors"
    )
    argp.add_argument(
        "--l-dim", default=50, type=int, help="Number of single effects"
    )
    argp.add_argument(
        "--real-l-dim", default=50, type=int, help="Number of real single effects"
    )
    argp.add_argument(
        "--tol", default=1e-3, type=float, help="Delta ELBO threshold to stop VI"
    )
    argp.add_argument("-o", "--output", type=str, default="/Users/dong/Documents/Project/SuSiE/SuSiEPCA/simulation_results")

    args = argp.parse_args(args)

    config.update("jax_enable_x64", True)

    rng_key = random.PRNGKey(args.seed)

    # initialize Z, W, and gamma

    rng_key, sim_key = random.split(rng_key, 2)
 
    Z_real,W_real,X,m = simulation_order(rng_key, args.n_dim, args.p_dim, args.real_z_dim)

    rng_key, init_key = random.split(rng_key, 2)
    params = init_params(init_key, args.n_dim, args.p_dim, args.z_dim, args.l_dim)

    # run inference
    elbo = -5e25
    kdxs = jnp.ones((args.z_dim,), dtype=bool)
    for idx in range(1, args.max_iter + 1):

        # compute expected residuals
        # use posterior mean of Z, W, and Alpha to calculate residuals
        W = jnp.sum(params.mu_w * params.alpha, axis=0)
        E_ZZ = params.mu_z.T @ params.mu_z + args.n_dim * params.var_z
        params = update_tau0_mle(params)
        
        # update locals (W, Gamma)
        for k in range(args.z_dim):
            kdxs = kdxs.at[k].set(False)
            E_zpzk = E_ZZ[k, kdxs]
            E_zzk = E_ZZ[k, k]
            RtZk = params.mu_z[:,k] @ X - W[kdxs,:].T @ E_zpzk
            Wk = W[k,:]
            
            
            for l in range(args.l_dim):
                Wkl = Wk - (params.mu_w[l, k] * params.alpha[l, k])
                E_RtZk = RtZk - E_zzk * Wkl

                #params = update_tau0(E_RtZk, E_zzk, params, k, l)
                params = update_w(E_RtZk, E_zzk, params, k, l)
                params = update_alpha_bf(E_RtZk, E_zzk, params, k, l)
                #params = update_alpha(E_RtZk, params, k, l)
                Wk = Wkl + (params.mu_w[l, k] * params.alpha[l, k])

            kdxs = kdxs.at[k].set(True)
            
        params = update_z(X, params)
        params = update_tau(X, params)


        # compute elbo
        elbo_tmp = compute_elbo(X, params,idx)

        diff = (elbo_tmp - elbo)
        #print(f"iter = {idx}, elbo = {elbo_tmp}")
        if diff < 0:
            print(
                f"Bug alert! Diff between elbo[{idx - 1}] and elbo[{idx}] = {diff}"
            )
        if diff < args.tol:
            print(f"Elbo diff tolerance reached at iteration {idx}")
            break
        if idx == args.max_iter:
            print("Maximum Iteration reached")
        elbo = elbo_tmp
        

    pip = compute_pip(params)
    pip_set,credible_index = compute_largest_pip(pip,n = args.l_dim)
    pve = compute_pve(params.mu_z,W,params.tau)
    print(f"Percent of variance explained is {pve}")
    
    #total_match,trans_susie = find_match(params,Z_real,S_real,credible_index)
    #cs = get_credset(params, rho=0.9)
    #dist_susie = W.T @ trans_susie - W_real.T
    #err_susie = jnp.sqrt(jnp.trace(dist_susie @ dist_susie.T))
    #summary_results(params,S_real,total_match,args.output)
    
    #Procruste Transformation
    proc_trans_susie = procrustes.orthogonal(np.asarray(W.T),np.asarray(W_real.T),scale = True)
    
    err_susie = proc_trans_susie.error
   
    W_trans = proc_trans_susie.new_a.T
    
    
    reconstruct_X = params.mu_z @ W
    #rmse
    err_susie2 = ((reconstruct_X-X)**2).sum() / (X**2).sum() 
        
    print("Begin inference using Sparse PCA")
    spca_z,spca_weights = mySparsePCA(X, args.z_dim, args.seed)
    
    
    #Procruste transformation
    proc_trans_spca = procrustes.orthogonal(np.asarray(spca_weights.T),np.asarray(W_real.T),scale=True)  
    spca_trans = (np.asarray(spca_weights.T) @ proc_trans_spca.t).T
    err_spca = proc_trans_spca.error
    
    reconstruct_spca_X = spca_z @ spca_weights
    #rmse
    err_spca2 = ((reconstruct_spca_X-X)**2).sum() /(X**2).sum() 
    
    import pdb;pdb.set_trace()
    
 
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


