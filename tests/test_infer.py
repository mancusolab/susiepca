import pytest

import jax.experimental.sparse as sparse
import jax.numpy as jnp
import jax.random as rdm

import susiepca as sp
import susiepca.common


params = susiepca.common.ModelParams(
    mu_z=jnp.array(
        [[1.5584918, -0.75374484], [1.1176406, 0.17358227], [-2.3289988, 0.59763706]]
    ),
    var_z=jnp.array([[0.20242101, 0.0], [0.0, 2.90467]]),
    mu_w=jnp.array(
        [
            [
                [
                    -1.1549549e-03,
                    2.5680402e-04,
                    -1.6877083e-03,
                    -9.1297348e-04,
                    1.7507891e-03,
                ],
                [
                    3.9870795e-04,
                    -6.7456358e-04,
                    1.7658224e-03,
                    -2.4815777e-04,
                    4.7399240e-04,
                ],
            ],
            [
                [
                    2.8350489e-04,
                    1.4239480e-03,
                    9.6746330e-04,
                    5.1718129e-05,
                    9.5367047e-04,
                ],
                [
                    -9.1759655e-05,
                    -1.1265561e-03,
                    4.5779414e-04,
                    -1.0159041e-03,
                    -2.0231635e-03,
                ],
            ],
        ]
    ),
    var_w=jnp.array([[0.09615356, 1.3323052], [0.1007237, 0.4017519]]),
    alpha=jnp.array(
        [
            [
                [0.26244566, 0.09673341, 0.22945851, 0.38307405, 0.02828841],
                [0.04656444, 0.28155947, 0.18396974, 0.45929772, 0.02860874],
            ],
            [
                [0.2513191, 0.08301223, 0.08566361, 0.47507766, 0.10492733],
                [0.11461553, 0.06789416, 0.43151987, 0.03160023, 0.35437024],
            ],
        ]
    ),
    tau=1,
    tau_0=jnp.array([[1.0, 1.0], [1.0, 1.0]]),
    theta=None,
    pi=jnp.array([0.2, 0.2, 0.2, 0.2, 0.2]),
)


def test_compute_pve():
    assert params.mu_w.shape == params.alpha.shape


def test_compute_pip():
    assert params.mu_w.shape == params.alpha.shape


def test_susie_pca():
    l_dim = 2
    z_dim = 2

    X = jnp.array([[1, 2, 3, 4, 5], [2, 8, 1, 2, 1], [0, 1, 3, 4, 1]])
    X_wrongshape = jnp.array(
        [[[1, 2, 3, 4, 5], [2, 8, 1, 2, 1]], [[0, 1, 3, 4, 1], [1, 5, 8, 3, 1]]]
    )
    X_nan = jnp.array([[1, 2, 3, 4, jnp.nan], [2, 8, 1, 2, 1], [0, 1, 3, 4, 1]])
    X_inf = jnp.array([[1, 2, 3, 4, jnp.inf], [2, 8, 1, 2, 1], [0, 1, 3, 4, 1]])
    # test X with wrong shape
    with pytest.raises(ValueError):
        sp.infer.susie_pca(X_wrongshape, z_dim, l_dim)
        # test l_dim > p_dim
        sp.infer.susie_pca(X, z_dim, l_dim=6)
        # test l_dim<=0
        sp.infer.susie_pca(X, z_dim, l_dim=0)
        # test z_dim>p_dim
        sp.infer.susie_pca(X, z_dim=6, l_dim=l_dim)
        # test z_dim>p_dim
        sp.infer.susie_pca(X, z_dim=3, l_dim=l_dim)
        # test z_dim<=0
        sp.infer.susie_pca(X, z_dim=0, l_dim=l_dim)
        # test X contain nan/inf
        sp.infer.susie_pca(X_nan, z_dim, l_dim=l_dim)
        sp.infer.susie_pca(X_inf, z_dim, l_dim=l_dim)
        # test wrong init method
        sp.infer.susie_pca(X, z_dim, l_dim, init="not sure")


def test_susie_pca_sparse():
    n_dim = 50
    p_dim = 200
    l_dim = 5
    z_dim = 5

    key = rdm.PRNGKey(0)
    key, i_key = rdm.split(key)

    X = sparse.random_bcoo(i_key, shape=(n_dim, p_dim))

    res = sp.infer.susie_pca(X, z_dim, l_dim)
    assert res is not None
