import pytest

import jax.numpy as jnp

import susiepca as sp


# define the test for MSE
def test_mse():
    X = jnp.array([[-0.58, -0.43, 0.70], [-0.50, -1.22, 0.91]])

    Xhat = jnp.array([[-0.90, -0.50, 0.40], [-1.17, -1.47, 0.91]])

    Xhat_wrongshape = jnp.array(
        [[-0.90, -0.50, 0.40], [-1.17, -1.47, 0.91], [-0.32, 0.52, 1.36]]
    )

    expected_res = 0.1980826
    actual_res = float(sp.metrics.mse(X, Xhat))

    assert pytest.approx(expected_res) == actual_res

    with pytest.raises(ValueError):
        sp.metrics.mse(X, Xhat_wrongshape)


# define the test for credible set
# l_dim = 2,z_dim = 2,p_dim = 4


def test_get_credset():
    alpha = jnp.array(
        [
            [
                [0.90, 0.05, 0.02, 0.03],
                [0.30, 0.40, 0.21, 0.09],
            ],
            [
                [0.00, 0.50, 0.50, 0.00],
                [0.003, 0.003, 0.004, 0.99],
            ],
        ]
    )
    l_dim, z_dim, p_dim = alpha.shape
    assert pytest.approx(jnp.sum(alpha, axis=-1)) == jnp.ones((l_dim, z_dim))

    # test get_cred_set
    set1 = sp.metrics.get_credset(alpha)

    # first single effect in first factor
    assert int(set1["z0"][0][0]) == 0
    # second single effect in first factor
    assert int(set1["z0"][1][0]) == 1
    assert int(set1["z0"][1][1]) == 2
    # first single effect in second factor
    assert int(set1["z1"][0][0]) == 1
    assert int(set1["z1"][0][1]) == 0
    assert int(set1["z1"][0][2]) == 2
    # second single effect in second factor
    assert int(set1["z1"][1][0]) == 3

    return
