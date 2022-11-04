import numpy as np
import pytest

import susiepca as sp


# define the test for MSE
def test_mse():
    X = np.array([[-0.58, -0.43, 0.70], [-0.50, -1.22, 0.91]])

    Xhat = np.array([[-0.90, -0.50, 0.40], [-1.17, -1.47, 0.91]])

    Xhat_wrongshape = np.array(
        [[-0.90, -0.50, 0.40], [-1.17, -1.47, 0.91], [-0.32, 0.52, 1.36]]
    )

    expected_res = 0.1981
    actual_res = sp.metrics.mse(X, Xhat)

    assert pytest.approx(expected_res) == actual_res

    with pytest.raises(ValueError):
        sp.metrics.mse(X, Xhat_wrongshape)


# define the test for credible set
# def test_get_credset():
