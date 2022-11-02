import numpy as np
import pytest

import susiepca as sp


# define the test for MSE
@pytest.mark.parametrize("seed", 0)
def test_mse(seed):
    np.random.seed(seed)
    X = np.random.normal(0, 1, size=(3, 5))
    Xhat = X + np.random.normal(0, 0.5, (3, 5))
    expected_res = 0.11514
    actual_res = sp.metrics.mse(X, Xhat)
    assert X.shape == Xhat.shape
    assert pytest.approx(expected_res) == actual_res

    # with pytest.raises(Exception):


# define the test for credible set
# def test_get_credset():
