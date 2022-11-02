
import numpy as np
import pytest as pt

import susiepca as sp


def test_mse():
    X = None #
    Xhat = None #
    expected_res = None
    actual_res = sp.metrics.mse(X, Xhat)
    assert pt.approx(expected_res) == actual_res