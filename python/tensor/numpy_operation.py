import numpy as np


def check_diag(x):
    np.random.seed(10)
    tensor = np.random.random((32, 10, 10))
    solution = tensor[np.arange(32), np.arange(10), np.arange(10)] = 1
    assert np.allclose(x, tensor)

