
import numpy as np

def check1(x):
    if x is None:
        return False
    right = np.zeros(len(x))
    return np.array_equal(x, right)

def check2(x):
    if x is None:
        return False
    right = np.zeros(len(x))
    right[4] = 1
    return np.array_equal(x, right)

def check3(x):
    if x is None:
        return False
    right = np.arange(10, 50)
    return np.array_equal(x, right)

def check4(x):
    if x is None:
        return False
    right = np.arange(9).reshape(3,3)
    return np.array_equal(x, right)

def check5(x, case):
    if x is None:
        return False
    right = np.nonzero(case)
    return np.array_equal(x, right)

def check6(x):
    if x is None:
        return False
    right = np.eye(3)
    return np.array_equal(x, right)

def check7(x):
    if x is None:
        return False
    right = np.diag(1 + np.arange(4), k=-1)
    return np.array_equal(x, right)

def check8(x, dim):
    if x is None:
        return False
    np.random.seed(2)
    right = np.random.random(dim)
    return np.array_equal(x, right)    
    
if __name__ == "__main__":
    
    right = np.arange(9).reshape(3,3)
    print right
    assert check4(right)
    