
import numpy as np

def check1(x):
    if x is None:
        return False
    Z = np.zeros((8,8),dtype=int)
    Z[1::2,::2] = 1
    Z[::2,1::2] = 1
    # Z = np.tile( np.array([[0,1],[1,0]]), (4,4))
    return np.array_equal(x, Z)

def check2(x, y):
    if x is None:
        return False
    np.random.seed(25)
    Z = np.random.random((10,10))
    Zmin, Zmax = Z.min(), Z.max()
    return Zmin == x and Zmax == y

def check3(x):
    Z = np.tile( np.array([[0,1],[1,0]]), (4,4))
    return np.array_equal(x, Z)
    
def check4(x):
    if x is None:
        return False
    np.random.seed(30)
    Z = np.random.random((5,5))
    Zmax,Zmin = Z.max(), Z.min()
    Z = (Z - Zmin)/(Zmax - Zmin)
    return np.array_equal(x, Z)

def check5(x):
    if x is None:
        return False
    Z = np.dot(np.ones((5,3)), np.ones((3,2)))
    return np.array_equal(x, Z)

def check5(x, case):
    if x is None:
        return False
    right = np.nonzero(case)
    return np.array_equal(x, right)

def check6(x):
    if x is None:
        return False
    Z = np.zeros((5,5))
    Z += np.arange(5)
    return np.array_equal(x, Z)

def check7(x):
    if x is None:
        return False
    right = np.linspace(0,1,12,endpoint=True)[1:-1]
    return np.array_equal(x, right)

def check8(x):
    if x is None:
        return False
    np.random.seed(30)
    Z = np.random.random(10)
    Z.sort()
    return np.array_equal(x, Z)    

def check9(e, a, b):
    t = np.allclose(a,b)
    return t == e
    
def check10(Z, avg):
    m = Z.mean()
    return avg == m
    
    
if __name__ == "__main__":
    
    right = np.arange(9).reshape(3,3)
    print right
    assert check4(right)
    