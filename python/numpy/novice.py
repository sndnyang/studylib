
import numpy as np

def check1(x):
    if x is None:
        return False
    # Z.flags.writeable = False
    try:
        x[0] = 1:
    except:
        return True
    return False   

def check9(Z, a, b):
    X,Y = Z[:,0], Z[:,1]
    R = np.sqrt(X**2+Y**2)
    T = np.arctan2(Y,X)

    return np.allclose(R, a) and np.allclose(T, b)
    
def check10(Z, avg):
    m = Z.mean()
    return avg == m
    
    
if __name__ == "__main__":
    
    right = np.arange(9).reshape(3,3)
    print right
    assert check4(right)
    