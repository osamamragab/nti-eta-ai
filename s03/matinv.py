import numpy as np


def canInv(mat):
    return mat.shape[0] == mat.shape[1] and np.linalg.det(mat)

mat = np.array([[1,2,3], [1,2,3], [1,2,3]])
print(canInv(mat))
