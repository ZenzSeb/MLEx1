"""Exercise Sheet P1, Task 2: Linear Algebra.

Fill in the TODOs. Do not add any imports and do not change the function signatures.
"""
import numpy as np

def dot(x, y):
    """Dot product x^T y of two vectors x, y of shape (D,) (Task 2.2).

    Do not use np.dot, np.inner, np.matmul or the @ operator.
    """
    dotproduct = 0.0

    if np.size(x) != np.size(y):
        raise ValueError("Dot product requires vectors of the same dimension!")

    for i in range(np.size(x)):
        dotproduct += x[i]*y[i]

    return(dotproduct)   

    raise NotImplementedError("Task 2.2: dot")


def matmul(A, B):
    """Matrix-matrix product C = A B for A of shape (M, N) and B of shape (N, K) (Task 2.2).

    Entry C[i, j] is the dot product of row i of A and column j of B -- use your dot.
    Do not use np.dot, np.matmul or the @ operator.
    """
    m = np.size(A,0)      # m rows
    n = np.size(B,1)      # n cols
    C = np.zeros((m,n))   # init m x n zero matrix

    for Ai in range(m):
        for Bi in range(n):
            C[Ai,Bi]=dot(A[Ai,:],B[:,Bi])

    return C
    raise NotImplementedError("Task 2.2: matmul")