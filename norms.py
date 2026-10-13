"""Exercise Sheet P1, Task 1: p-Norms.

Fill in the TODOs. Do not add any imports and do not change the function signatures.
"""
import numpy as np

def p_norm(x, p):
    """p-norm of a vector x of shape (D,) (Task 1.2).

    p is either a number p >= 1 or np.inf (max norm).
    Do not use np.linalg.norm.
    """
    if p < 1:
        raise ValueError("p_norm is only defined for p>=1")
    if p == np.inf:
        return np.max(np.abs(x))
    return np.sum(np.abs(x) ** p) ** (1 / p)
    
    raise NotImplementedError("Task 1.2: p_norm")