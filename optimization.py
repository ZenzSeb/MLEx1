"""Exercise Sheet P1, Task 3: Optimization.

Fill in the TODOs. Do not add any imports and do not change the function signatures.
"""
import numpy as np


def f(x):
    """f(x) = x^4 - 2 x^2 (Task 3.2). x can be a float or a NumPy array (evaluate element-wise)."""
    return x**4 - 2 * x**2
    raise NotImplementedError("Task 3.2: f")


def df(x):
    """Derivative f'(x) that you computed in Task 3.1 (Task 3.2). x can be a float or a NumPy array."""
    return 4 * x**3 - 4 * x
    raise NotImplementedError("Task 3.2: df")


def gradient_descent(grad, x0, learning_rate, eps=1e-6, max_iters=200):
    """Gradient descent with a constant learning rate, starting at the float x0 (Task 3.3).

    grad maps a point x to the derivative at x.
    Stops as soon as |grad(x^(t))| <= eps or after max_iters steps.
    Returns all iterates x^(0), ..., x^(T) as an array of shape (T + 1,), with T <= max_iters.
    """
    #init
    x = x0
    iterations = 0
    trajectory = [x0]

    while iterations < max_iters and abs(grad(x)) > eps:
        x = x - learning_rate * grad(x) # x(t+1) = x(t) - eta*df(x(t))
        trajectory.append(x)
        iterations += 1
    return np.array(trajectory)
    raise NotImplementedError("Task 3.3: gradient_descent")
