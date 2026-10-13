"""Exercise Sheet P1 -- Mathematical Preliminaries: p-Norms, Linear Algebra and Optimization.

Run all tasks:            python main.py
Run only selected tasks:  python main.py 1 3
Plots are saved to the folder plots/.

Fill in the TODOs. Do not add any imports.
"""
import sys

import numpy as np

from linear_algebra import dot, matmul
from norms import p_norm
from optimization import f, df, gradient_descent
from plot_utils import plot_norms_over_p, plot_function_and_derivative, plot_trajectories, plot_over_iterations

SEED = 2026  # do not change


def task_1(rng):
    x = rng.normal(size=5)
    for p in [1, 2, 3, np.inf]:
        print(f"p_norm (p = {p}) matches np.linalg.norm:", np.allclose(p_norm(x, p), np.linalg.norm(x, ord=p)))

    x = np.array([3.0, -4.0])
    # TODO: print ||x||_1, ||x||_2 and ||x||_inf and compare them with your results from Task 1.1

    vectors = {
        r"$\mathbf{x}_1 = (3, -4)^T$": np.array([3.0, -4.0]),
        r"$\mathbf{x}_2 = (1, 1, 1, 1)^T$": np.array([1.0, 1.0, 1.0, 1.0]),
        r"$\mathbf{x}_3 = (0, 2, 0)^T$": np.array([0.0, 2.0, 0.0]),
    }
    plot_norms_over_p(p_norm, vectors, np.arange(1, 21))


def task_2(rng):
    C = np.array([[1.0, 2.0, 0.0], [2.0, 1.0, 3.0], [3.0, 3.0, 3.0]])
    # TODO: print the rank of C (np.linalg.matrix_rank) and compare it with your result from Task 2.1

    x, y = np.array([3.0, 1.0, 2.2]), np.array([4.0, 0.0, -1.0])
    print("dot(x, y) =", dot(x, y), " matches np.dot:", np.allclose(dot(x, y), np.dot(x, y)))

    A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    B = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print("A B =\n", matmul(A, B), "\nmatches np.matmul:", np.allclose(matmul(A, B), np.matmul(A, B)))

    # TODO: compare your matmul with np.matmul for random matrices of shape (3, 5) and (5, 2) (use rng.normal)
    # TODO: use your matmul to check, for two random 3 x 3 matrices A and B, whether AB = BA and
    #       whether (AB)^T = B^T A^T


def task_3(rng):
    plot_function_and_derivative(f, df)

    trajectories = {}
    for x0 in [-1.5, -0.3, 0.0, 0.3, 1.5]:
        trajectories[f"$x^{{(0)}} = {x0}$"] = gradient_descent(df, x0, learning_rate=0.05)
    # TODO: print the number of iterations and the final iterate x_T for every initial point
    plot_trajectories(f, trajectories, filename="3_trajectories_initial_points.pdf")

    trajectories, function_values = {}, {}
    for learning_rate in [0.01, 0.05, 0.15, 0.3]:
        label = rf"$\eta = {learning_rate}$"
        trajectories[label] = gradient_descent(df, 1.5, learning_rate)
        function_values[label] = f(trajectories[label])
    # TODO: print the number of iterations and the final iterate x_T for every learning rate
    plot_trajectories(f, trajectories, filename="3_trajectories_learning_rates.pdf")
    plot_over_iterations(function_values, ylabel=r"$f(x^{(t)})$")


TASKS = {"1": task_1, "2": task_2, "3": task_3}


def main():
    selected = sys.argv[1:] or list(TASKS)
    for name in selected:
        if name not in TASKS:
            sys.exit(f"Unknown task '{name}', choose from: {', '.join(TASKS)}")
        print(f"\n========== Task {name} ==========")
        TASKS[name](np.random.default_rng(SEED))  # every task gets the same seed, so tasks can be run separately


if __name__ == "__main__":
    main()
