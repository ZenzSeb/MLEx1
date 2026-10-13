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
    print("=== Test 1.1: compare p_norm with np.linalg.norm for a random vector (size=5) for p in(1,2,3,inf) ===")

    for p in [1, 2, 3, np.inf]:
        print(f"p_norm (p = {p}) matches np.linalg.norm:", np.allclose(p_norm(x, p), np.linalg.norm(x, ord=p)))

    x = np.array([3.0, -4.0])
    print(f"=== Test 1.2: evaluate p_norm(x,p) for vector x=({x[0]},{x[1]}) and p in(1,2,inf) ===") 

    for p in [1, 2, np.inf]:
            print(f"p_norm (p = {p}) of x: {p_norm(x,p)}")

    vectors = {
        r"$\mathbf{x}_1 = (3, -4)^T$": np.array([3.0, -4.0]),
        r"$\mathbf{x}_2 = (1, 1, 1, 1)^T$": np.array([1.0, 1.0, 1.0, 1.0]),
        r"$\mathbf{x}_3 = (0, 2, 0)^T$": np.array([0.0, 2.0, 0.0]),
    }
    plot_norms_over_p(p_norm, vectors, np.arange(1, 21))


def task_2(rng):

    C = np.array([[1.0, 2.0, 0.0], [2.0, 1.0, 3.0], [3.0, 3.0, 3.0]])
    print("=== Test 2.1: calculate Rank(C) for given matrix C  ===")

    rank_C = np.linalg.matrix_rank(C)
    print(f"Matrix C:\n {C} \n... evalutes with np.linalg.matrix_rank to: Rank(C) = {rank_C}")

    x, y = np.array([3.0, 1.0, 2.2]), np.array([4.0, 0.0, -1.0])
    print(f"=== Test 2.2: compare dot with np.dot with given vectors x={x}, y={y}  ===")

    print("dot(x, y) =", dot(x, y), " matches np.dot:", np.allclose(dot(x, y), np.dot(x, y)))

    A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    B = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print("=== Test 2.3: compare matmul with np.matmul with examples from the lecture ===")

    print("A B =\n", matmul(A, B), "\nmatches np.matmul:", np.allclose(matmul(A, B), np.matmul(A, B)))

    A_random = rng.normal(size=(3, 5))
    B_random = rng.normal(size=(5, 2))
    print("=== Test 2.4: compare matmul with np.matmul for two random matrices of shape (3, 5) and (5, 2) ===")

    print("A_random B_random =\n", matmul(A_random, B_random), "\nmatches np.matmul:", np.allclose(matmul(A_random, B_random), np.matmul(A_random, B_random)))

    A_random = rng.normal(size=(3, 3))
    B_random = rng.normal(size=(3, 3))
    print("=== Test 2.5: for two random 3x3 matrices A and B, show wheter AB = BA and whether (AB)^T = B^T A^T ===")    

    AB = matmul(A_random, B_random)
    BA = matmul(B_random, A_random)
    print("AB equals BA:", np.allclose(AB, BA))

    AB_transposed = AB.T
    BT_AT = matmul(B_random.T, A_random.T)
    print("(AB)^T equals B^T A^T:", np.allclose(AB_transposed, BT_AT))


def task_3(rng):
    plot_function_and_derivative(f, df)

    trajectories = {}
    print("=== Test 3.1: Number of iterations and final iterate for eta=0.05 for different x0 ===") 
    for x0 in [-1.5, -0.3, 0.0, 0.3, 1.5]:
        trajectories[f"$x^{{(0)}} = {x0}$"] = gradient_descent(df, x0, learning_rate=0.05)
    for key in trajectories:
        print(f"x_0: {trajectories[key][0]},\t n_iter: {np.size(trajectories[key])-1},\t final_iter: {trajectories[key][-1]}")
    plot_trajectories(f, trajectories, filename="3_trajectories_initial_points.pdf")

    trajectories, function_values = {}, {}
    print("=== Test 3.2: Number of iterations and final iterate for x0=1.5 for different etas ===") 
    for learning_rate in [0.01, 0.05, 0.15, 0.3]:
        label = rf"$\eta = {learning_rate}$"
        trajectories[label] = gradient_descent(df, 1.5, learning_rate)
        function_values[label] = f(trajectories[label])
        # TODO: print the number of iterations and the final iterate x_T for every learning rate
        print(f"eta: {learning_rate},\t n_iter:{np.size(trajectories[label])-1},\t final_iter: {trajectories[label][-1]}")
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
