"""Plotting functions for Exercise Sheet P1. You do not need to edit this file.

All plots are saved to the folder plots/ next to this file.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")


def _save(fig, filename):
    os.makedirs(PLOT_DIR, exist_ok=True)
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {path}")


def plot_norms_over_p(p_norm, vectors, ps, filename="1_norms_over_p.pdf"):
    """Plots ||x||_p over p for every vector x in vectors, with ||x||_inf as dashed lines (Task 1.3).

    vectors: dict {label: vector}
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (label, x) in enumerate(vectors.items()):
        ax.plot(ps, [p_norm(x, p) for p in ps], "o-", color=f"C{i}", markersize=3, label=label)
        ax.axhline(p_norm(x, np.inf), color=f"C{i}", linestyle="--", linewidth=1)
    ax.plot([], [], "k--", linewidth=1, label=r"$\|\mathbf{x}\|_\infty$")
    ax.set_xlabel("$p$")
    ax.set_ylabel(r"$\|\mathbf{x}\|_p$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, filename)


def plot_function_and_derivative(f, df, lim=2.0, filename="3_function_and_derivative.pdf"):
    """Plots f and its derivative df on [-lim, lim] (Task 3.2)."""
    x = np.linspace(-lim, lim, 401)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, f(x), label="$f(x)$")
    ax.plot(x, df(x), label="$f'(x)$")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel("$x$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, filename)


def plot_trajectories(f, trajectories, lim=2.0, filename="3_trajectories.pdf"):
    """Plots f on [-lim, lim] and the gradient descent iterates on top of it (Task 3.3).

    trajectories: dict {label: array of shape (T + 1,)}
    """
    x = np.linspace(-lim, lim, 401)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, f(x), color="gray", label="$f(x)$")
    for i, (label, trajectory) in enumerate(trajectories.items()):
        ax.plot(trajectory, f(trajectory), "o-", color=f"C{i}", markersize=3, linewidth=0.8, label=label)
        ax.plot(trajectory[-1], f(trajectory[-1]), "s", color=f"C{i}", markersize=7)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-1.5, 3)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="small")
    _save(fig, filename)


def plot_over_iterations(values, ylabel, filename="3_function_over_iterations.pdf"):
    """Plots values over the gradient descent iterations (Task 3.3).

    values: dict {label: array of shape (T + 1,)}
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, value in values.items():
        ax.plot(value, linewidth=1.5, label=label)
    ax.set_xlabel("iteration $t$")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, filename)
