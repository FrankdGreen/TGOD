from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SinkhornResult:
    distance: float
    converged: bool
    iterations: int
    marginal_error: float


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def pairwise_squared_distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1]:
        raise ValueError(f"Expected (N,D) and (M,D) arrays, got {first.shape} and {second.shape}.")
    costs = np.sum((first[:, None, :] - second[None, :, :]) ** 2, axis=-1)
    return np.maximum(costs, 0.0)


def solve_sinkhorn(
    first: np.ndarray,
    second: np.ndarray,
    *,
    epsilon: float = 0.05,
    max_iterations: int = 300,
    tolerance: float = 1e-7,
) -> SinkhornResult:
    """Entropy-regularized optimal-transport cost with uniform trajectory weights.

    Updates are performed in the log domain so long trajectories and small epsilon
    values do not underflow.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    costs = pairwise_squared_distance(first, second)
    rows, columns = costs.shape
    log_a = np.full(rows, -np.log(rows), dtype=np.float64)
    log_b = np.full(columns, -np.log(columns), dtype=np.float64)
    log_kernel = -costs / float(epsilon)
    log_u = np.zeros(rows, dtype=np.float64)
    log_v = np.zeros(columns, dtype=np.float64)

    converged = False
    marginal_error = float("inf")
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        log_u = log_a - _logsumexp(log_kernel + log_v[None, :], axis=1)
        log_v = log_b - _logsumexp(log_kernel.T + log_u[None, :], axis=1)
        if iteration == 1 or iteration % 5 == 0 or iteration == max_iterations:
            current_log_plan = log_u[:, None] + log_kernel + log_v[None, :]
            row_marginal = np.exp(_logsumexp(current_log_plan, axis=1))
            column_marginal = np.exp(_logsumexp(current_log_plan, axis=0))
            marginal_error = float(
                max(
                    np.max(np.abs(row_marginal - np.exp(log_a))),
                    np.max(np.abs(column_marginal - np.exp(log_b))),
                )
            )
            if marginal_error <= tolerance:
                converged = True
                break

    log_plan = log_u[:, None] + log_kernel + log_v[None, :]
    if float(np.max(log_plan)) > 1e-6:
        raise FloatingPointError("Sinkhorn transport plan contains mass above one.")
    plan = np.exp(np.clip(log_plan, -745.0, 0.0))
    distance = float(np.sum(plan * costs))
    if not np.isfinite(distance):
        raise FloatingPointError("Sinkhorn distance became non-finite.")
    return SinkhornResult(distance, converged, iteration, marginal_error)


def sinkhorn_distance(
    first: np.ndarray,
    second: np.ndarray,
    *,
    epsilon: float = 0.05,
    max_iterations: int = 300,
    tolerance: float = 1e-7,
) -> float:
    return solve_sinkhorn(
        first,
        second,
        epsilon=epsilon,
        max_iterations=max_iterations,
        tolerance=tolerance,
    ).distance
