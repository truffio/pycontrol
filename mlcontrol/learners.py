"""Black-box learning algorithms used to design controllers.

Both optimisers treat the closed-loop simulation as a black box: they
propose a population of parameter vectors, receive one scalar cost per
candidate, and update a search distribution. The cost function has the
signature ``cost_fn(population, iteration) -> costs`` where ``population``
is an ``(n, d)`` array, so a caller can vectorise its simulations or
re-sample random plants/initial conditions on each iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

CostFn = Callable[[np.ndarray, int], np.ndarray]


@dataclass
class IterationInfo:
    """Progress report passed to callbacks after each iteration."""

    iteration: int
    n_iter: int
    best_cost: float
    mean_cost: float
    params: np.ndarray
    stage: str = ""
    validation_cost: Optional[float] = None    # cost on a fixed held-out set, if computed


@dataclass
class OptimResult:
    x: np.ndarray                  # final solution (distribution mean)
    x_best: np.ndarray             # best candidate seen during the search
    f_best: float
    history: list = field(default_factory=list)   # list of dicts per iteration


def _report(callback, history, info: IterationInfo):
    history.append({
        "iteration": info.iteration,
        "best_cost": float(info.best_cost),
        "mean_cost": float(info.mean_cost),
        "params": np.asarray(info.params, dtype=float).tolist(),
        "stage": info.stage,
    })
    if callback is not None:
        callback(info)


def cem(cost_fn: CostFn, mean, std, *, n_iter: int = 30, pop_size: int = 40,
        elite_frac: float = 0.2, lower=None, upper=None, smoothing: float = 0.7,
        min_std: float = 1e-3, rng: Optional[np.random.Generator] = None,
        callback: Optional[Callable[[IterationInfo], None]] = None,
        stage: str = "CEM", record_params: bool = True) -> OptimResult:
    """Cross-Entropy Method with a diagonal Gaussian search distribution.

    Each iteration samples ``pop_size`` candidates, keeps the
    ``elite_frac`` best, and moves the Gaussian towards the elites
    (``smoothing`` is the weight of the new estimate).
    """
    rng = np.random.default_rng() if rng is None else rng
    mean = np.asarray(mean, dtype=float).copy()
    std = np.asarray(std, dtype=float).copy()
    lower = -np.inf if lower is None else np.asarray(lower, dtype=float)
    upper = np.inf if upper is None else np.asarray(upper, dtype=float)
    n_elite = max(2, int(round(elite_frac * pop_size)))

    x_best, f_best = mean.copy(), np.inf
    history: list = []
    for it in range(n_iter):
        pop = rng.normal(mean, std, size=(pop_size, mean.size))
        pop = np.clip(pop, lower, upper)
        costs = np.asarray(cost_fn(pop, it), dtype=float)
        costs = np.where(np.isfinite(costs), costs, np.inf)
        order = np.argsort(costs)
        elites = pop[order[:n_elite]]
        if costs[order[0]] < f_best:
            f_best, x_best = float(costs[order[0]]), pop[order[0]].copy()
        mean = smoothing * elites.mean(axis=0) + (1 - smoothing) * mean
        std = np.maximum(smoothing * elites.std(axis=0) + (1 - smoothing) * std, min_std)
        finite = costs[np.isfinite(costs)]
        _report(callback, history, IterationInfo(
            it + 1, n_iter, costs[order[0]],
            finite.mean() if finite.size else np.inf,
            mean if record_params else np.empty(0), stage))
    return OptimResult(mean, x_best, f_best, history)


class Adam:
    """Adam optimiser on a flat parameter vector (minimisation)."""

    def __init__(self, dim: int, lr: float = 1e-2, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = np.zeros(dim)
        self.v = np.zeros(dim)
        self.t = 0

    def step(self, grad: np.ndarray) -> np.ndarray:
        """Return the parameter increment for gradient ``grad``."""
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad ** 2
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        return -self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def centered_ranks(x: np.ndarray) -> np.ndarray:
    """Map values to ranks scaled to [-0.5, 0.5] (robust fitness shaping)."""
    ranks = np.empty(x.size)
    ranks[np.argsort(x)] = np.arange(x.size)
    return ranks / (x.size - 1) - 0.5


def evolution_strategies(cost_fn: CostFn, theta0, *, n_iter: int = 100,
                         pop_size: int = 32, sigma: float = 0.05, lr: float = 0.02,
                         weight_decay: float = 0.0,
                         rng: Optional[np.random.Generator] = None,
                         callback: Optional[Callable[[IterationInfo], None]] = None,
                         stage: str = "ES", record_params: bool = False) -> OptimResult:
    """OpenAI-style Evolution Strategies with antithetic sampling and Adam.

    ``pop_size`` antithetic pairs are evaluated per iteration together with
    the current parameters (last row of the population), whose cost is
    reported as ``best_cost`` for the learning curve.
    """
    rng = np.random.default_rng() if rng is None else rng
    theta = np.asarray(theta0, dtype=float).copy()
    opt = Adam(theta.size, lr=lr)
    x_best, f_best = theta.copy(), np.inf
    history: list = []
    for it in range(n_iter):
        eps = rng.standard_normal((pop_size, theta.size))
        pop = np.vstack([theta + sigma * eps, theta - sigma * eps, theta[None, :]])
        costs = np.asarray(cost_fn(pop, it), dtype=float)
        costs = np.where(np.isfinite(costs), costs, 1e12)
        f_theta = costs[-1]
        if f_theta < f_best:
            f_best, x_best = float(f_theta), theta.copy()
        shaped = centered_ranks(costs[:-1])
        grad = (shaped[:pop_size] - shaped[pop_size:]) @ eps / (2 * pop_size * sigma)
        theta = theta + opt.step(grad + weight_decay * theta)
        _report(callback, history, IterationInfo(
            it + 1, n_iter, f_theta, costs[:-1].mean(),
            theta if record_params else np.empty(0), stage))
    return OptimResult(theta, x_best, f_best, history)
