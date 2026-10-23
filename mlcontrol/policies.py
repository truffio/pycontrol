"""Neural network policy written in numpy.

The parameters live in one flat vector so that Evolution Strategies can
perturb them directly, and :meth:`MLP.forward_batch` evaluates a whole
population of parameter vectors at once.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


class MLP:
    """Tanh multilayer perceptron ``y = out_scale * tanh(f(x / in_scale))``.

    The output tanh bounds the action, which matches an actuator with a
    saturation limit of ``out_scale``.
    """

    def __init__(self, sizes: Sequence[int] = (4, 16, 16, 1), in_scale=None,
                 out_scale: float = 1.0, rng: Optional[np.random.Generator] = None):
        self.sizes = tuple(int(s) for s in sizes)
        self.in_scale = np.ones(self.sizes[0]) if in_scale is None else np.asarray(in_scale, float)
        self.out_scale = float(out_scale)
        self.shapes = [(a, b) for a, b in zip(self.sizes[:-1], self.sizes[1:])]
        self.n_params = sum(a * b + b for a, b in self.shapes)
        rng = np.random.default_rng() if rng is None else rng
        parts = []
        for a, b in self.shapes:                       # Xavier-style init
            parts += [rng.normal(0, 1 / np.sqrt(a), a * b), np.zeros(b)]
        self.theta = np.concatenate(parts)

    # -- parameter handling -------------------------------------------------
    def _unpack(self, theta: np.ndarray):
        """Split flat parameters (..., n_params) into [(W, b), ...] per layer."""
        lead = theta.shape[:-1]
        layers, i = [], 0
        for a, b in self.shapes:
            W = theta[..., i:i + a * b].reshape(*lead, a, b)
            i += a * b
            bias = theta[..., i:i + b]
            i += b
            layers.append((W, bias))
        return layers

    # -- inference ----------------------------------------------------------
    def forward(self, X: np.ndarray, theta: Optional[np.ndarray] = None) -> np.ndarray:
        """Actions for states ``X`` of shape (N, n_in) -> (N,)."""
        h = X / self.in_scale
        layers = self._unpack(self.theta if theta is None else theta)
        for W, b in layers[:-1]:
            h = np.tanh(h @ W + b)
        W, b = layers[-1]
        return self.out_scale * np.tanh(h @ W + b)[..., 0]

    def forward_batch(self, X: np.ndarray, thetas: np.ndarray) -> np.ndarray:
        """Actions for a population: X (P, N, n_in), thetas (P, n_params) -> (P, N)."""
        h = X / self.in_scale
        layers = self._unpack(thetas)
        for W, b in layers[:-1]:
            h = np.tanh(np.einsum("pni,pio->pno", h, W) + b[:, None, :])
        W, b = layers[-1]
        out = np.einsum("pni,pio->pno", h, W) + b[:, None, :]
        return self.out_scale * np.tanh(out)[..., 0]

    # -- supervised learning --------------------------------------------------
    def mse_loss_and_grad(self, X: np.ndarray, Y: np.ndarray, theta: Optional[np.ndarray] = None):
        """Mean squared error on normalised outputs and its gradient (backprop)."""
        theta = self.theta if theta is None else theta
        layers = self._unpack(theta)
        acts = [X / self.in_scale]
        for W, b in layers:
            acts.append(np.tanh(acts[-1] @ W + b))
        pred = acts[-1][:, 0]
        target = Y / self.out_scale
        err = pred - target
        loss = float(np.mean(err ** 2))

        grads = []
        delta = (2 * err / X.shape[0])[:, None] * (1 - acts[-1] ** 2)
        for li in range(len(layers) - 1, -1, -1):
            W, _ = layers[li]
            grads.append((acts[li].T @ delta, delta.sum(axis=0)))
            if li > 0:
                delta = (delta @ W.T) * (1 - acts[li] ** 2)
        flat = []
        for gW, gb in reversed(grads):
            flat += [gW.ravel(), gb]
        return loss, np.concatenate(flat)
