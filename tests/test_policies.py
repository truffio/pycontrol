import numpy as np

from mlcontrol.policies import MLP


def test_flat_parameter_count():
    net = MLP((4, 16, 16, 1))
    assert net.n_params == 4 * 16 + 16 + 16 * 16 + 16 + 16 + 1 == net.theta.size


def test_forward_batch_matches_forward():
    rng = np.random.default_rng(0)
    net = MLP((4, 8, 8, 1), in_scale=[1, 2, 0.5, 2], out_scale=10.0, rng=rng)
    thetas = rng.normal(size=(3, net.n_params))
    X = rng.normal(size=(3, 5, 4))
    batch = net.forward_batch(X, thetas)
    for p in range(3):
        assert np.allclose(batch[p], net.forward(X[p], thetas[p]))
    assert np.all(np.abs(batch) <= 10.0)


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    net = MLP((3, 5, 4, 1), out_scale=2.0, rng=rng)
    X, Y = rng.normal(size=(20, 3)), rng.uniform(-1.5, 1.5, 20)
    _, grad = net.mse_loss_and_grad(X, Y)
    eps = 1e-6
    for i in rng.choice(net.n_params, 15, replace=False):
        d = np.zeros(net.n_params)
        d[i] = eps
        lp, _ = net.mse_loss_and_grad(X, Y, net.theta + d)
        lm, _ = net.mse_loss_and_grad(X, Y, net.theta - d)
        assert np.isclose(grad[i], (lp - lm) / (2 * eps), rtol=1e-4, atol=1e-8)
