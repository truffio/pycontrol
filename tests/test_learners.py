import numpy as np

from mlcontrol.learners import Adam, cem, centered_ranks, evolution_strategies


def test_cem_finds_quadratic_minimum():
    target = np.array([1.5, -2.0, 0.5])

    def cost(pop, it):
        return ((pop - target) ** 2).sum(axis=1)

    calls = []
    res = cem(cost, np.zeros(3), np.full(3, 2.0), n_iter=40, pop_size=50,
              rng=np.random.default_rng(0), callback=calls.append)
    assert np.allclose(res.x, target, atol=0.05)
    assert len(res.history) == 40 == len(calls)
    assert res.history[-1]["best_cost"] < res.history[0]["best_cost"]


def test_cem_respects_bounds():
    res = cem(lambda p, it: -p.sum(axis=1), np.zeros(2), np.ones(2), n_iter=15,
              lower=[-1, -1], upper=[0.5, 0.25], rng=np.random.default_rng(1))
    assert np.all(res.x <= np.array([0.5, 0.25]) + 1e-12)


def test_centered_ranks_range():
    r = centered_ranks(np.array([3.0, -1.0, 10.0, 2.0]))
    assert r.min() == -0.5 and r.max() == 0.5
    assert r[2] == 0.5 and r[1] == -0.5


def test_adam_descends():
    opt = Adam(1, lr=0.1)
    x = np.array([5.0])
    for _ in range(300):
        x = x + opt.step(2 * x)
    assert abs(x[0]) < 0.1


def test_es_minimises_rosenbrock():
    def rosen(pop, it):
        x, y = pop[:, 0], pop[:, 1]
        return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

    res = evolution_strategies(rosen, np.array([-1.0, 1.0]), n_iter=600, pop_size=20,
                               sigma=0.05, lr=0.02, rng=np.random.default_rng(0))
    assert rosen(res.x[None, :], 0)[0] < 0.05
