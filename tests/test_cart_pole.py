import control as ct
import numpy as np

from mlcontrol.examples import get_example
from mlcontrol.examples.cart_pole import (
    CartPoleExample, cartpole_nlsys, dynamics, lqr_gain, plant_params, rollout,
)


def cfg():
    return get_example("cart_pole").default_config()


def test_upright_is_equilibrium_and_unstable():
    c = cfg()
    assert np.allclose(dynamics(np.zeros(4), 0.0, plant_params(c)), 0)
    lin = ct.linearize(cartpole_nlsys(c), np.zeros(4), np.zeros(1))
    assert np.max(lin.poles().real) > 0


def test_lqr_recovers_small_angle():
    c = cfg()
    K = lqr_gain(c)
    _, ok = rollout(lambda X: np.clip(-X @ K, -c["F_max"], c["F_max"]), np.array([[0, 0, 0.1, 0]]), c)
    assert ok[0]


def test_rk4_agrees_with_python_control():
    c = cfg()
    K = lqr_gain(c)
    pol = lambda X: np.clip(-X @ K, -c["F_max"], c["F_max"])  # noqa: E731
    X0 = np.array([0.2, 0.0, 0.2, 0.0])
    _, _, t, X, _ = rollout(pol, X0[None], c, record=True)
    tv, Xv, _, ok = CartPoleExample._validate_with_python_control(c, pol, X0)
    assert ok
    assert np.abs(X[0, :len(tv), 2] - Xv[:, 2]).max() < 0.02


def test_two_stage_training_improves_on_imitation():
    ex = get_example("cart_pole")
    r = ex.train({"bc_steps": 600, "es_iter": 30, "es_pop": 8, "es_episodes": 12, "T_final": 4.0})
    stages = {h["stage"] for h in r.history}
    assert stages == {"Imitation (behaviour cloning)", "Evolution Strategies (growing initial-state range)"}
    assert r.controller["validation_cost"] <= r.extras["bc_validation_cost"]
    ev = ex.evaluate(r)
    assert {"Learning curve", "Time response", "Region of attraction", "Animation"} <= set(ev.figures)
