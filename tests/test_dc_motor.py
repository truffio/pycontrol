import control as ct
import numpy as np

from mlcontrol.examples import get_example
from mlcontrol.examples.dc_motor import UNSTABLE_COST, cost, motor_tf, plant_from_cfg, simulate


def test_motor_dc_gain_and_poles():
    P = motor_tf(J=0.01, b=0.1, K=0.01, R=1.0, L=0.5)
    assert np.isclose(ct.dcgain(P), 0.01 / (0.1 * 1.0 + 0.01 ** 2))
    assert np.all(P.poles().real < 0)


def test_cost_penalises_instability_and_voltage():
    ex = get_example("dc_motor")
    cfg = ex.default_config()
    plant = plant_from_cfg(cfg)
    assert cost(plant, (-50.0, 1.0, 0.0), cfg) == UNSTABLE_COST
    # The textbook PID violates the 48 V limit, so it must cost much more.
    assert cost(plant, ex.baseline_gains, cfg) > 100 * cost(plant, (16.0, 33.0, 0.07), cfg)


def test_simulation_returns_plain_arrays():
    ex = get_example("dc_motor")
    cfg = ex.default_config()
    t, y, u, poles = simulate(plant_from_cfg(cfg), (16.0, 33.0, 0.07), cfg)
    assert type(y) is np.ndarray and y.shape == t.shape == u.shape
    assert abs(y[-1] - cfg["r"]) < 0.01


def test_training_beats_baseline_and_evaluates():
    ex = get_example("dc_motor")
    result = ex.train({"n_iter": 8, "pop_size": 20, "seed": 1})
    assert len(result.history) == 8
    assert result.controller["cost"] < result.baseline["cost"]
    ev = ex.evaluate(result)
    assert {"Learning curve", "Step response", "Control voltage", "Loop Bode plot"} <= set(ev.figures)
    peak = next(r for r in ev.metrics if r["metric"] == "Peak voltage [V]")
    assert peak["learned"] <= result.config["V_max"]
