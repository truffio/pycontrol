import numpy as np

from mlcontrol.examples import get_example
from mlcontrol.examples.two_mass import (
    UNSTABLE_COST, closed_loop, design_model_params, lqr_gains, nominal_params,
    plant_cost, plant_matrices, sample_plants,
)


def cfg():
    return get_example("two_mass").default_config()


def test_resonance_frequency():
    c = cfg()
    A, *_ = plant_matrices(**design_model_params(c))
    wn = np.sqrt(c["k"] * (1 / c["m1"] + 1 / c["m2"]))
    assert np.min(np.abs(np.abs(np.linalg.eigvals(A)) - wn)) < 0.1 * wn


def test_actuator_lag_adds_state():
    c = cfg()
    assert plant_matrices(**design_model_params(c))[0].shape == (4, 4)
    assert plant_matrices(**nominal_params(c))[0].shape == (5, 5)
    assert closed_loop(nominal_params(c), np.ones(5)).nstates == 6


def test_ensemble_within_bounds():
    c = cfg()
    plants = sample_plants(c, 50, np.random.default_rng(0))
    m2 = np.array([p["m2"] for p in plants])
    assert np.all(np.abs(m2 / c["m2"] - 1) <= c["uncertainty"] + 1e-12)
    assert m2.std() > 0


def test_lqr_stabilises_design_model():
    c = cfg()
    g = lqr_gains(c)
    assert plant_cost(design_model_params(c) | {"tau": 0.0}, g, c) < UNSTABLE_COST


def test_training_is_robust():
    ex = get_example("two_mass")
    result = ex.train({"n_iter": 12, "pop_size": 20, "n_plants": 3, "n_test": 10, "seed": 0})
    assert result.controller["cost"] < result.baseline["cost"]
    ev = ex.evaluate(result)
    stable = next(r for r in ev.metrics if r["metric"] == "Test plants stable [%]")
    assert stable["learned"] == 100
    assert {"Nominal response", "Uncertain plants", "Closed-loop poles"} <= set(ev.figures)
