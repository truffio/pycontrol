"""Example 2 - Robust state feedback for a two-mass flexible drive.

Plant: a motor-side mass m1 driven by force u is coupled to a load mass m2
through a spring k and damper c (a lightly damped, non-collocated
4th-order system). The load position x2 must track a reference while a load
disturbance force acts on m2.

    x = [x1, v1, x2, v2],   y = x2

Controller: full state feedback with integral action on the load position
error (5 gains),

    u = -K x + ki * xi,     d(xi)/dt = r - y

The simulated "true" plant adds a first-order actuator lag that the
simplified design model omits.

Learning: CEM with *domain randomisation*. Every iteration draws a fresh
batch of plants whose load mass, stiffness and damping are perturbed within
the stated uncertainty; each candidate gain vector is simulated on the whole
batch with python-control and scored by its mean + worst-case cost. The
result is a controller that is robust by construction, compared with an LQR
designed with ``control.lqr`` on the nominal model only.
"""

from __future__ import annotations

import control as ct
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mlcontrol import plotting as P
from mlcontrol.examples import register
from mlcontrol.examples.base import Evaluation, Example, Param, TrainResult
from mlcontrol.learners import cem

UNSTABLE_COST = 1e3
GAIN_SCALE = np.array([100.0, 20.0, 100.0, 20.0, 100.0])   # parameter scaling for CEM


def plant_matrices(m1, m2, k, c, b1, tau=0.0):
    """State-space matrices of the two-mass system (inputs: u, load disturbance d).

    With ``tau > 0`` a first-order actuator lag is appended as a 5th state
    (the force actually applied to m1); it is not measured by the controller.
    """
    A = np.array([
        [0, 1, 0, 0],
        [-k / m1, -(c + b1) / m1, k / m1, c / m1],
        [0, 0, 0, 1],
        [k / m2, c / m2, -k / m2, -c / m2],
    ])
    B = np.array([[0.0], [1 / m1], [0.0], [0.0]])
    Bd = np.array([[0.0], [0.0], [0.0], [1 / m2]])
    C = np.array([[0.0, 0.0, 1.0, 0.0]])
    if tau > 0:
        A = np.block([[A, B], [np.zeros((1, 4)), np.array([[-1 / tau]])]])
        B = np.array([[0.0], [0.0], [0.0], [0.0], [1 / tau]])
        Bd = np.vstack([Bd, [[0.0]]])
        C = np.hstack([C, np.zeros((1, 1))])
    return A, B, Bd, C


def nominal_params(cfg) -> dict:
    """Nominal plant including the actuator lag (the 'true' simulator)."""
    return {k: cfg[k] for k in ("m1", "m2", "k", "c", "b1", "tau")}


def design_model_params(cfg) -> dict:
    """The simplified model a classical designer would use (no actuator lag)."""
    return {k: cfg[k] for k in ("m1", "m2", "k", "c", "b1")}


def plant_ss(p) -> ct.StateSpace:
    A, B, Bd, C = plant_matrices(**p)
    return ct.ss(A, np.hstack([B, Bd]), C, 0, inputs=["u", "d"], outputs=["y"])


def sample_plants(cfg, n, rng) -> list:
    """Draw ``n`` perturbed plants within the configured uncertainty."""
    unc = cfg["uncertainty"]
    out = []
    for _ in range(n):
        p = nominal_params(cfg)
        p["m2"] *= rng.uniform(1 - unc, 1 + unc)
        p["k"] *= rng.uniform(1 - unc, 1 + unc)
        p["c"] *= rng.uniform(0.5, 2.0)
        p["tau"] *= rng.uniform(0.5, 1.5)
        out.append(p)
    return out


def closed_loop(p, gains) -> ct.StateSpace:
    """Closed loop with integral action; inputs [r, d], outputs [y, u]."""
    A, B, Bd, C = plant_matrices(**p)
    n = A.shape[0]
    K = np.zeros((1, n))
    K[0, :4] = gains[:4]                       # only the mechanical states are measured
    ki = gains[4]
    Acl = np.block([[A - B @ K, B * ki], [-C, np.zeros((1, 1))]])
    Bcl = np.block([[np.zeros((n, 1)), Bd], [np.ones((1, 1)), np.zeros((1, 1))]])
    Ccl = np.block([[C, np.zeros((1, 1))], [-K, np.array([[ki]])]])
    return ct.ss(Acl, Bcl, Ccl, 0, inputs=["r", "d"], outputs=["y", "u"])


def scenario(cfg, n_points=400):
    """Reference step at t=0 and load-disturbance step at half the horizon."""
    t = np.linspace(0, cfg["T_final"], n_points)
    r = np.full_like(t, cfg["r"])
    d = np.where(t >= cfg["T_final"] / 2, cfg["d"], 0.0)
    return t, np.vstack([r, d])


def simulate(p, gains, cfg, n_points=400):
    sys = closed_loop(p, gains)
    t, U = scenario(cfg, n_points)
    resp = ct.forced_response(sys, t, U)
    y, u = np.asarray(resp.outputs, dtype=float)
    return t, y, u, sys.poles()


def plant_cost(p, gains, cfg) -> float:
    t, y, u, poles = simulate(p, gains, cfg, n_points=250)
    if np.any(poles.real >= -1e-6) or not np.all(np.isfinite(y)):
        return UNSTABLE_COST
    dt = t[1] - t[0]
    itae = np.sum(t * np.abs(cfg["r"] - y)) * dt / (cfg["r"] * cfg["T_final"])
    effort = np.sqrt(np.mean(u ** 2)) / cfg["F_max"]
    violation = max(0.0, np.abs(u).max() / cfg["F_max"] - 1.0)
    return itae + cfg["w_effort"] * effort ** 2 + cfg["w_saturation"] * violation ** 2


def robust_cost(plants, gains, cfg) -> float:
    costs = np.array([plant_cost(p, gains, cfg) for p in plants])
    return float(costs.mean() + cfg["w_worst"] * costs.max())


def lqr_gains(cfg) -> np.ndarray:
    """LQR on the nominal augmented model [x, xi], u = -[K, -ki][x; xi]."""
    A, B, _, C = plant_matrices(**design_model_params(cfg))
    Aa = np.block([[A, np.zeros((4, 1))], [-C, np.zeros((1, 1))]])
    Ba = np.vstack([B, [[0.0]]])
    Q = np.diag([cfg["lqr_q_pos"], 1.0, cfg["lqr_q_pos"], 1.0, cfg["lqr_q_int"]])
    K, _, _ = ct.lqr(Aa, Ba, Q, np.array([[cfg["lqr_r"]]]))
    K = np.asarray(K).ravel()
    return np.concatenate([K[:4], [-K[4]]])


@register
class TwoMassExample(Example):
    key = "two_mass"
    title = "Two-mass flexible drive - robust learned state feedback"
    level = "Intermediate"
    summary = """
**Plant.** A motor mass drives a load mass through a flexible coupling
(spring + light damping). Only the load position matters, but the force acts
on the motor side, so there is a lightly damped resonance between actuator
and output. The 4th-order mechanical model is a `control.ss` system. The
"real" plant also has an actuator lag (τ ≈ 50 ms) that the classical design
model leaves out, and its load mass, stiffness, damping and actuator time
constant are **uncertain** (default ±30 %).

**Controller.** Full state feedback plus integral action on the load
position error (5 gains), which also rejects a step load disturbance.

**Learning algorithm.** *Cross-Entropy Method with domain randomisation*.
Every iteration draws a new batch of random plants inside the uncertainty
range, simulates every candidate gain vector on every plant with
`control.forced_response`, and scores it by **mean + worst-case** cost. The
learner never sees the nominal model, only simulated responses.

**Baseline.** An LQR designed with `control.lqr` on the simplified
*nominal* model, which is optimal for that one model but ignores the
actuator lag, the uncertainty and the force limit.
"""
    params = [
        Param("m1", "Motor mass m1 [kg]", 1.0, 0.1, 5.0, 0.1, "plant"),
        Param("m2", "Load mass m2 [kg] (nominal)", 0.5, 0.1, 5.0, 0.1, "plant"),
        Param("k", "Coupling stiffness k [N/m] (nominal)", 50.0, 5.0, 500.0, 5.0, "plant"),
        Param("c", "Coupling damping c [N s/m] (nominal)", 0.2, 0.0, 5.0, 0.05, "plant"),
        Param("b1", "Motor-side friction b1 [N s/m]", 0.5, 0.0, 5.0, 0.1, "plant"),
        Param("tau", "Actuator time constant [s] (nominal, unmodelled by LQR)", 0.05, 0.0, 0.3, 0.01, "plant"),
        Param("uncertainty", "Parameter uncertainty (±fraction)", 0.3, 0.0, 0.6, 0.05, "plant"),
        Param("F_max", "Actuator force limit [N]", 20.0, 1.0, 200.0, 1.0, "plant"),
        Param("r", "Position reference [m]", 1.0, 0.1, 5.0, 0.1, "plant"),
        Param("d", "Load disturbance force [N]", 2.0, 0.0, 20.0, 0.5, "plant"),
        Param("T_final", "Simulation horizon [s]", 10.0, 2.0, 30.0, 1.0, "plant"),
        Param("n_iter", "CEM iterations", 40, 5, 150, 1, "learning"),
        Param("pop_size", "Population size", 30, 10, 200, 5, "learning"),
        Param("n_plants", "Random plants per iteration", 6, 1, 30, 1, "learning"),
        Param("elite_frac", "Elite fraction", 0.2, 0.05, 0.5, 0.05, "learning"),
        Param("w_worst", "Weight: worst-case plant", 1.0, 0.0, 5.0, 0.1, "learning"),
        Param("w_effort", "Weight: control effort", 0.05, 0.0, 5.0, 0.01, "learning"),
        Param("w_saturation", "Weight: force-limit violation", 5.0, 0.0, 100.0, 1.0, "learning"),
        Param("n_test", "Test plants for evaluation", 40, 5, 200, 5, "learning"),
        Param("lqr_q_pos", "LQR baseline: position weight", 1000.0, 1.0, 1e5, 10.0, "learning"),
        Param("lqr_q_int", "LQR baseline: integral weight", 5000.0, 1.0, 1e5, 10.0, "learning"),
        Param("lqr_r", "LQR baseline: input weight", 0.1, 0.001, 10.0, 0.01, "learning"),
        Param("seed", "Random seed", 0, 0, 9999, 1, "learning"),
    ]

    def _train(self, cfg, callback):
        rng = np.random.default_rng(int(cfg["seed"]))
        n_plants = int(cfg["n_plants"])

        def batch_cost(pop, it):
            plants = sample_plants(cfg, n_plants, rng)      # fresh draw every iteration
            return np.array([robust_cost(plants, theta * GAIN_SCALE, cfg) for theta in pop])

        res = cem(batch_cost, mean=np.full(5, 0.5), std=np.full(5, 1.0),
                  n_iter=int(cfg["n_iter"]), pop_size=int(cfg["pop_size"]),
                  elite_frac=cfg["elite_frac"], rng=rng, callback=callback,
                  stage="CEM + domain randomisation")
        for h in res.history:
            h["params"] = (np.asarray(h["params"]) * GAIN_SCALE).tolist()

        # Choose between best sample and final mean on a common validation set.
        val = sample_plants(cfg, 3 * n_plants, np.random.default_rng(int(cfg["seed"]) + 1))
        cands = [res.x_best * GAIN_SCALE, res.x * GAIN_SCALE]
        scores = [robust_cost(val, g, cfg) for g in cands]
        gains = cands[int(np.argmin(scores))]
        base = lqr_gains(cfg)
        return TrainResult(
            example=self.key, config=cfg,
            controller={"gains": gains.tolist(), "cost": min(scores)},
            baseline={"gains": base.tolist(), "cost": robust_cost(val, base, cfg)},
            history=res.history)

    def evaluate(self, result: TrainResult) -> Evaluation:
        cfg = result.config
        nominal = nominal_params(cfg)
        test = sample_plants(cfg, int(cfg["n_test"]), np.random.default_rng(12345))
        ctrls = {"learned": np.array(result.controller["gains"]),
                 "baseline": np.array(result.baseline["gains"])}
        colors = {"learned": P.LEARNED, "baseline": P.BASELINE}
        labels = {"learned": "Learned (robust CEM)", "baseline": "LQR (nominal)"}

        figs = {"Learning curve": P.learning_curve(result.history)}

        nom = {k: simulate(nominal, g, cfg) for k, g in ctrls.items()}
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
        t = nom["learned"][0]
        fig.add_trace(P.line(t, np.full_like(t, cfg["r"]), "Reference", P.REFERENCE, dash="dash"), 1, 1)
        for k, (t, y, u, _) in nom.items():
            fig.add_trace(P.line(t, y, labels[k], colors[k]), 1, 1)
            fig.add_trace(P.line(t, u, labels[k], colors[k], showlegend=False), 2, 1)
        for sgn in (1, -1):
            fig.add_hline(y=sgn * cfg["F_max"], line=dict(color=P.REFERENCE, width=1, dash="dash"),
                          row=2, col=1)
        fig.add_vline(x=cfg["T_final"] / 2, line=dict(color=P.REFERENCE, width=1, dash="dot"),
                      annotation_text="load disturbance", annotation_position="top right")
        P.style(fig, "Nominal plant: load position and control force", height=520)
        fig.update_yaxes(title_text="Load position [m]", row=1, col=1)
        fig.update_yaxes(title_text="Force [N]", row=2, col=1)
        fig.update_xaxes(title_text="Time [s]", row=2, col=1)
        figs["Nominal response"] = fig

        ens = {k: [simulate(p, g, cfg) for p in test] for k, g in ctrls.items()}
        figs["Uncertain plants"] = self._ensemble_figure(ens, cfg, labels, colors)
        figs["Closed-loop poles"] = self._pole_figure(ens, labels, colors)
        figs["Cost distribution"] = self._cost_figure(test, ctrls, cfg, labels, colors)

        return Evaluation(metrics=self._metrics(cfg, test, ctrls, nom, ens), figures=figs)

    def _ensemble_figure(self, ens, cfg, labels, colors):
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                            subplot_titles=[labels["learned"], labels["baseline"]])
        for col, k in enumerate(("learned", "baseline"), start=1):
            for i, (t, y, u, poles) in enumerate(ens[k]):
                y = np.clip(y, -cfg["r"], 3 * cfg["r"])      # keep unstable runs on-screen
                fig.add_trace(P.line(t, y, labels[k], colors[k], width=1, showlegend=i == 0,
                                     opacity=0.45, hoverinfo="skip"), 1, col)
            fig.add_trace(P.line(t, np.full_like(t, cfg["r"]), "Reference", P.REFERENCE,
                                 dash="dash", showlegend=col == 1), 1, col)
        P.style(fig, f"Load position on {len(ens['learned'])} random plants "
                     f"(±{cfg['uncertainty']:.0%} mass/stiffness, damping ×0.5-2)", height=400)
        fig.update_layout(hovermode="closest")
        fig.update_xaxes(title_text="Time [s]")
        fig.update_yaxes(title_text="Load position [m]", col=1)
        return fig

    def _pole_figure(self, ens, labels, colors):
        fig = go.Figure()
        for k in ("baseline", "learned"):
            poles = np.concatenate([e[3] for e in ens[k]])
            fig.add_trace(go.Scatter(x=poles.real, y=poles.imag, mode="markers", name=labels[k],
                                     marker=dict(color=colors[k], size=8, opacity=0.6,
                                                 symbol="x" if k == "baseline" else "circle",
                                                 line=dict(width=1, color="white"))))
        fig.add_vline(x=0, line=dict(color=P.REFERENCE, width=1, dash="dash"))
        P.style(fig, "Closed-loop poles across the uncertain plants", "Real", "Imaginary", 420)
        fig.update_layout(hovermode="closest")
        return fig

    def _cost_figure(self, test, ctrls, cfg, labels, colors):
        fig = go.Figure()
        for k, g in ctrls.items():
            costs = [plant_cost(p, g, cfg) for p in test]
            fig.add_trace(go.Box(y=costs, name=labels[k], marker_color=colors[k],
                                 boxpoints="all", jitter=0.4, pointpos=0))
        fig.update_yaxes(type="log")
        P.style(fig, "Per-plant cost on the test set (lower is better)", "", "Cost (log scale)")
        fig.update_layout(hovermode="closest", showlegend=False)
        return fig

    def _metrics(self, cfg, test, ctrls, nom, ens):
        rows = {}
        for k, g in ctrls.items():
            t, y, u, poles = nom[k]
            half = t < cfg["T_final"] / 2
            info = ct.step_info(y[half], t[half], yfinal=cfg["r"])
            costs = np.array([plant_cost(p, g, cfg) for p in test])
            stable = [np.all(e[3].real < 0) for e in ens[k]]
            peak_u = [np.abs(e[2]).max() for e in ens[k]]
            rows[k] = {
                "Gains [k1 k2 k3 k4 ki]": " ".join(f"{v:.3g}" for v in g),
                "Nominal settling time [s]": info["SettlingTime"],
                "Nominal overshoot [%]": info["Overshoot"],
                "Nominal peak force [N]": float(np.abs(u).max()),
                "Nominal cost": plant_cost(nominal_params(cfg), g, cfg),
                "Test plants stable [%]": 100 * float(np.mean(stable)),
                "Test plants within force limit [%]": 100 * float(np.mean(np.array(peak_u) <= cfg["F_max"])),
                "Test mean cost": float(costs.mean()),
                "Test worst-case cost": float(costs.max()),
            }
        return [{"metric": m, "learned": rows["learned"][m], "baseline": rows["baseline"][m]}
                for m in rows["learned"]]
