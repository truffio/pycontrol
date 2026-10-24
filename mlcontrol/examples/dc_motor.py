"""Example 1 - DC motor speed control with a PID tuned by the Cross-Entropy Method.

Plant: armature-controlled DC motor, voltage -> shaft speed,

    P(s) = K / ((J s + b)(L s + R) + K^2)

Controller: PID with first-order derivative filter,

    C(s) = Kp + Ki / s + Kd s / (Tf s + 1)

Learning: CEM searches log10(Kp, Ki, Kd). Every candidate is scored by
simulating the closed loop with python-control (``feedback`` +
``step_response``) and combining tracking error, overshoot, control effort
and violation of the supply-voltage limit into one cost.
"""

from __future__ import annotations

import control as ct
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from mlcontrol import plotting as P
from mlcontrol.examples import register
from mlcontrol.examples.base import Evaluation, Example, Param, TrainResult
from mlcontrol.learners import cem

UNSTABLE_COST = 1e6


def motor_tf(J=0.01, b=0.1, K=0.01, R=1.0, L=0.5) -> ct.TransferFunction:
    """Voltage-to-speed transfer function of the DC motor."""
    s = ct.tf("s")
    return K / ((J * s + b) * (L * s + R) + K ** 2)


def pid_tf(kp, ki, kd, tf=0.01) -> ct.TransferFunction:
    s = ct.tf("s")
    return kp + ki / s + kd * s / (tf * s + 1)


def plant_from_cfg(cfg) -> ct.TransferFunction:
    return motor_tf(cfg["J"], cfg["b"], cfg["K"], cfg["R"], cfg["L"])


def simulate(plant, gains, cfg, n_points=600):
    """Closed-loop unit step: returns t, y (speed), u (voltage), poles."""
    C = pid_tf(*gains, tf=cfg["Tf"])
    T_ry = ct.feedback(C * plant, 1)
    T_ru = ct.feedback(C, plant)
    poles = T_ry.poles()
    t = np.linspace(0, cfg["T_final"], n_points)
    y = ct.step_response(T_ry, t).outputs * cfg["r"]
    u = ct.step_response(T_ru, t).outputs * cfg["r"]
    return t, np.asarray(y, dtype=float).ravel(), np.asarray(u, dtype=float).ravel(), poles


def cost(plant, gains, cfg) -> float:
    """Scalar closed-loop performance cost (lower is better)."""
    t, y, u, poles = simulate(plant, gains, cfg, n_points=300)
    if np.any(poles.real >= -1e-6) or not np.all(np.isfinite(y)):
        return UNSTABLE_COST
    r = cfg["r"]
    e = (r - y) / r
    dt = t[1] - t[0]
    itae = np.sum(t * np.abs(e)) * dt
    overshoot = max(0.0, (y.max() - r) / r)
    effort = np.sqrt(np.mean(u ** 2)) / cfg["V_max"]
    violation = max(0.0, np.abs(u).max() / cfg["V_max"] - 1.0)
    return (itae
            + cfg["w_overshoot"] * overshoot ** 2
            + cfg["w_effort"] * effort ** 2
            + cfg["w_saturation"] * violation ** 2)


@register
class DCMotorExample(Example):
    key = "dc_motor"
    title = "DC motor speed - learned PID"
    level = "Basic"
    summary = """
**Plant.** A DC motor converts armature voltage into shaft speed. Its model
is a linear 2nd-order transfer function built with `control.tf`.

**Controller.** A PID controller with a filtered derivative (3 gains).

**Learning algorithm.** The *Cross-Entropy Method* (CEM) treats the closed
loop as a black box. Each iteration it samples a population of gain sets
(in log scale), simulates every closed-loop step response with
`control.feedback` + `control.step_response`, keeps the best 20 %, and moves
its search distribution towards them. The cost combines tracking error
(ITAE), overshoot, control effort and a penalty when the voltage exceeds the
supply limit.

**Baseline.** A textbook hand-tuned PID (Kp=100, Ki=200, Kd=10), tuned for
speed of response without regard to the voltage limit.
"""
    params = [
        Param("J", "Rotor inertia J [kg m²]", 0.01, 0.001, 0.1, 0.001, "plant"),
        Param("b", "Viscous friction b [N m s]", 0.1, 0.01, 1.0, 0.01, "plant"),
        Param("K", "Motor constant K [N m/A]", 0.01, 0.001, 0.1, 0.001, "plant"),
        Param("R", "Armature resistance R [Ω]", 1.0, 0.1, 10.0, 0.1, "plant"),
        Param("L", "Armature inductance L [H]", 0.5, 0.01, 2.0, 0.01, "plant"),
        Param("V_max", "Supply voltage limit [V]", 48.0, 12.0, 500.0, 1.0, "plant"),
        Param("r", "Speed reference [rad/s]", 1.0, 0.1, 10.0, 0.1, "plant"),
        Param("Tf", "Derivative filter time constant [s]", 0.01, 0.001, 0.1, 0.001, "plant"),
        Param("T_final", "Simulation horizon [s]", 3.0, 0.5, 10.0, 0.5, "plant"),
        Param("n_iter", "CEM iterations", 30, 5, 100, 1, "learning"),
        Param("pop_size", "Population size", 40, 10, 200, 5, "learning"),
        Param("elite_frac", "Elite fraction", 0.2, 0.05, 0.5, 0.05, "learning"),
        Param("w_overshoot", "Weight: overshoot", 10.0, 0.0, 100.0, 1.0, "learning"),
        Param("w_effort", "Weight: control effort", 0.05, 0.0, 10.0, 0.01, "learning"),
        Param("w_saturation", "Weight: voltage-limit violation", 5.0, 0.0, 100.0, 1.0, "learning"),
        Param("seed", "Random seed", 0, 0, 9999, 1, "learning"),
    ]

    baseline_gains = (100.0, 200.0, 10.0)
    # log10 search box for (Kp, Ki, Kd)
    lower = np.array([-1.0, -1.0, -3.0])
    upper = np.array([3.0, 3.0, 2.0])

    def describe_plant(self, cfg):
        return "P(s) = speed / voltage\n" + str(plant_from_cfg(cfg))

    def _train(self, cfg, callback):
        plant = plant_from_cfg(cfg)

        def batch_cost(pop, it):
            return np.array([cost(plant, 10.0 ** x, cfg) for x in pop])

        res = cem(batch_cost, mean=[1.0, 1.0, -1.0], std=[1.0, 1.0, 1.0],
                  n_iter=int(cfg["n_iter"]), pop_size=int(cfg["pop_size"]),
                  elite_frac=cfg["elite_frac"], lower=self.lower, upper=self.upper,
                  rng=np.random.default_rng(int(cfg["seed"])), callback=callback,
                  stage="CEM")
        # Prefer whichever of the best sample and the final mean scores better.
        cands = [res.x_best, res.x]
        scores = [cost(plant, 10.0 ** c, cfg) for c in cands]
        best = 10.0 ** cands[int(np.argmin(scores))]
        return TrainResult(
            example=self.key, config=cfg,
            controller={"Kp": best[0], "Ki": best[1], "Kd": best[2], "cost": min(scores)},
            baseline={"Kp": self.baseline_gains[0], "Ki": self.baseline_gains[1],
                      "Kd": self.baseline_gains[2],
                      "cost": cost(plant, self.baseline_gains, cfg)},
            history=res.history)

    @staticmethod
    def _gains(d):
        return (d["Kp"], d["Ki"], d["Kd"])

    def evaluate(self, result: TrainResult) -> Evaluation:
        cfg = result.config
        plant = plant_from_cfg(cfg)
        sims = {name: simulate(plant, self._gains(d), cfg)
                for name, d in (("learned", result.controller), ("baseline", result.baseline))}
        colors = {"learned": P.LEARNED, "baseline": P.BASELINE}
        labels = {"learned": "Learned PID (CEM)", "baseline": "Textbook PID"}

        figs = {"Learning curve": P.learning_curve(result.history)}

        # Gain evolution of the CEM distribution mean.
        fig = go.Figure()
        it = list(range(1, len(result.history) + 1))
        names = ["Kp", "Ki", "Kd"]
        for i, (n, c) in enumerate(zip(names, (P.LEARNED, P.BASELINE, P.THIRD))):
            fig.add_trace(P.line(it, [10 ** h["params"][i] for h in result.history], n, c))
        fig.update_yaxes(type="log")
        figs["Gain evolution"] = P.style(fig, "PID gains during learning (CEM mean)",
                                         "Iteration", "Gain (log scale)")

        fig = go.Figure()
        t = sims["learned"][0]
        fig.add_trace(P.line(t, np.full_like(t, cfg["r"]), "Reference", P.REFERENCE, dash="dash"))
        for k, (t, y, u, _) in sims.items():
            fig.add_trace(P.line(t, y, labels[k], colors[k]))
        figs["Step response"] = P.style(fig, "Closed-loop speed step response",
                                        "Time [s]", "Speed [rad/s]")

        fig = go.Figure()
        for k, (t, y, u, _) in sims.items():
            fig.add_trace(P.line(t, u, labels[k], colors[k]))
        for sgn in (1, -1):
            fig.add_trace(P.line(t, np.full_like(t, sgn * cfg["V_max"]), "Voltage limit",
                                 P.REFERENCE, dash="dash", showlegend=sgn > 0))
        u_peak = max(np.abs(s[2]).max() for s in sims.values())
        if u_peak > 4 * cfg["V_max"]:
            fig.update_yaxes(range=[-1.5 * cfg["V_max"], 4 * cfg["V_max"]])
            fig.add_annotation(x=t[-1], y=4 * cfg["V_max"], xanchor="right", yanchor="top",
                               showarrow=False,
                               text=f"axis clipped - peak voltage {u_peak:.0f} V")
        figs["Control voltage"] = P.style(fig, "Control signal (armature voltage)",
                                          "Time [s]", "Voltage [V]")

        figs["Loop Bode plot"] = self._bode(plant, result, labels, colors)
        figs["Gain evolution"] = figs.pop("Gain evolution")          # show it last

        metrics = self._metrics(plant, result, sims)
        return Evaluation(metrics=metrics, figures=figs)

    def _bode(self, plant, result, labels, colors):
        omega = np.logspace(-2, 4, 400)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
        for k, d in (("learned", result.controller), ("baseline", result.baseline)):
            L = pid_tf(*self._gains(d), tf=result.config["Tf"]) * plant
            resp = ct.frequency_response(L, omega)
            mag = np.squeeze(resp.magnitude)
            phase = np.degrees(np.unwrap(np.squeeze(resp.phase)))
            fig.add_trace(P.line(omega, 20 * np.log10(mag), labels[k], colors[k]), 1, 1)
            fig.add_trace(P.line(omega, phase, labels[k], colors[k], showlegend=False), 2, 1)
        fig.add_hline(y=0, line=dict(color=P.REFERENCE, width=1, dash="dash"), row=1, col=1)
        fig.add_hline(y=-180, line=dict(color=P.REFERENCE, width=1, dash="dash"), row=2, col=1)
        fig.update_xaxes(type="log")
        P.style(fig, "Open-loop Bode plot L(s) = C(s)P(s)", height=480)
        fig.update_xaxes(title_text="Frequency [rad/s]", row=2, col=1)
        fig.update_yaxes(title_text="Magnitude [dB]", row=1, col=1)
        fig.update_yaxes(title_text="Phase [deg]", row=2, col=1)
        return fig

    def _metrics(self, plant, result, sims):
        cfg = result.config
        rows = {}
        for k, d in (("learned", result.controller), ("baseline", result.baseline)):
            t, y, u, _ = sims[k]
            info = ct.step_info(y, t, yfinal=cfg["r"])
            gm, pm, *_ = ct.stability_margins(pid_tf(*self._gains(d), tf=cfg["Tf"]) * plant)
            rows[k] = {
                "Kp": d["Kp"], "Ki": d["Ki"], "Kd": d["Kd"],
                "Rise time [s]": info["RiseTime"],
                "Settling time (2 %) [s]": info["SettlingTime"],
                "Overshoot [%]": info["Overshoot"],
                "Final error [%]": 100 * abs(cfg["r"] - y[-1]) / cfg["r"],
                "Peak voltage [V]": float(np.abs(u).max()),
                "Within voltage limit": bool(np.abs(u).max() <= cfg["V_max"]),
                "Phase margin [deg]": pm,
                "Gain margin": gm if np.isfinite(gm) else "inf",
                "Cost": d["cost"],
            }
        return [{"metric": m, "learned": rows["learned"][m], "baseline": rows["baseline"][m]}
                for m in rows["learned"]]
