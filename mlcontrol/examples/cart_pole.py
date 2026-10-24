"""Example 3 - Neural network controller for a nonlinear cart-pole.

Plant: an inverted pendulum on a cart, state ``[x, x_dot, theta, theta_dot]``
with ``theta = 0`` upright, driven by a horizontal force limited to
``|F| <= F_max`` on a track of finite length. The full nonlinear dynamics are
wrapped as a ``control.nlsys`` and linearised about upright with
``control.linearize``.

Controller: a tanh multilayer perceptron (4-16-16-1) whose output tanh
respects the force limit.

Learning, in two stages:

1. **Imitation learning (behaviour cloning).** The network is trained by
   supervised regression to reproduce the saturated LQR on random states.
   This gives a safe starting point quickly.
2. **Evolution Strategies fine-tuning.** The network weights are then
   optimised directly on the nonlinear plant, over random and increasingly
   large initial conditions (a curriculum), with a cost that penalises
   state error, effort, and falling or leaving the track.

Training rollouts use a vectorised numpy RK4 simulator (hundreds of
episodes in parallel); final responses are re-simulated with
``control.input_output_response`` on the ``nlsys`` model as a check.
"""

from __future__ import annotations

import control as ct
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mlcontrol import plotting as P
from mlcontrol.examples import register
from mlcontrol.examples.base import Evaluation, Example, Param, TrainResult
from mlcontrol.learners import Adam, IterationInfo, evolution_strategies
from mlcontrol.policies import MLP

STATE_SCALE = np.array([1.0, 2.0, 0.5, 2.0])     # network input normalisation
THETA_FAIL = np.pi / 3                            # the pole is considered fallen


# --------------------------------------------------------------------------- plant
def dynamics(X: np.ndarray, F: np.ndarray, p: dict) -> np.ndarray:
    """Nonlinear cart-pole dynamics, vectorised over leading dimensions.

    ``X[..., :]`` = [x, x_dot, theta, theta_dot]; ``F`` broadcasts with X[..., 0].
    Pole modelled as a uniform rod of half-length ``l`` (Barto et al. 1983).
    """
    M, m, l, g, b = p["M"], p["m"], p["l"], 9.81, p["b_cart"]
    xd, th, thd = X[..., 1], X[..., 2], X[..., 3]
    s, c = np.sin(th), np.cos(th)
    total = M + m
    temp = (F - b * xd + m * l * thd ** 2 * s) / total
    thdd = (g * s - c * temp) / (l * (4.0 / 3.0 - m * c ** 2 / total))
    xdd = temp - m * l * thdd * c / total
    return np.stack([xd, xdd, thd, thdd], axis=-1)


def plant_params(cfg) -> dict:
    return {k: cfg[k] for k in ("M", "m", "l", "b_cart")}


def cartpole_nlsys(cfg) -> ct.NonlinearIOSystem:
    p = plant_params(cfg)

    def updfcn(t, x, u, params):
        return dynamics(np.asarray(x), np.asarray(u)[0], p)

    return ct.nlsys(updfcn, None, inputs=["F"], states=["x", "x_dot", "theta", "theta_dot"],
                    outputs=["x", "x_dot", "theta", "theta_dot"], name="cartpole")


def lqr_gain(cfg) -> np.ndarray:
    """LQR on the model linearised about the upright equilibrium."""
    lin = ct.linearize(cartpole_nlsys(cfg), np.zeros(4), np.zeros(1))
    Q = np.diag([cfg["q_x"], 1.0, cfg["q_theta"], 1.0])
    K, _, _ = ct.lqr(lin.A, lin.B, Q, np.array([[cfg["r_force"]]]))
    return np.asarray(K).ravel()


# ---------------------------------------------------------------------- simulation
def rollout(policy, X0: np.ndarray, cfg: dict, record: bool = False):
    """Vectorised fixed-step RK4 rollout with zero-order-hold control.

    ``policy(X) -> F`` maps states of shape (..., 4) to forces (...,).
    Returns (cost, success) per episode, plus trajectories if ``record``.
    """
    p = plant_params(cfg)
    dt, n = cfg["dt"], int(round(cfg["T_final"] / cfg["dt"]))
    X = X0.astype(float).copy()
    alive = np.ones(X.shape[:-1], dtype=bool)
    cost = np.zeros(X.shape[:-1])
    fail_step = np.full(X.shape[:-1], n)
    traj, forces = ([X.copy()], []) if record else (None, None)
    for k in range(n):
        F = np.clip(policy(X), -cfg["F_max"], cfg["F_max"])
        k1 = dynamics(X, F, p)
        k2 = dynamics(X + 0.5 * dt * k1, F, p)
        k3 = dynamics(X + 0.5 * dt * k2, F, p)
        k4 = dynamics(X + dt * k3, F, p)
        X = X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        # Same quadratic cost as the LQR design: x'Qx + rF^2.
        stage = (cfg["q_x"] * X[..., 0] ** 2 + X[..., 1] ** 2
                 + cfg["q_theta"] * X[..., 2] ** 2 + X[..., 3] ** 2 + cfg["r_force"] * F ** 2)
        cost += np.where(alive, stage * dt, 0.0)
        failed = alive & ((np.abs(X[..., 2]) > THETA_FAIL) | (np.abs(X[..., 0]) > cfg["x_limit"]))
        fail_step = np.where(failed, k, fail_step)
        alive &= ~failed
        if record:
            traj.append(X.copy())
            forces.append(F)
    remaining = (n - fail_step) / n
    cost += np.where(alive, 0.0, cfg["fail_cost"] * (0.5 + remaining))
    success = alive & (np.abs(X[..., 2]) < 0.05) & (np.abs(X[..., 0]) < 0.2)
    if record:
        t = np.arange(n + 1) * dt
        return cost, success, t, np.stack(traj, axis=-2), np.stack(forces, axis=-1)
    return cost, success


def sample_initial_states(cfg, n, rng, frac=1.0) -> np.ndarray:
    """Uniform initial states; ``frac`` scales the range (curriculum)."""
    hi = frac * np.array([cfg["x0_max"], 0.2, cfg["theta0_max"], cfg["thetadot0_max"]])
    return rng.uniform(-hi, hi, size=(n, 4))


def make_mlp(cfg, rng=None) -> MLP:
    h = int(cfg["hidden"])
    return MLP((4, h, h, 1), in_scale=STATE_SCALE, out_scale=cfg["F_max"], rng=rng)


# ----------------------------------------------------------------------- example
@register
class CartPoleExample(Example):
    key = "cart_pole"
    title = "Cart-pole - neural network controller"
    level = "Advanced"
    summary = """
**Plant.** An inverted pendulum on a cart: the full **nonlinear** dynamics
are a `control.nlsys`, with a force limit (default ±10 N) and a finite track
(±2.4 m). The task is to recover the upright position and centre the cart
from large initial angles. A linear controller struggles here: near the edge
of what is physically recoverable it saturates or runs off the track.

**Controller.** A neural network (multilayer perceptron, 4 → 16 → 16 → 1,
tanh) mapping the state to a force; its output is bounded by the force limit.

**Learning algorithm, in two stages:**
1. *Imitation learning*: supervised training (Adam, mean-squared error) to
   copy the saturated LQR on random states, which gives a safe start.
2. *Evolution Strategies*: the network weights are fine-tuned directly on
   the nonlinear simulation, over random and progressively larger initial
   conditions. Each iteration runs hundreds of episodes in parallel in a
   vectorised RK4 simulator.

**Baseline.** An LQR designed with `control.linearize` + `control.lqr`
about the upright equilibrium, with the same force saturation.

**Validation.** The final responses are re-simulated with
`control.input_output_response` on the `nlsys` model.
"""
    params = [
        Param("M", "Cart mass M [kg]", 1.0, 0.2, 5.0, 0.1, "plant"),
        Param("m", "Pole mass m [kg]", 0.1, 0.01, 1.0, 0.01, "plant"),
        Param("l", "Pole half-length l [m]", 0.5, 0.1, 2.0, 0.05, "plant"),
        Param("b_cart", "Cart friction [N s/m]", 0.1, 0.0, 2.0, 0.05, "plant"),
        Param("F_max", "Force limit [N]", 10.0, 2.0, 50.0, 0.5, "plant"),
        Param("x_limit", "Track half-length [m]", 2.4, 0.5, 5.0, 0.1, "plant"),
        Param("T_final", "Episode length [s]", 6.0, 2.0, 15.0, 0.5, "plant"),
        Param("dt", "Control/simulation step [s]", 0.02, 0.005, 0.05, 0.005, "plant"),
        Param("theta0_max", "Max initial angle [rad]", 0.6, 0.1, 1.0, 0.05, "plant"),
        Param("thetadot0_max", "Max initial angular rate [rad/s]", 1.0, 0.0, 3.0, 0.1, "plant"),
        Param("x0_max", "Max initial cart offset [m]", 1.0, 0.0, 2.0, 0.1, "plant"),
        Param("hidden", "Hidden units per layer", 16, 4, 64, 4, "learning"),
        Param("bc_steps", "Imitation learning steps", 1500, 0, 10000, 100, "learning"),
        Param("bc_lr", "Imitation learning rate", 0.01, 0.0005, 0.05, 0.0005, "learning"),
        Param("es_iter", "ES iterations", 150, 0, 1000, 10, "learning"),
        Param("es_pop", "ES antithetic pairs", 24, 4, 128, 4, "learning"),
        Param("es_episodes", "Episodes per candidate", 24, 4, 128, 4, "learning"),
        Param("es_sigma", "ES noise std", 0.03, 0.001, 0.2, 0.005, "learning"),
        Param("es_lr", "ES learning rate", 0.01, 0.001, 0.1, 0.001, "learning"),
        Param("q_x", "Cost weight Q: cart position (LQR and NN)", 10.0, 0.1, 1000.0, 1.0, "learning"),
        Param("q_theta", "Cost weight Q: pole angle (LQR and NN)", 100.0, 1.0, 10000.0, 10.0, "learning"),
        Param("r_force", "Cost weight R: force (LQR and NN)", 0.01, 0.0001, 1.0, 0.001, "learning"),
        Param("fail_cost", "Cost of falling / leaving track", 200.0, 10.0, 2000.0, 10.0, "learning"),
        Param("demo_theta0", "Demo run: initial angle [rad]", 0.5, -1.0, 1.0, 0.05, "plant"),
        Param("demo_x0", "Demo run: initial cart offset [m]", 0.5, -2.0, 2.0, 0.1, "plant"),
        Param("seed", "Random seed", 0, 0, 9999, 1, "learning"),
    ]

    # ------------------------------------------------------------------ train
    def _train(self, cfg, callback):
        rng = np.random.default_rng(int(cfg["seed"]))
        K = lqr_gain(cfg)
        net = make_mlp(cfg, rng)
        history = []

        def lqr_policy(X):
            return np.clip(-X @ K, -cfg["F_max"], cfg["F_max"])

        # Fixed validation set to report a comparable cost in both stages.
        X_val = sample_initial_states(cfg, 64, np.random.default_rng(int(cfg["seed"]) + 7))

        def val_cost(theta):
            c, s = rollout(lambda X: net.forward(X, theta), X_val, cfg)
            return float(c.mean()), float(s.mean())

        # Stage 1: behaviour cloning of the saturated LQR.
        n_bc = int(cfg["bc_steps"])
        n_iter_total = n_bc // 100 + int(cfg["es_iter"])
        lo = -np.array([2.0, 3.0, 1.0, 3.0])
        opt = Adam(net.n_params, lr=cfg["bc_lr"])
        bc_loss = None
        for step in range(1, n_bc + 1):
            Xb = rng.uniform(lo, -lo, size=(256, 4))
            bc_loss, grad = net.mse_loss_and_grad(Xb, lqr_policy(Xb))
            net.theta = net.theta + opt.step(grad)
            if step % 100 == 0:
                vc, vs = val_cost(net.theta)
                info = IterationInfo(len(history) + 1, n_iter_total, vc, vc, np.empty(0),
                                     "Imitation (behaviour cloning)", validation_cost=vc)
                history.append({"iteration": info.iteration, "best_cost": vc,
                                "mean_cost": vc, "validation_cost": vc, "imitation_mse": bc_loss,
                                "success_rate": vs, "params": [], "stage": info.stage})
                if callback:
                    callback(info)
        theta_bc = net.theta.copy()

        # Stage 2: ES fine-tuning on the nonlinear plant with a curriculum.
        n_es = int(cfg["es_iter"])
        n_ep = int(cfg["es_episodes"])

        def es_cost(pop, it):
            frac = min(1.0, 0.6 + 0.8 * it / max(n_es, 1))     # grow IC range to 100 %
            X0 = sample_initial_states(cfg, n_ep, rng, frac)
            Xb = np.broadcast_to(X0, (pop.shape[0], n_ep, 4))
            c, _ = rollout(lambda X: net.forward_batch(X, pop), Xb, cfg)
            return c.mean(axis=1)

        offset = len(history)
        best = {"cost": val_cost(theta_bc)[0], "theta": theta_bc}   # best validated snapshot
        validations = {}

        def es_callback(info):
            vcost = None
            if info.iteration % 5 == 0 or info.iteration == n_es:
                vcost = val_cost(info.params)[0]
                validations[info.iteration] = vcost
                if vcost < best["cost"]:
                    best.update(cost=vcost, theta=info.params.copy())
            if callback:
                callback(IterationInfo(offset + info.iteration, n_iter_total, info.best_cost,
                                       info.mean_cost, info.params, info.stage, vcost))

        if n_es > 0:
            res = evolution_strategies(es_cost, theta_bc, n_iter=n_es, pop_size=int(cfg["es_pop"]),
                                       sigma=cfg["es_sigma"], lr=cfg["es_lr"], rng=rng,
                                       callback=es_callback, stage="Evolution Strategies",
                                       record_params=True)
            for h in res.history:
                if h["iteration"] in validations:
                    h["validation_cost"] = validations[h["iteration"]]
                h["iteration"] += offset
                h["params"] = []                 # weights are not kept per iteration
            history += res.history
        theta = best["theta"]
        vc, vs = val_cost(theta)
        bc_c, bc_s = val_cost(theta_bc)
        lqr_c, lqr_s = rollout(lqr_policy, X_val, cfg)
        return TrainResult(
            example=self.key, config=cfg,
            controller={"theta": theta.tolist(), "sizes": list(net.sizes),
                        "validation_cost": vc, "validation_success": vs},
            baseline={"K": K.tolist(), "validation_cost": float(lqr_c.mean()),
                      "validation_success": float(lqr_s.mean())},
            history=history,
            extras={"imitation_mse": bc_loss, "bc_theta": theta_bc.tolist(),
                    "bc_validation_cost": bc_c, "bc_validation_success": bc_s})

    # --------------------------------------------------------------- evaluate
    def _policies(self, result):
        cfg = result.config
        net = make_mlp(cfg)
        theta = np.array(result.controller["theta"])
        K = np.array(result.baseline["K"])
        nn = lambda X: net.forward(X.reshape(-1, 4), theta).reshape(X.shape[:-1])  # noqa: E731
        lqr = lambda X: np.clip(-X @ K, -cfg["F_max"], cfg["F_max"])  # noqa: E731
        return {"learned": nn, "baseline": lqr}

    def evaluate(self, result: TrainResult) -> Evaluation:
        cfg = result.config
        pols = self._policies(result)
        colors = {"learned": P.LEARNED, "baseline": P.BASELINE}
        labels = {"learned": "Neural network (BC + ES)", "baseline": "LQR (saturated)"}

        figs = {"Learning curve": P.learning_curve(result.history)}

        X0 = np.array([cfg["demo_x0"], 0.0, cfg["demo_theta0"], 0.0])
        sims = {k: self._validate_with_python_control(cfg, pol, X0) for k, pol in pols.items()}
        figs["Time response"] = self._time_figure(sims, cfg, labels, colors)

        th_grid = np.linspace(-0.9, 0.9, 37)
        thd_grid = np.linspace(-2.5, 2.5, 31)
        TH, THD = np.meshgrid(th_grid, thd_grid)
        Xg = np.zeros(TH.shape + (4,))
        Xg[..., 2], Xg[..., 3] = TH, THD
        roa = {k: rollout(pol, Xg, cfg)[1] for k, pol in pols.items()}
        figs["Region of attraction"] = self._roa_figure(th_grid, thd_grid, roa, labels)
        figs["Animation"] = self._animation(pols, X0, cfg, labels, colors)

        # Monte-Carlo test over the training distribution of initial states.
        X_test = sample_initial_states(cfg, 400, np.random.default_rng(2024))
        test = {k: rollout(pol, X_test, cfg) for k, pol in pols.items()}

        rows = {}
        for k in pols:
            t, X, F, ok = sims[k]
            settle = _settling_time(t, X[:, 2], 0.05)
            c, s = test[k]
            rows[k] = {
                "Random initial states recovered [%]": 100 * float(s.mean()),
                "Mean episode cost (random states)": float(c.mean()),
                "Region-of-attraction grid recovered [%]": 100 * float(roa[k].mean()),
                f"Demo run (θ0={cfg['demo_theta0']:.2f} rad, x0={cfg['demo_x0']:.1f} m)":
                    "recovered" if ok else "failed",
                "Demo: angle settling time (±0.05 rad) [s]": settle,
                "Demo: max cart excursion [m]": float(np.abs(X[:, 0]).max()),
                "Demo: RMS force [N]": float(np.sqrt(np.mean(F ** 2))),
            }
        metrics = [{"metric": m, "learned": rows["learned"][m], "baseline": rows["baseline"][m]}
                   for m in rows["learned"]]
        metrics.append({"metric": "Imitation-only network: validation states recovered [%]",
                        "learned": 100 * result.extras.get("bc_validation_success", float("nan")),
                        "baseline": "-"})
        return Evaluation(metrics=metrics, figures=figs)

    @staticmethod
    def _validate_with_python_control(cfg, policy, X0):
        """Closed-loop simulation with control.input_output_response on the nlsys plant."""
        plant = cartpole_nlsys(cfg)
        ctrl = ct.nlsys(None, lambda t, x, u, params: np.atleast_1d(
                            np.clip(policy(np.asarray(u)[None, :])[0], -cfg["F_max"], cfg["F_max"])),
                        inputs=["x", "x_dot", "theta", "theta_dot"], outputs=["F"], name="controller")
        cl = ct.interconnect([plant, ctrl], inplist=[], outlist=["x", "x_dot", "theta", "theta_dot", "F"],
                             name="closed_loop")
        t = np.arange(0, cfg["T_final"] + 1e-9, cfg["dt"])
        resp = ct.input_output_response(cl, t, 0, X0, solve_ivp_kwargs={"max_step": cfg["dt"]})
        out = np.asarray(resp.outputs)
        X, F = out[:4].T, out[4]
        fell = (np.abs(X[:, 2]) > THETA_FAIL) | (np.abs(X[:, 0]) > cfg["x_limit"])
        if fell.any():                                       # stop plotting once it has failed
            idx = int(np.argmax(fell)) + 1
            t, X, F = t[:idx], X[:idx], F[:idx]
        ok = (not fell.any()) and abs(X[-1, 2]) < 0.05 and abs(X[-1, 0]) < 0.2
        return t, X, F, ok

    @staticmethod
    def _time_figure(sims, cfg, labels, colors):
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06)
        for k, (t, X, F, ok) in sims.items():
            name = labels[k] + ("" if ok else " - failed")
            fig.add_trace(P.line(t, np.degrees(X[:, 2]), name, colors[k]), 1, 1)
            fig.add_trace(P.line(t, X[:, 0], name, colors[k], showlegend=False), 2, 1)
            fig.add_trace(P.line(t, F, name, colors[k], showlegend=False), 3, 1)
        for sgn in (1, -1):
            fig.add_hline(y=sgn * cfg["x_limit"], line=dict(color=P.REFERENCE, width=1, dash="dash"),
                          row=2, col=1)
            fig.add_hline(y=sgn * cfg["F_max"], line=dict(color=P.REFERENCE, width=1, dash="dash"),
                          row=3, col=1)
        P.style(fig, "Nonlinear closed-loop response (simulated with control.input_output_response)",
                height=620)
        fig.update_yaxes(title_text="Pole angle [deg]", row=1, col=1)
        fig.update_yaxes(title_text="Cart position [m]", row=2, col=1)
        fig.update_yaxes(title_text="Force [N]", row=3, col=1)
        fig.update_xaxes(title_text="Time [s]", row=3, col=1)
        return fig

    @staticmethod
    def _roa_figure(th_grid, thd_grid, roa, labels):
        nn, lq = roa["learned"], roa["baseline"]
        # 0 neither, 1 baseline only, 2 learned only, 3 both
        z = nn.astype(int) * 2 + lq.astype(int)
        cats = ["Neither", f"{labels['baseline']} only", f"{labels['learned']} only", "Both"]
        cols = ["#e8e7e3", P.BASELINE, P.LEARNED, P.THIRD]
        scale = []
        for i, c in enumerate(cols):
            scale += [(i / 4, c), ((i + 1) / 4, c)]
        text = np.array(cats, dtype=object)[z]
        fig = go.Figure(go.Heatmap(
            x=np.degrees(th_grid), y=thd_grid, z=z, zmin=-0.5, zmax=3.5, colorscale=scale,
            text=text, hovertemplate="θ0 %{x:.1f}°<br>θ̇0 %{y:.2f} rad/s<br>%{text}<extra></extra>",
            colorbar=dict(tickvals=[0, 1, 2, 3], ticktext=cats, len=0.9), xgap=1, ygap=1))
        P.style(fig, f"Recovered initial states (cart starts centred) - "
                     f"NN {100 * nn.mean():.0f} %, LQR {100 * lq.mean():.0f} % of grid",
                "Initial pole angle θ0 [deg]", "Initial angular rate θ̇0 [rad/s]", 460)
        fig.update_layout(hovermode="closest")
        return fig

    @staticmethod
    def _animation(pols, X0, cfg, labels, colors):
        """Side-by-side cart-pole animation from the demo initial state."""
        trajs = {}
        for k, pol in pols.items():
            _, _, t, X, _ = rollout(pol, X0[None, :], cfg, record=True)
            trajs[k] = X[0]
        stride = max(1, int(round(0.05 / cfg["dt"])))
        idx = list(range(0, len(t), stride))
        L = 2 * cfg["l"]
        rows = {"learned": 1.0, "baseline": -1.0}           # vertical offset of each track

        def shapes(i):
            data = []
            for k, X in trajs.items():
                x, th = X[i, 0], X[i, 2]
                y0 = rows[k]
                data.append(go.Scatter(x=[x - 0.2, x + 0.2, x + 0.2, x - 0.2, x - 0.2],
                                       y=[y0 - 0.1, y0 - 0.1, y0 + 0.1, y0 + 0.1, y0 - 0.1],
                                       fill="toself", mode="lines", line=dict(color=colors[k], width=2),
                                       name=labels[k], showlegend=True))
                data.append(go.Scatter(x=[x, x + L * np.sin(th)], y=[y0, y0 + L * np.cos(th) * 0.8],
                                       mode="lines+markers", line=dict(color=colors[k], width=4),
                                       marker=dict(size=[4, 10]), showlegend=False))
            return data

        fig = go.Figure(data=shapes(0),
                        frames=[go.Frame(data=shapes(i), name=f"{t[i]:.2f}") for i in idx])
        for y0 in rows.values():
            fig.add_shape(type="line", x0=-cfg["x_limit"], x1=cfg["x_limit"], y0=y0 - 0.1, y1=y0 - 0.1,
                          line=dict(color=P.REFERENCE, width=1))
        fig.update_layout(
            xaxis=dict(range=[-cfg["x_limit"] - 0.3, cfg["x_limit"] + 0.3], title="Cart position [m]"),
            yaxis=dict(range=[-1.4, 2.4], visible=False, scaleanchor="x"),
            updatemenus=[dict(type="buttons", showactive=False, x=0, y=-0.15, xanchor="left",
                              buttons=[dict(label="▶ Play", method="animate",
                                            args=[None, dict(frame=dict(duration=50, redraw=True),
                                                             fromcurrent=True, transition=dict(duration=0))]),
                                       dict(label="❚❚ Pause", method="animate",
                                            args=[[None], dict(mode="immediate", frame=dict(duration=0))])])],
            sliders=[dict(currentvalue=dict(prefix="t = ", suffix=" s"), x=0.15, len=0.85, y=-0.1,
                          steps=[dict(method="animate", label=f"{t[i]:.1f}",
                                      args=[[f"{t[i]:.2f}"], dict(mode="immediate",
                                                                  frame=dict(duration=0, redraw=True))])
                                 for i in idx])])
        P.style(fig, "Animation from the demo initial state (top: NN, bottom: LQR)", height=480)
        fig.update_layout(hovermode=False)
        return fig


def _settling_time(t, y, band):
    outside = np.nonzero(np.abs(y) > band)[0]
    if outside.size == 0:
        return 0.0
    if outside[-1] == len(y) - 1:
        return float("nan")
    return float(t[outside[-1] + 1])
