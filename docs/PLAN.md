# Project plan — ML-designed controllers with python-control

## Goal

Show that a controller can be **designed by a machine-learning algorithm** that
learns from a **simulated plant** built with
[python-control](https://github.com/python-control/python-control).
Three examples of increasing complexity each define a plant, a learning
algorithm that produces a controller, and a classical baseline to compare
against. A Streamlit UI starts training and shows graphs to judge the results.

## Approach

All three examples use **simulation-based policy search**, a model-free
reinforcement-learning approach: the learner proposes controller parameters,
the plant is simulated in closed loop, a scalar cost is computed from the
response, and the learner updates its search distribution. The learner only
sees the cost, never the plant equations, so the same method would work on a
real rig or a black-box simulator.

| | Example 1 (basic) | Example 2 (intermediate) | Example 3 (advanced) |
|---|---|---|---|
| Plant | DC motor speed (2nd order, linear) | Two-mass flexible drive (4th order, lightly damped, non-collocated) | Cart-pole inverted pendulum (nonlinear, actuator saturation) |
| Controller structure | PID with derivative filter (3 params) | State feedback + integral action (5 params) | Neural network, MLP 4-16-16-1 (~370 params) |
| Learning algorithm | Cross-Entropy Method (CEM) | CEM with **domain randomisation** (robust to ±30 % plant uncertainty) | **Imitation learning** (behaviour cloning from LQR), then **Evolution Strategies** fine-tuning |
| Simulation | `control.feedback` + `control.step_response` | `control.ss` interconnection + `control.forced_response` on a plant ensemble | Batched numpy RK4 for training; `control.nlsys` + `input_output_response` for validation |
| Baseline | Textbook hand-tuned PID | `control.lqr` (nominal model) | `control.linearize` + `control.lqr`, with saturation |
| Assessment graphs | learning curve, gain evolution, step response, control effort, loop Bode + margins | learning curve, nominal + ensemble responses, closed-loop poles, Bode | learning curve, time responses, success-rate map over initial conditions (region of attraction), animation |

The advanced example stays reasonable: a stabilisation task (not swing-up)
from large initial angles under force saturation, where a linear controller
starts to fail and a learned nonlinear policy can do better.

## Architecture

```
pycontrol/
├── mlcontrol/              # library
│   ├── learners.py         # CEM, Evolution Strategies (Adam), shared result types
│   ├── policies.py         # MLP policy (numpy forward/backprop)
│   ├── metrics.py          # cost terms, step metrics
│   └── examples/
│       ├── base.py         # common Example interface (train / evaluate / baseline)
│       ├── dc_motor.py     # Example 1
│       ├── two_mass.py     # Example 2
│       └── cart_pole.py    # Example 3
├── app/streamlit_app.py    # UI
├── tests/                  # pytest
└── docs/PLAN.md
```

Each example exposes the same interface so the UI is generic:
`default_config()`, `build_plant(cfg)`, `baseline(cfg)`,
`train(cfg, callback)`, `evaluate(cfg, controller)` and `figures(...)`.

Dependencies: `control`, `numpy`, `scipy`, `plotly`, `streamlit`, `pytest`.
No deep-learning framework is needed; the small networks and optimisers are
written in numpy so the learning steps are easy to follow.

## Work breakdown (tracked as GitHub issues)

- **Epic 1 — Foundation:** packaging, learners (CEM, ES), shared interface, CI tests
- **Epic 2 — Example 1, DC motor PID:** plant, CEM tuning, baseline, figures
- **Epic 3 — Example 2, two-mass robust state feedback:** plant and ensemble, domain-randomised CEM, LQR baseline, figures
- **Epic 4 — Example 3, cart-pole neural controller:** nonlinear plant, vectorised simulator, behaviour cloning, ES fine-tuning, LQR baseline, region-of-attraction analysis, animation
- **Epic 5 — User interface:** Streamlit app shell, training controls with live progress, results dashboards
- **Epic 6 — Documentation and QA:** README, usage guide, tests, results write-up

## Delivery

One branch and one pull request per epic. Commits reference their feature
issues (`Closes #n`), and each PR is merged into `main` once its tests pass.
