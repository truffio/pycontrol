"""Streamlit UI: pick an example, tune settings, train, and assess the results.

Run with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # allow running from a checkout

from mlcontrol import plotting as P  # noqa: E402
from mlcontrol.examples import EXAMPLES  # noqa: E402

st.set_page_config(page_title="ML controller design", page_icon="🎛️", layout="wide")

if "results" not in st.session_state:
    st.session_state.results = {}          # example key -> (TrainResult, Evaluation)


# ----------------------------------------------------------------- helpers
def fmt(v):
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)):
        if isinstance(v, float) and math.isnan(v):
            return "-"
        return f"{v:.4g}"
    return str(v)


def param_widget(ex, p):
    key = f"{ex.key}.{p.key}"
    if key not in st.session_state:
        st.session_state[key] = p.default
    if p.is_int:
        return st.number_input(p.label, min_value=int(p.min), max_value=int(p.max),
                               step=int(p.step or 1), key=key, help=p.help or None)
    return st.number_input(p.label, min_value=float(p.min), max_value=float(p.max),
                           step=float(p.step) if p.step else None, key=key,
                           format="%.4g", help=p.help or None)


def reset_defaults(ex):
    for p in ex.params:
        st.session_state[f"{ex.key}.{p.key}"] = p.default


def live_curve(history):
    fig = go.Figure()
    it = [h["iteration"] for h in history]
    fig.add_trace(P.line(it, [h["mean_cost"] for h in history], "Population mean (training batch)", P.REFERENCE, dash="dot"))
    fig.add_trace(P.line(it, [h["best_cost"] for h in history], "Best candidate (training batch)", P.LEARNED))
    val = [(h["iteration"], h["validation_cost"]) for h in history if h.get("validation_cost") is not None]
    if val:
        fig.add_trace(go.Scatter(x=[v[0] for v in val], y=[v[1] for v in val], mode="lines+markers",
                                 name="Validation", line=dict(color=P.THIRD, width=2), marker=dict(size=7)))
    fig.update_yaxes(type="log")
    return P.style(fig, "Learning progress", "Iteration", "Cost (log scale)", height=320)


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🎛️ ML controller design")
    st.caption("Plants simulated with python-control; controllers designed by machine learning.")
    keys = list(EXAMPLES)
    labels = {f"{i + 1}. {EXAMPLES[k].title} ({EXAMPLES[k].level})": k for i, k in enumerate(keys)}
    ex = EXAMPLES[labels[st.radio("Example", list(labels))]]

    with st.expander("Plant settings", expanded=False):
        cfg = {p.key: param_widget(ex, p) for p in ex.params if p.group == "plant"}
    with st.expander("Learning settings", expanded=False):
        cfg |= {p.key: param_widget(ex, p) for p in ex.params if p.group == "learning"}

    train_clicked = st.button("▶ Train controller", type="primary", use_container_width=True)
    st.button("Reset settings to defaults", on_click=reset_defaults, args=(ex,), use_container_width=True)

# ----------------------------------------------------------------- main
st.header(f"{ex.title}")
st.caption(f"Example {keys.index(ex.key) + 1} of {len(keys)} · complexity: {ex.level}")

stored = st.session_state.results.get(ex.key)
col_text, col_model = st.columns([3, 2])
with col_text:
    with st.expander("How this example works", expanded=stored is None):
        st.markdown(ex.summary)
with col_model:
    with st.expander("Plant model (python-control)", expanded=False):
        try:
            st.code(ex.describe_plant(cfg), language="text")
        except Exception as exc:                       # invalid parameter combination
            st.warning(f"Cannot build the plant with these settings: {exc}")

if train_clicked:
    progress = st.progress(0.0, text="Starting…")
    chart = st.empty()
    history = []

    def on_iteration(info):
        history.append({"iteration": info.iteration, "best_cost": info.best_cost,
                        "mean_cost": info.mean_cost, "validation_cost": info.validation_cost})
        progress.progress(min(1.0, info.iteration / max(info.n_iter, 1)),
                          text=f"{info.stage}: iteration {info.iteration}/{info.n_iter} · "
                               f"cost {info.best_cost:.4g}")
        every = max(1, info.n_iter // 40)
        if info.iteration % every == 0 or info.iteration == info.n_iter:
            chart.plotly_chart(live_curve(history), use_container_width=True)

    try:
        result = ex.train(cfg, callback=on_iteration)
        with st.spinner("Evaluating learned and baseline controllers…"):
            evaluation = ex.evaluate(result)
    except Exception as exc:
        progress.empty()
        st.error(f"Training failed: {exc}")
        st.stop()
    st.session_state.results[ex.key] = (result, evaluation)
    stored = (result, evaluation)
    progress.empty()
    chart.empty()

if stored is None:
    st.info("Adjust the settings in the sidebar if you like, then press **Train controller**.")
    st.stop()

result, evaluation = stored
if result.config != cfg:
    st.warning("Settings have changed since this controller was trained. Press **Train controller** "
               "to retrain with the new settings.")

st.success(f"Controller learned in {result.elapsed_s:.1f} s over {len(result.history)} iterations.")

left, right = st.columns([1, 1])
with left:
    st.subheader("Learned vs baseline")
    rows = [{"Metric": r["metric"], "Learned": fmt(r["learned"]), "Baseline": fmt(r["baseline"])}
            for r in evaluation.metrics]
    st.dataframe(rows, hide_index=True, use_container_width=True, height=38 + 35 * len(rows),
                 column_config={"Metric": st.column_config.TextColumn(width="medium")})
    st.download_button("Download result (JSON)", json.dumps(result.to_dict(), indent=1, default=float),
                       file_name=f"{ex.key}_result.json", mime="application/json")
with right:
    st.subheader("Learning curve")
    curve = go.Figure(evaluation.figures["Learning curve"]).update_layout(title_text="")
    st.plotly_chart(curve, use_container_width=True)

names = [n for n in evaluation.figures if n != "Learning curve"]
for tab, name in zip(st.tabs(names), names):
    with tab:
        st.plotly_chart(evaluation.figures[name], use_container_width=True)
