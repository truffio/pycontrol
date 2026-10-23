"""Shared Plotly styling so every example's figures read as one system."""

from __future__ import annotations

import plotly.graph_objects as go

# Validated categorical order (blue, orange, aqua); colour follows the entity.
LEARNED = "#2a78d6"
BASELINE = "#eb6834"
THIRD = "#1baf7a"
REFERENCE = "#8a8984"
GRID = "rgba(128,128,128,0.18)"


def style(fig: go.Figure, title: str = "", xlabel: str = "", ylabel: str = "",
          height: int = 360) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15)),
        height=height,
        margin=dict(l=60, r=20, t=50, b=50),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(title_text=xlabel, gridcolor=GRID, zeroline=False, showline=False)
    fig.update_yaxes(title_text=ylabel, gridcolor=GRID, zeroline=False, showline=False)
    return fig


def line(x, y, name, color, dash=None, width=2, showlegend=True, **kw) -> go.Scatter:
    return go.Scatter(x=x, y=y, name=name, mode="lines", showlegend=showlegend,
                      line=dict(color=color, width=width, dash=dash), **kw)


def learning_curve(history: list, title: str = "Learning curve") -> go.Figure:
    """Best and population-mean cost per iteration, split by training stage."""
    fig = go.Figure()
    it = list(range(1, len(history) + 1))
    best = [h["best_cost"] for h in history]
    mean = [h["mean_cost"] for h in history]
    fig.add_trace(line(it, mean, "Population mean (training batch)", REFERENCE, dash="dot"))
    fig.add_trace(line(it, best, "Best candidate (training batch)", LEARNED))
    val = [(i, h["validation_cost"]) for i, h in zip(it, history) if h.get("validation_cost") is not None]
    if val:
        fig.add_trace(go.Scatter(x=[v[0] for v in val], y=[v[1] for v in val], mode="lines+markers",
                                 name="Validation (fixed held-out set)",
                                 line=dict(color=THIRD, width=2), marker=dict(size=8)))
    # Mark stage boundaries (e.g. curriculum or pre-training -> fine-tuning).
    stages = [h.get("stage", "") for h in history]
    n_labels = 0
    for i in range(len(stages)):
        if len(set(stages)) > 1 and (i == 0 or stages[i] != stages[i - 1]):
            # Stagger the labels vertically so close stage boundaries stay readable.
            fig.add_vline(x=i + 0.5, line=dict(color=REFERENCE, width=1, dash="dash"),
                          annotation=dict(text=stages[i], yshift=-16 * n_labels, xanchor="left",
                                          font=dict(size=11)),
                          annotation_position="top right")
            n_labels += 1
    fig.update_yaxes(type="log")
    return style(fig, title, "Iteration", "Cost (log scale)")
