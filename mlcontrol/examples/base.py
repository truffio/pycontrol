"""Common interface implemented by every example.

The UI and CLI only talk to this interface, so adding an example means
subclassing :class:`Example` and registering it in
:mod:`mlcontrol.examples`.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

from mlcontrol.learners import IterationInfo

Callback = Optional[Callable[[IterationInfo], None]]


@dataclass
class Param:
    """A user-configurable setting, rendered as a widget by the UI."""

    key: str
    label: str
    default: float
    min: float
    max: float
    step: Optional[float] = None
    group: str = "learning"          # "plant" or "learning"
    help: str = ""

    @property
    def is_int(self) -> bool:
        return isinstance(self.default, int) and not isinstance(self.default, bool)


@dataclass
class TrainResult:
    """Everything produced by one training run (JSON-serialisable)."""

    example: str
    config: dict
    controller: dict                 # learned controller parameters
    baseline: dict                   # classical baseline parameters
    history: list                    # per-iteration learning progress
    elapsed_s: float = 0.0
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Evaluation:
    """Assessment of a trained controller against the baseline."""

    metrics: list                    # rows: {"metric": ..., "learned": ..., "baseline": ...}
    figures: dict                    # name -> plotly Figure


class Example(ABC):
    key: str = ""
    title: str = ""
    level: str = ""
    summary: str = ""                # markdown description shown in the UI
    params: list = []

    def default_config(self) -> dict:
        return {p.key: p.default for p in self.params}

    def make_config(self, overrides: Optional[dict] = None) -> dict:
        cfg = self.default_config()
        for k, v in (overrides or {}).items():
            if k not in cfg:
                raise KeyError(f"Unknown parameter {k!r} for example {self.key!r}")
            cfg[k] = v
        return cfg

    def train(self, config: Optional[dict] = None, callback: Callback = None) -> TrainResult:
        cfg = self.make_config(config)
        t0 = time.perf_counter()
        result = self._train(cfg, callback)
        result.elapsed_s = time.perf_counter() - t0
        return result

    @abstractmethod
    def _train(self, cfg: dict, callback: Callback) -> TrainResult:
        """Run the learning algorithm and return the learned controller."""

    @abstractmethod
    def evaluate(self, result: TrainResult) -> Evaluation:
        """Simulate learned and baseline controllers and build figures."""
