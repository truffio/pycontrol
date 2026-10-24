"""Registry of the available examples, in increasing order of complexity."""

from mlcontrol.examples.base import Evaluation, Example, Param, TrainResult

EXAMPLES: dict = {}


def register(example_cls):
    EXAMPLES[example_cls.key] = example_cls()
    return example_cls


def get_example(key: str) -> Example:
    return EXAMPLES[key]


__all__ = ["EXAMPLES", "Evaluation", "Example", "Param", "TrainResult", "get_example", "register"]

# Importing the modules registers the examples (order = display order).
from mlcontrol.examples import dc_motor  # noqa: E402,F401
from mlcontrol.examples import two_mass  # noqa: E402,F401
from mlcontrol.examples import cart_pole  # noqa: E402,F401
