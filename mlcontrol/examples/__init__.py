"""Registry of the available examples, in increasing order of complexity."""

from mlcontrol.examples.base import Evaluation, Example, Param, TrainResult

EXAMPLES: dict = {}


def register(example_cls):
    EXAMPLES[example_cls.key] = example_cls()
    return example_cls


def get_example(key: str) -> Example:
    return EXAMPLES[key]


__all__ = ["EXAMPLES", "Evaluation", "Example", "Param", "TrainResult", "get_example", "register"]
