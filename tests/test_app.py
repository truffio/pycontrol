from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def test_app_loads_every_example():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    for label in at.sidebar.radio[0].options:
        at.sidebar.radio[0].set_value(label).run()
        assert not at.exception
        assert at.info, "prompt to train should be shown before training"


def test_train_button_produces_results():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.number_input(key="dc_motor.n_iter").set_value(5)
    at.number_input(key="dc_motor.pop_size").set_value(10)
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert at.success, "training summary should be shown"
    assert len(at.dataframe) == 1
    assert len(at.tabs) == 4
