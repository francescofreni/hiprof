from __future__ import annotations

import importlib.util

import pytest

from hiprof import HPFalsifier


def test_check_none_without_ananke_raises_informative_error() -> None:
    if importlib.util.find_spec("ananke") is not None:
        pytest.skip("ananke-causal is installed")

    falsifier = HPFalsifier("X -> Y; X <-> Y", treatments="X", outcomes="Y")

    with pytest.raises(ImportError, match="hiprof\\[nonidentifiability\\]"):
        falsifier.check(None)


def test_check_none_with_ananke_if_installed() -> None:
    if importlib.util.find_spec("ananke") is None:
        pytest.skip("ananke-causal is not installed")

    falsifier = HPFalsifier("X -> Y; X <-> Y", treatments="X", outcomes="Y")
    result = falsifier.check(None)

    assert isinstance(result.accepted, bool)
    assert result.repetitions == 0
