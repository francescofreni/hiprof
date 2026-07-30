from __future__ import annotations

import importlib.util

import pytest

import hiprof
from hiprof import CheckResult, HPFalsifier, IDAlgorithm
from hiprof.formula import parse_and_validate
from hiprof.verification import DegreeBoundEvaluator


def test_package_level_exports() -> None:
    assert hiprof.__all__ == ["CheckResult", "HPFalsifier", "IDAlgorithm"]
    assert hiprof.CheckResult is CheckResult
    assert hiprof.HPFalsifier is HPFalsifier
    assert hiprof.IDAlgorithm is IDAlgorithm


def test_formula_and_verification_subpackage_exports() -> None:
    assert parse_and_validate("p(Y)").signature.outputs
    assert DegreeBoundEvaluator(1)


def test_id_algorithm_without_y0_raises_informative_error() -> None:
    if importlib.util.find_spec("y0") is not None:
        pytest.skip("y0 is installed")

    with pytest.raises(ImportError, match="hiprof\\[identification\\]"):
        IDAlgorithm("X -> Y", treatments="X", outcomes="Y")


def test_id_algorithm_with_y0_if_installed() -> None:
    if importlib.util.find_spec("y0") is None:
        pytest.skip("y0 is not installed")

    formula = IDAlgorithm("X -> Y", treatments="X", outcomes="Y").run()

    assert formula is None or isinstance(formula, str)
