from __future__ import annotations

import pytest

import hiprof
import hiprof.formula as formula
import hiprof.identification as identification
from hiprof import CheckResult, HPFalsifier, IDAlgorithm
from hiprof.formula import format_ast, parse_and_validate
from hiprof.verification import DegreeBoundEvaluator


def test_package_level_exports() -> None:
    assert hiprof.__all__ == ["CheckResult", "HPFalsifier", "IDAlgorithm"]
    assert hiprof.CheckResult is CheckResult
    assert hiprof.HPFalsifier is HPFalsifier
    assert hiprof.IDAlgorithm is IDAlgorithm


def test_formula_and_verification_subpackage_exports() -> None:
    assert formula.__all__ == ["format_ast", "parse_and_validate"]

    parsed = parse_and_validate("p(Y)")

    assert parsed.signature.outputs
    assert format_ast(parsed.formula).startswith("BaseKernel(")
    assert DegreeBoundEvaluator(1)


def test_id_algorithm_without_y0_raises_informative_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(identification, "_dsl", None)
    monkeypatch.setattr(identification, "_identify_outcomes", None)
    monkeypatch.setattr(identification, "_mixed_graph", None)

    with pytest.raises(
        ImportError,
        match=r"hiprof\[identification\]",
    ):
        identification.IDAlgorithm(
            "X -> Y",
            treatments="X",
            outcomes="Y",
        )
