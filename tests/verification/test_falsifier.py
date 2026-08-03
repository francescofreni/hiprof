from __future__ import annotations

from fractions import Fraction

import pytest

from hiprof import CheckResult, HPFalsifier
from hiprof.graph import parse_graph
from hiprof.verification.degree import DegreeBound
from hiprof.verification.falsifier import (
    _equality_test_degree,
    _minimum_bits_below_one,
    _repeat_until_target,
    _validate_target_bound,
    _zippel_ratio,
)


NAPKIN_GRAPH = "X -> Y; W -> Z; Z -> X; X <-> W; W <-> Y"

NAPKIN_CONDITIONAL_FORMULA = (
    "icd_{X | Z} { "
    "sum_{W} { p(Y, X | Z, W) p(W) } "
    "}"
)

NAPKIN_SUMMED_FORMULA = (
    "sum_{Z} { "
    "icd_{X | Z} { "
    "sum_{W} { p(Y, X | Z, W) p(W) } "
    "} "
    "p(Z) "
    "}"
)


def fixed_getrandbits(bits: int) -> int:
    return 1 if bits else 0


def test_check_result_bool_string_and_repr() -> None:
    accepted = CheckResult(
        accepted=True,
        false_acceptance_bound=Fraction(1, 100),
        degree=2,
        entropy_bits=8,
        repetitions=1,
    )

    assert bool(accepted)
    assert str(accepted).startswith("True\nFalse-acceptance bound:")
    assert repr(accepted) == str(accepted)
    assert str(CheckResult(accepted=True)) == "True"
    assert str(CheckResult(accepted=False)) == "False"


def test_target_bound_validation_accepts_fraction_and_float() -> None:
    assert _validate_target_bound(Fraction(1, 10)) == Fraction(1, 10)
    assert _validate_target_bound(0.25) == Fraction(1, 4)


@pytest.mark.parametrize(
    "target_bound",
    [0.0, 1.0, Fraction(0), Fraction(1)],
)
def test_target_bound_validation_rejects_out_of_range(
    target_bound: Fraction | float,
) -> None:
    with pytest.raises(ValueError, match="strictly between"):
        _validate_target_bound(target_bound)


def test_bound_planning_helpers_are_deterministic() -> None:
    degree = _equality_test_degree(DegreeBound(2, 3, 4, 5), 3)

    assert degree == 10
    assert _zippel_ratio(10, 4) == Fraction(10, 16)
    assert _repeat_until_target(Fraction(1, 2), Fraction(1, 8)) == (
        3,
        Fraction(1, 8),
    )
    assert _minimum_bits_below_one(64, 1) == 7


def test_bound_helpers_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _zippel_ratio(-1, 4)
    with pytest.raises(ValueError, match="positive"):
        _zippel_ratio(1, 0)
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        _repeat_until_target(Fraction(1), Fraction(1, 2))
    with pytest.raises(TypeError, match="Fraction or float"):
        _validate_target_bound("0.1")  # type: ignore[arg-type]


def test_falsifier_validates_graph_and_query_inputs() -> None:
    with pytest.raises(ValueError, match="directed cycle"):
        parse_graph("X -> Y; Y -> X")
    with pytest.raises(ValueError, match="Self-edge"):
        parse_graph("X -> X")
    with pytest.raises(ValueError, match="duplicate"):
        HPFalsifier("X -> Y", treatments=("X", "X"), outcomes="Y")
    with pytest.raises(ValueError, match="disjoint"):
        HPFalsifier("X -> Y", treatments="X", outcomes="X")


def test_check_rejects_formula_with_wrong_outputs() -> None:
    falsifier = HPFalsifier("X -> Y", treatments="X", outcomes="Y")

    with pytest.raises(ValueError, match="yield exactly"):
        falsifier.check("p(X)")


def test_check_rejects_variables_not_in_graph() -> None:
    falsifier = HPFalsifier("X -> Y", treatments="X", outcomes="Y")

    with pytest.raises(ValueError, match="observed variables"):
        falsifier.check("p(Y | Z)")


def test_check_accepts_backdoor_adjustment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "hiprof.verification.falsifier.getrandbits",
        fixed_getrandbits,
    )
    falsifier = HPFalsifier(
        graph="C -> X; C -> Y; X -> Y",
        treatments="X",
        outcomes="Y",
    )

    result = falsifier.check("sum_{C} { p(Y | X, C) p(C) }")

    assert result.accepted
    assert result.false_acceptance_bound is not None
    assert result.repetitions >= 1


def test_check_accepts_frontdoor_formula(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "hiprof.verification.falsifier.getrandbits",
        fixed_getrandbits,
    )
    falsifier = HPFalsifier(
        graph="Z -> Y; X -> Z; X <-> Y",
        treatments="X",
        outcomes="Y",
    )

    result = falsifier.check(
        "sum_{Z} { p(Z | X) sum_{X'} { p(Y | X', Z) p(X') } }"
    )

    assert result.accepted


def test_napkin_formula_summed_over_z_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "hiprof.verification.falsifier.getrandbits",
        fixed_getrandbits,
    )
    falsifier = HPFalsifier(
        graph=NAPKIN_GRAPH,
        treatments="X",
        outcomes="Y",
    )

    result = falsifier.check(NAPKIN_SUMMED_FORMULA)

    assert result.accepted


def test_napkin_formula_rejects_undeclared_free_z() -> None:
    falsifier = HPFalsifier(
        graph=NAPKIN_GRAPH,
        treatments="X",
        outcomes="Y",
    )

    with pytest.raises(
        ValueError,
        match="free inputs other than the treatments",
    ):
        falsifier.check(NAPKIN_CONDITIONAL_FORMULA)


def test_napkin_formula_accepts_z_declared_redundant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "hiprof.verification.falsifier.getrandbits",
        fixed_getrandbits,
    )
    falsifier = HPFalsifier(
        graph=NAPKIN_GRAPH,
        treatments="X",
        outcomes="Y",
    )

    result = falsifier.check(
        NAPKIN_CONDITIONAL_FORMULA,
        redundant_inputs="Z",
    )

    assert result.accepted


def test_check_rejects_observational_conditional_on_napkin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "hiprof.verification.falsifier.getrandbits",
        fixed_getrandbits,
    )
    falsifier = HPFalsifier(
        graph=NAPKIN_GRAPH,
        treatments="X",
        outcomes="Y",
    )

    assert not falsifier.check("p(Y | X)").accepted


def test_check_rejects_nonredundant_declared_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "hiprof.verification.falsifier.getrandbits",
        fixed_getrandbits,
    )
    falsifier = HPFalsifier(
        graph="X -> Y; Z -> Y",
        treatments="X",
        outcomes="Y",
    )

    result = falsifier.check(
        "p(Y | X, Z)",
        redundant_inputs="Z",
    )

    assert not result.accepted


def test_check_rejects_unused_redundant_input_declaration() -> None:
    falsifier = HPFalsifier(
        graph="Z -> X; X -> Y",
        treatments="X",
        outcomes="Y",
    )

    with pytest.raises(ValueError, match="not free inputs"):
        falsifier.check(
            "p(Y | X)",
            redundant_inputs="Z",
        )


@pytest.mark.parametrize("redundant_input", ["X", "Y"])
def test_check_rejects_query_variables_as_redundant(
    redundant_input: str,
) -> None:
    falsifier = HPFalsifier(
        graph="X -> Y",
        treatments="X",
        outcomes="Y",
    )

    with pytest.raises(
        ValueError,
        match="distinct from treatments and outcomes",
    ):
        falsifier.check(
            "p(Y | X)",
            redundant_inputs=redundant_input,
        )


def test_check_rejects_redundant_inputs_for_none_formula() -> None:
    falsifier = HPFalsifier(
        graph="Z -> X; X -> Y",
        treatments="X",
        outcomes="Y",
    )

    with pytest.raises(
        ValueError,
        match="cannot be used when formula is None",
    ):
        falsifier.check(
            None,
            redundant_inputs="Z",
        )


def test_check_rejects_non_string_formula() -> None:
    falsifier = HPFalsifier("X -> Y", treatments="X", outcomes="Y")

    with pytest.raises(TypeError, match="string or None"):
        falsifier.check(1)  # type: ignore[arg-type]
