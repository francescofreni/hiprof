from __future__ import annotations

import pytest
from lark import UnexpectedInput

from hiprof.formula.formula import (
    BaseKernel,
    BaseQuotient,
    InternalConditionalDivision,
    Marginalisation,
    Product,
    Variable,
    format_ast,
)
from hiprof.formula.parser import parse


def test_variable_copy_suffixes_round_trip() -> None:
    assert Variable.from_token("X") == Variable("X")
    assert Variable.from_token("X'") == Variable("X", 0)
    assert Variable.from_token("X'2") == Variable("X", 2)
    assert str(Variable("X", 0)) == "X'"
    assert str(Variable("X", 2)) == "X'2"


def test_parse_base_kernel_with_inputs() -> None:
    formula = parse("p(Y, X | Z, W)")

    assert formula == BaseKernel(
        outputs=(Variable("Y"), Variable("X")),
        inputs=(Variable("Z"), Variable("W")),
    )


def test_parse_product_marginalisation_and_icd() -> None:
    formula = parse("icd_{X | Z} { sum_{W} { p(Y, X | Z, W) p(W) } }")

    assert isinstance(formula, InternalConditionalDivision)
    assert formula.denominator_outputs == (Variable("X"),)
    assert formula.denominator_inputs == (Variable("Z"),)
    assert isinstance(formula.body, Marginalisation)
    assert formula.body.variables == (Variable("W"),)
    assert isinstance(formula.body.body, Product)


def test_parse_standalone_base_quotient() -> None:
    formula = parse("p(A, B | C) / p(B | C)")

    assert formula == BaseQuotient(
        numerator=BaseKernel(
            outputs=(Variable("A"), Variable("B")),
            inputs=(Variable("C"),),
        ),
        denominator=BaseKernel(
            outputs=(Variable("B"),),
            inputs=(Variable("C"),),
        ),
    )


@pytest.mark.parametrize(
    "source",
    [
        "",
        "p()",
        "p(x)",
        "sum_{} { p(X) }",
        "p(X) / p(Y) p(Z)",
        "p(X | )",
        "p(X01)",
    ],
)
def test_parse_rejects_invalid_syntax(source: str) -> None:
    with pytest.raises(UnexpectedInput):
        parse(source)


def test_format_ast_is_stable_for_public_debug_output() -> None:
    assert "BaseKernel" in format_ast(parse("p(Y | X)"))
