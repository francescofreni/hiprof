from __future__ import annotations

import pytest

from hiprof.formula.formula import BaseKernel, Formula, Product, Variable
from hiprof.formula.validation import (
    ValidationError,
    parse_and_validate,
    validate,
)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("p(X, X')", "multiple versions"),
        ("p(Y) p(Y | X)", "repeat outputs"),
        ("sum_{Z} { p(Y | Z) }", "Cannot marginalise non-outputs"),
        ("sum_{Y, Y} { p(Y, Z) }", "Repeated variable"),
        ("icd_{Z | X} { p(Y | X) }", "must be body outputs"),
        ("icd_{Y | Z} { p(Y | X) }", "retain every body input"),
        ("icd_{Y | Y} { p(Y) }", "overlap"),
        ("icd_{Y | Z} { p(Y, X) }", "invalid"),
    ],
)
def test_validation_rejects_invalid_formulae(
    source: str,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        parse_and_validate(source)


def test_validation_orders_product_sequentially() -> None:
    result = parse_and_validate("p(Y | X) p(X)")

    assert isinstance(result.formula, Product)
    assert result.formula.factors == (
        BaseKernel(outputs=(Variable("X"),)),
        BaseKernel(outputs=(Variable("Y"),), inputs=(Variable("X"),)),
    )


def test_product_with_cyclic_dependencies_is_rejected() -> None:
    with pytest.raises(ValidationError, match="no valid sequential ordering"):
        parse_and_validate("p(X | Y) p(Y | X)")


def test_validate_rejects_unknown_node_type() -> None:
    class Unknown(Formula):
        pass

    with pytest.raises(TypeError, match="Unknown formula node"):
        validate(Unknown())
