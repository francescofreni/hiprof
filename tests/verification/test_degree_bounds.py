from __future__ import annotations

import pytest

from hiprof.formula.validation import parse_and_validate
from hiprof.verification.degree import DegreeBound, DegreeBoundEvaluator


def test_degree_bound_string_is_readable() -> None:
    assert str(DegreeBound(1, 2, 3, 4)) == (
        "DegreeBound(mean=(num=1, den=2), " "covariance=(num=3, den=4))"
    )


def test_base_degree_bound_without_conditioning() -> None:
    bound = DegreeBoundEvaluator(3).evaluate(parse_and_validate("p(Y)"))

    assert bound == DegreeBound(
        mean_numerator=3,
        mean_denominator=0,
        covariance_numerator=5,
        covariance_denominator=0,
    )


def test_base_degree_bound_with_conditioning() -> None:
    bound = DegreeBoundEvaluator(3).evaluate(parse_and_validate("p(Y | X, Z)"))

    assert bound == DegreeBound(
        mean_numerator=13,
        mean_denominator=10,
        covariance_numerator=15,
        covariance_denominator=10,
    )


def test_product_and_icd_degree_bounds_are_composed() -> None:
    evaluator = DegreeBoundEvaluator(3)
    conditional_bound = evaluator.conditional_degree_bound(
        DegreeBound(8, 5, 10, 5),
        1,
    )

    assert evaluator.evaluate(parse_and_validate("p(X) p(Y | X)")) == (
        evaluator.product_degree_bound(
            DegreeBound(3, 0, 5, 0),
            DegreeBound(8, 5, 10, 5),
        )
    )
    assert evaluator.evaluate(
        parse_and_validate("icd_{X | Z} { p(Y, X | Z) }")
    ) == (
        evaluator.product_degree_bound(
            DegreeBound(8, 5, 10, 5),
            conditional_bound,
        )
    )


def test_degree_evaluator_requires_positive_variable_count() -> None:
    with pytest.raises(ValueError, match="positive"):
        DegreeBoundEvaluator(number_of_variables=0)
