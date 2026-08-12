from __future__ import annotations

from flint import fmpq_mat

from hiprof.formula.formula import Variable
from hiprof.formula.validation import parse_and_validate
from hiprof.verification.gaussian import (
    GaussianDistribution,
    GaussianEvaluator,
)

from .helpers import matrix_entries, matrix_shape, q


def test_marginalisation_selects_remaining_outputs_in_original_order() -> None:
    joint = GaussianDistribution(
        variables=(Variable("X"), Variable("Y"), Variable("Z")),
        mean=fmpq_mat(3, 1, [1, 2, 3]),
        covariance=fmpq_mat(3, 3, [4, 1, 2, 1, 5, 3, 2, 3, 6]),
    )

    kernel = GaussianEvaluator(joint).evaluate(
        parse_and_validate("sum_{Y} { p(X, Y, Z) }")
    )

    assert kernel.outputs == (Variable("X"), Variable("Z"))
    assert matrix_entries(kernel.mean_intercept) == ((q(1),), (q(3),))
    assert matrix_entries(kernel.covariance) == (
        (q(4), q(2)),
        (q(2), q(6)),
    )


def test_marginalisation_can_leave_empty_output_set() -> None:
    joint = GaussianDistribution(
        variables=(Variable("X"),),
        mean=fmpq_mat(1, 1, [5]),
        covariance=fmpq_mat(1, 1, [7]),
    )

    kernel = GaussianEvaluator(joint).evaluate(
        parse_and_validate("sum_{X} { p(X) }")
    )

    assert kernel.outputs == ()
    assert kernel.inputs == ()
    assert matrix_shape(kernel.mean_intercept) == (0, 1)
    assert matrix_shape(kernel.mean_linear) == (0, 0)
    assert matrix_shape(kernel.covariance) == (0, 0)
