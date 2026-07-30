from __future__ import annotations

from flint import fmpq_mat

from hiprof.formula.formula import Variable
from hiprof.formula.validation import parse_and_validate
from hiprof.verification.gaussian import (
    GaussianDistribution,
    GaussianEvaluator,
)

from .helpers import matrix_entries, matrix_shape, q


def three_variable_joint() -> GaussianDistribution:
    return GaussianDistribution(
        variables=(Variable("X"), Variable("Y"), Variable("Z")),
        mean=fmpq_mat(3, 1, [1, 2, 3]),
        covariance=fmpq_mat(3, 3, [4, 2, 1, 2, 9, 3, 1, 3, 16]),
    )


def test_icd_with_no_remaining_outputs_returns_empty_output_kernel() -> None:
    kernel = GaussianEvaluator(three_variable_joint()).evaluate(
        parse_and_validate("icd_{Y | X} { p(Y | X) }")
    )

    assert kernel.outputs == ()
    assert kernel.inputs == (Variable("X"), Variable("Y"))
    assert matrix_shape(kernel.mean_intercept) == (0, 1)
    assert matrix_shape(kernel.mean_linear) == (0, 2)
    assert matrix_shape(kernel.covariance) == (0, 0)


def test_icd_of_p_yz_given_x_returns_p_y_given_xz() -> None:
    kernel = GaussianEvaluator(three_variable_joint()).evaluate(
        parse_and_validate("icd_{Z | X} { p(Y, Z | X) }")
    )

    assert kernel.outputs == (Variable("Y"),)
    assert kernel.inputs == (Variable("X"), Variable("Z"))
    assert matrix_entries(kernel.mean_intercept) == ((q(67, 63),),)
    assert matrix_entries(kernel.mean_linear) == ((q(29, 63), q(10, 63)),)
    assert matrix_entries(kernel.covariance) == ((q(479, 63),),)


def test_icd_retains_body_inputs_and_adds_denominator_outputs() -> None:
    kernel = GaussianEvaluator(three_variable_joint()).evaluate(
        parse_and_validate("icd_{X | Z} { p(Y, X | Z) }")
    )

    assert kernel.outputs == (Variable("Y"),)
    assert kernel.inputs == (Variable("Z"), Variable("X"))
