from __future__ import annotations

import pytest
from flint import fmpq_mat

from hiprof.formula.formula import Formula, Variable
from hiprof.formula.validation import parse_and_validate
from hiprof.verification.gaussian import (
    GaussianDistribution,
    GaussianEvaluator,
)
from hiprof.verification.utils import submatrix

from .helpers import matrix_entries, matrix_shape, q


def two_variable_joint() -> GaussianDistribution:
    return GaussianDistribution(
        variables=(Variable("X"), Variable("Y")),
        mean=fmpq_mat(2, 1, [1, 2]),
        covariance=fmpq_mat(2, 2, [4, 2, 2, 9]),
    )


def test_base_kernel_without_inputs_selects_marginal_exactly() -> None:
    kernel = GaussianEvaluator(two_variable_joint()).evaluate(
        parse_and_validate("p(Y)")
    )

    assert kernel.outputs == (Variable("Y"),)
    assert kernel.inputs == ()
    assert matrix_entries(kernel.mean_intercept) == ((q(2),),)
    assert matrix_shape(kernel.mean_linear) == (1, 0)
    assert matrix_entries(kernel.covariance) == ((q(9),),)


def test_base_kernel_with_inputs_conditions_exactly() -> None:
    kernel = GaussianEvaluator(two_variable_joint()).evaluate(
        parse_and_validate("p(Y | X)")
    )

    assert kernel.outputs == (Variable("Y"),)
    assert kernel.inputs == (Variable("X"),)
    assert matrix_entries(kernel.mean_intercept) == ((q(3, 2),),)
    assert matrix_entries(kernel.mean_linear) == ((q(1, 2),),)
    assert matrix_entries(kernel.covariance) == ((q(8),),)


def test_base_kernel_rejects_variable_out_of_joint_scope() -> None:
    evaluator = GaussianEvaluator(two_variable_joint())

    try:
        evaluator.evaluate(parse_and_validate("p(Z)"))
    except KeyError as error:
        assert "not in scope" in str(error)
    else:
        raise AssertionError("Expected KeyError")


def test_evaluator_rejects_unknown_formula_node() -> None:
    evaluator = GaussianEvaluator(two_variable_joint())

    with pytest.raises(TypeError, match="Formula"):
        evaluator._evaluate(Formula())


def test_submatrix_handles_zero_rows_and_columns() -> None:
    matrix = fmpq_mat(2, 2, [1, 2, 3, 4])

    assert matrix_shape(submatrix(matrix, (), ())) == (0, 0)
    assert matrix_shape(submatrix(matrix, (), (0, 1))) == (0, 2)
    assert matrix_shape(submatrix(matrix, (0, 1), ())) == (2, 0)
