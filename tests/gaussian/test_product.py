from __future__ import annotations

from flint import fmpq_mat

from hiprof.formula.formula import Variable
from hiprof.formula.validation import parse_and_validate
from hiprof.verification.gaussian import (
    GaussianDistribution,
    GaussianEvaluator,
)

from .helpers import matrix_entries, q


def joint() -> GaussianDistribution:
    return GaussianDistribution(
        variables=(Variable("X"), Variable("Y")),
        mean=fmpq_mat(2, 1, [1, 2]),
        covariance=fmpq_mat(2, 2, [4, 2, 2, 9]),
    )


def test_product_reconstructs_joint_from_marginal_and_conditional() -> None:
    kernel = GaussianEvaluator(joint()).evaluate(
        parse_and_validate("p(Y | X) p(X)")
    )

    assert kernel.outputs == (Variable("X"), Variable("Y"))
    assert kernel.inputs == ()
    assert matrix_entries(kernel.mean_intercept) == ((q(1),), (q(2),))
    assert matrix_entries(kernel.mean_linear) == ((), ())
    assert matrix_entries(kernel.covariance) == (
        (q(4), q(2)),
        (q(2), q(9)),
    )
