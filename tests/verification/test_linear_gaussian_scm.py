from __future__ import annotations

from flint import fmpq_mat

from hiprof.formula.formula import Variable
from hiprof.verification.falsifier import _LinearGaussianSCM
from hiprof.verification.gaussian import GaussianDistribution, GaussianKernel


def three_variable_scm() -> _LinearGaussianSCM:
    # X = 1 + eps_X
    # M = 7 + 2 X + eps_M
    # Y = 11 + 3 X + 5 M + eps_Y
    return _LinearGaussianSCM(
        variables=("X", "M", "Y"),
        coefficients=fmpq_mat(
            3,
            3,
            [
                0,
                0,
                0,
                2,
                0,
                0,
                3,
                5,
                0,
            ],
        ),
        intercepts=fmpq_mat(3, 1, [1, 7, 11]),
        noise_covariance=fmpq_mat(
            3,
            3,
            [
                13,
                0,
                0,
                0,
                17,
                0,
                0,
                0,
                19,
            ],
        ),
    )


def test_joint_distribution_solves_linear_scm_exactly() -> None:
    joint = three_variable_scm().joint_distribution()

    assert joint == GaussianDistribution(
        variables=(Variable("X"), Variable("M"), Variable("Y")),
        mean=fmpq_mat(3, 1, [1, 9, 59]),
        covariance=fmpq_mat(
            3,
            3,
            [
                13,
                26,
                169,
                26,
                69,
                423,
                169,
                423,
                2641,
            ],
        ),
    )


def test_interventional_kernel_includes_direct_and_mediated_effects() -> None:
    kernel = three_variable_scm().interventional_kernel(
        treatments=("X",),
        outcomes=("M", "Y"),
    )

    assert kernel == GaussianKernel(
        outputs=(Variable("M"), Variable("Y")),
        inputs=(Variable("X"),),
        mean_intercept=fmpq_mat(2, 1, [7, 46]),
        mean_linear=fmpq_mat(2, 1, [2, 13]),
        covariance=fmpq_mat(2, 2, [17, 85, 85, 444]),
    )
