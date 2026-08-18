from __future__ import annotations

from dataclasses import replace

import pytest
from flint import fmpq_mat

from hiprof.formula.formula import KernelSignature, Variable
from hiprof.verification.gaussian import GaussianKernel, _hstack, _vstack


def test_align_reorders_kernel() -> None:
    x = Variable("X")
    w = Variable("W")
    y = Variable("Y")
    z = Variable("Z")
    kernel = GaussianKernel(
        outputs=(y, z),
        inputs=(x, w),
        mean_intercept=fmpq_mat(2, 1, [1, 2]),
        mean_linear=fmpq_mat(2, 2, [3, 4, 5, 6]),
        covariance=fmpq_mat(2, 2, [7, 8, 8, 9]),
    )

    aligned = kernel.align(
        outputs=(z, y),
        inputs=(w, x),
    )

    assert aligned == GaussianKernel(
        outputs=(z, y),
        inputs=(w, x),
        mean_intercept=fmpq_mat(2, 1, [2, 1]),
        mean_linear=fmpq_mat(2, 2, [6, 5, 4, 3]),
        covariance=fmpq_mat(2, 2, [9, 8, 8, 7]),
    )


def test_align_inserts_zero_columns_for_new_inputs() -> None:
    x = Variable("X")
    z = Variable("Z")
    y = Variable("Y")
    kernel = GaussianKernel(
        outputs=(y,),
        inputs=(x,),
        mean_intercept=fmpq_mat(1, 1, [1]),
        mean_linear=fmpq_mat(1, 1, [3]),
        covariance=fmpq_mat(1, 1, [2]),
    )

    aligned = kernel.align(outputs=(y,), inputs=(z, x))

    assert aligned.mean_linear == fmpq_mat(1, 2, [0, 3])


def test_kernel_multiplication_retains_external_inputs_once() -> None:
    a = Variable("A")
    b = Variable("B")
    x = Variable("X")
    z = Variable("Z")
    left = GaussianKernel(
        outputs=(a,),
        inputs=(x,),
        mean_intercept=fmpq_mat(1, 1, [1]),
        mean_linear=fmpq_mat(1, 1, [2]),
        covariance=fmpq_mat(1, 1, [3]),
    )
    right = GaussianKernel(
        outputs=(b,),
        inputs=(a, z, x),
        mean_intercept=fmpq_mat(1, 1, [4]),
        mean_linear=fmpq_mat(1, 3, [5, 6, 7]),
        covariance=fmpq_mat(1, 1, [7]),
    )

    assert left * right == GaussianKernel(
        outputs=(a, b),
        inputs=(x, z),
        mean_intercept=fmpq_mat(2, 1, [1, 9]),
        mean_linear=fmpq_mat(2, 2, [2, 0, 17, 6]),
        covariance=fmpq_mat(2, 2, [3, 15, 15, 82]),
    )


def test_kernel_noop_operations_return_same_instance() -> None:
    y = Variable("Y")
    x = Variable("X")
    kernel = GaussianKernel(
        outputs=(y,),
        inputs=(x,),
        mean_intercept=fmpq_mat(1, 1, [1]),
        mean_linear=fmpq_mat(1, 1, [2]),
        covariance=fmpq_mat(1, 1, [3]),
    )

    assert kernel.select_outputs((y,)) is kernel
    assert kernel.condition_on(()) is kernel
    assert kernel.align(outputs=(y,), inputs=(x,)) is kernel


def test_kernel_signature_and_equality_include_all_parameters() -> None:
    y = Variable("Y")
    x = Variable("X")
    kernel = GaussianKernel(
        outputs=(y,),
        inputs=(x,),
        mean_intercept=fmpq_mat(1, 1, [1]),
        mean_linear=fmpq_mat(1, 1, [2]),
        covariance=fmpq_mat(1, 1, [3]),
    )
    equal_kernel = GaussianKernel(
        outputs=(y,),
        inputs=(x,),
        mean_intercept=fmpq_mat(1, 1, [1]),
        mean_linear=fmpq_mat(1, 1, [2]),
        covariance=fmpq_mat(1, 1, [3]),
    )

    assert kernel.signature == KernelSignature(
        outputs=frozenset({y}),
        inputs=frozenset({x}),
    )
    assert kernel == equal_kernel
    assert kernel != replace(kernel, covariance=fmpq_mat(1, 1, [4]))


def test_matrix_stacking_rejects_incompatible_shapes() -> None:
    with pytest.raises(ValueError, match="same number of rows"):
        _hstack(fmpq_mat(1, 1), fmpq_mat(2, 1))

    with pytest.raises(ValueError, match="same number of columns"):
        _vstack(fmpq_mat(1, 1), fmpq_mat(1, 2))
