from __future__ import annotations

from dataclasses import dataclass

from flint import fmpq_mat

from ..formula.formula import (
    BaseKernel,
    Formula,
    InternalConditionalDivision,
    KernelSignature,
    Marginalisation,
    Product,
    Variable,
)
from ..formula.validation import ValidationResult
from .utils import align_columns, submatrix


@dataclass(frozen=True)
class GaussianDistribution:
    variables: tuple[Variable, ...]
    mean: fmpq_mat  # (n, 1)
    covariance: fmpq_mat  # (n, n)


@dataclass(frozen=True)
class GaussianKernel:
    outputs: tuple[Variable, ...]
    inputs: tuple[Variable, ...]
    mean_intercept: fmpq_mat  # (n_outputs, 1)
    mean_linear: fmpq_mat  # (n_outputs, n_inputs)
    covariance: fmpq_mat  # (n_outputs, n_outputs)

    @property
    def signature(self) -> KernelSignature:
        return KernelSignature(
            outputs=frozenset(self.outputs),
            inputs=frozenset(self.inputs),
        )


class GaussianEvaluator:
    def __init__(self, joint: GaussianDistribution) -> None:
        self.joint = joint
        self._joint_indices = {
            variable.original: index
            for index, variable in enumerate(joint.variables)
        }

    def evaluate(self, validated: ValidationResult) -> GaussianKernel:
        return self._evaluate(validated.formula)

    def _evaluate(self, formula: Formula) -> GaussianKernel:
        if isinstance(formula, BaseKernel):
            return self.base_kernel(formula)

        if isinstance(formula, Marginalisation):
            return self.marginalisation(formula)

        if isinstance(formula, Product):
            return self.product(formula)

        if isinstance(formula, InternalConditionalDivision):
            return self.icd(formula)

        raise TypeError(f"Unknown formula node: {type(formula).__name__}")

    def base_kernel(self, kernel: BaseKernel) -> GaussianKernel:
        outputs = kernel.outputs
        inputs = kernel.inputs

        try:
            output_indices = [
                self._joint_indices[variable.original] for variable in outputs
            ]
            input_indices = [
                self._joint_indices[variable.original] for variable in inputs
            ]
        except KeyError as error:
            raise KeyError(
                f"Variable {error.args[0]} is not in scope."
            ) from error

        mu_y = submatrix(self.joint.mean, output_indices, (0,))
        sigma_yy = submatrix(
            self.joint.covariance,
            output_indices,
            output_indices,
        )

        if not inputs:
            return GaussianKernel(
                outputs=outputs,
                inputs=(),
                mean_intercept=mu_y,
                mean_linear=fmpq_mat(len(outputs), 0),
                covariance=sigma_yy,
            )

        mu_x = submatrix(self.joint.mean, input_indices, (0,))
        sigma_xy = submatrix(
            self.joint.covariance,
            input_indices,
            output_indices,
        )
        sigma_xx = submatrix(
            self.joint.covariance,
            input_indices,
            input_indices,
        )

        mean_linear = sigma_xx.solve(sigma_xy).transpose()

        return GaussianKernel(
            outputs=outputs,
            inputs=inputs,
            mean_intercept=mu_y - mean_linear * mu_x,
            mean_linear=mean_linear,
            covariance=sigma_yy - mean_linear * sigma_xy,
        )

    def marginalisation(
        self,
        marginalisation: Marginalisation,
    ) -> GaussianKernel:
        body = self._evaluate(marginalisation.body)
        remaining_outputs = tuple(
            variable
            for variable in body.outputs
            if variable not in marginalisation.variables
        )
        return _select_outputs(body, remaining_outputs)

    def product(self, product: Product) -> GaussianKernel:
        iterator = iter(product.factors)
        result = self._evaluate(next(iterator))

        for factor in iterator:
            result = self._product_two(
                result,
                self._evaluate(factor),
            )

        return result

    def icd(self, icd: InternalConditionalDivision) -> GaussianKernel:
        body = self._evaluate(icd.body)

        remaining_outputs = tuple(
            v for v in body.outputs if v not in icd.denominator_outputs
        )
        m_outputs = tuple(
            v for v in remaining_outputs if v in icd.denominator_inputs
        )
        u_kernel = self._conditioning(
            body,
            m_outputs + icd.denominator_outputs,
        )

        if not m_outputs:
            return u_kernel

        m_kernel = _select_outputs(body, m_outputs)
        result = self._product_two(m_kernel, u_kernel)
        result = _select_outputs(result, remaining_outputs)
        return result

    @staticmethod
    def _conditioning(
        kernel: GaussianKernel,
        conditioned_outputs: tuple[Variable, ...],
    ) -> GaussianKernel:
        if not conditioned_outputs:
            return kernel

        kept_outputs = tuple(
            variable
            for variable in kernel.outputs
            if variable not in conditioned_outputs
        )
        output_index = {
            variable: index for index, variable in enumerate(kernel.outputs)
        }
        kept_indices = tuple(
            output_index[variable] for variable in kept_outputs
        )
        conditioned_indices = tuple(
            output_index[variable] for variable in conditioned_outputs
        )
        input_indices = range(len(kernel.inputs))

        a_y = submatrix(
            kernel.mean_intercept,
            kept_indices,
            (0,),
        )
        a_d = submatrix(
            kernel.mean_intercept,
            conditioned_indices,
            (0,),
        )
        m_y = submatrix(
            kernel.mean_linear,
            kept_indices,
            input_indices,
        )
        m_d = submatrix(
            kernel.mean_linear,
            conditioned_indices,
            input_indices,
        )
        s_yy = submatrix(
            kernel.covariance,
            kept_indices,
            kept_indices,
        )
        s_dy = submatrix(
            kernel.covariance,
            conditioned_indices,
            kept_indices,
        )
        s_dd = submatrix(
            kernel.covariance,
            conditioned_indices,
            conditioned_indices,
        )

        regression = s_dd.solve(s_dy).transpose()

        return GaussianKernel(
            outputs=kept_outputs,
            inputs=kernel.inputs + conditioned_outputs,
            mean_intercept=a_y - regression * a_d,
            mean_linear=_hstack(
                m_y - regression * m_d,
                regression,
            ),
            covariance=s_yy - regression * s_dy,
        )

    @staticmethod
    def _product_two(
        left: GaussianKernel,
        right: GaussianKernel,
    ) -> GaussianKernel:
        right_external_inputs = tuple(
            v for v in right.inputs if v not in left.outputs
        )
        inputs = _ordered_union(
            left.inputs,
            right_external_inputs,
        )

        left_linear = align_columns(
            left.mean_linear,
            left.inputs,
            inputs,
        )
        left_output_linear = align_columns(
            right.mean_linear,
            right.inputs,
            left.outputs,
        )
        external_linear = align_columns(
            right.mean_linear,
            right.inputs,
            inputs,
        )

        right_mean_intercept = (
            right.mean_intercept + left_output_linear * left.mean_intercept
        )
        right_mean_linear = external_linear + left_output_linear * left_linear

        cross_covariance = left.covariance * left_output_linear.transpose()
        right_covariance = (
            right.covariance
            + left_output_linear
            * left.covariance
            * left_output_linear.transpose()
        )

        return GaussianKernel(
            outputs=left.outputs + right.outputs,
            inputs=inputs,
            mean_intercept=_vstack(
                left.mean_intercept,
                right_mean_intercept,
            ),
            mean_linear=_vstack(
                left_linear,
                right_mean_linear,
            ),
            covariance=_vstack(
                _hstack(
                    left.covariance,
                    cross_covariance,
                ),
                _hstack(
                    cross_covariance.transpose(),
                    right_covariance,
                ),
            ),
        )


def _select_outputs(
    kernel: GaussianKernel,
    outputs: tuple[Variable, ...],
) -> GaussianKernel:
    if outputs == kernel.outputs:
        return kernel

    output_index = {
        variable: index for index, variable in enumerate(kernel.outputs)
    }
    indices = tuple(output_index[variable] for variable in outputs)

    return GaussianKernel(
        outputs=outputs,
        inputs=kernel.inputs,
        mean_intercept=submatrix(
            kernel.mean_intercept,
            indices,
            (0,),
        ),
        mean_linear=submatrix(
            kernel.mean_linear,
            indices,
            range(len(kernel.inputs)),
        ),
        covariance=submatrix(
            kernel.covariance,
            indices,
            indices,
        ),
    )


def _hstack(
    first: fmpq_mat,
    *rest: fmpq_mat,
) -> fmpq_mat:
    matrices = (first,) + rest
    nrows = first.nrows()

    if any(matrix.nrows() != nrows for matrix in rest):
        raise ValueError("Matrices must have the same number of rows.")

    return fmpq_mat(
        nrows,
        sum(matrix.ncols() for matrix in matrices),
        [
            matrix[i, j]
            for i in range(nrows)
            for matrix in matrices
            for j in range(matrix.ncols())
        ],
    )


def _ordered_union(
    first: tuple[Variable, ...],
    second: tuple[Variable, ...],
) -> tuple[Variable, ...]:
    return tuple(dict.fromkeys(first + second))


def _vstack(
    first: fmpq_mat,
    *rest: fmpq_mat,
) -> fmpq_mat:
    matrices = (first,) + rest
    ncols = first.ncols()

    if any(matrix.ncols() != ncols for matrix in rest):
        raise ValueError("Matrices must have the same number of columns.")

    return fmpq_mat(
        sum(matrix.nrows() for matrix in matrices),
        ncols,
        [
            matrix[i, j]
            for matrix in matrices
            for i in range(matrix.nrows())
            for j in range(ncols)
        ],
    )
