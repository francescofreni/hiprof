from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from ..formula.formula import (
    BaseKernel,
    Formula,
    InternalConditionalDivision,
    KernelSignature,
    Marginalisation,
    Product,
)
from ..formula.validation import ValidationResult


class DegreeBound(NamedTuple):
    mean_numerator: int
    mean_denominator: int
    covariance_numerator: int
    covariance_denominator: int

    def __str__(self) -> str:
        return (
            "DegreeBound("
            f"mean=(num={self.mean_numerator}, "
            f"den={self.mean_denominator}), "
            f"covariance=(num={self.covariance_numerator}, "
            f"den={self.covariance_denominator})"
            ")"
        )


@dataclass(frozen=True)
class _Evaluation:
    signature: KernelSignature
    degree_bound: DegreeBound


class DegreeBoundEvaluator:
    def __init__(self, number_of_observed_variables: int) -> None:
        if number_of_observed_variables < 1:
            raise ValueError("number_of_observed_variables must be positive")
        self.n = number_of_observed_variables

    def evaluate(self, validated: ValidationResult) -> DegreeBound:
        return self._evaluate(validated.formula).degree_bound

    def _evaluate(self, formula: Formula) -> _Evaluation:
        if isinstance(formula, BaseKernel):
            signature = KernelSignature(
                frozenset(formula.outputs),
                frozenset(formula.inputs),
            )
            return _Evaluation(signature, self.base_degree_bound(formula))

        if isinstance(formula, Product):
            iterator = iter(formula.factors)
            result = self._evaluate(next(iterator))
            for factor in iterator:
                right = self._evaluate(factor)
                outputs = result.signature.outputs | right.signature.outputs
                inputs = (
                    result.signature.inputs | right.signature.inputs
                ) - outputs
                result = _Evaluation(
                    KernelSignature(outputs, inputs),
                    self.product_degree_bound(
                        result.degree_bound,
                        right.degree_bound,
                    ),
                )
            return result

        if isinstance(formula, Marginalisation):
            body = self._evaluate(formula.body)
            variables = frozenset(formula.variables)
            return _Evaluation(
                KernelSignature(
                    body.signature.outputs - variables,
                    body.signature.inputs,
                ),
                body.degree_bound,
            )

        if isinstance(formula, InternalConditionalDivision):
            numerator = self._evaluate(formula.body)
            m_variables = (
                set(formula.denominator_inputs) - numerator.signature.inputs
            )
            conditioning_size = len(m_variables) + len(
                formula.denominator_outputs
            )
            denominator_bound = self.conditional_degree_bound(
                numerator.degree_bound,
                conditioning_size,
            )
            denominator_outputs = frozenset(formula.denominator_outputs)
            return _Evaluation(
                KernelSignature(
                    numerator.signature.outputs - denominator_outputs,
                    numerator.signature.inputs | denominator_outputs,
                ),
                self.product_degree_bound(
                    numerator.degree_bound,
                    denominator_bound,
                ),
            )

        raise TypeError(f"Unknown formula node: {type(formula).__name__}")

    def base_degree_bound(self, kernel: BaseKernel) -> DegreeBound:
        n = self.n
        conditioning_size = len(kernel.inputs)
        if conditioning_size == 0:
            return DegreeBound(n, 0, 2 * n - 1, 0)

        covariance_degree = 2 * n - 1
        return DegreeBound(
            n + conditioning_size * covariance_degree,
            conditioning_size * covariance_degree,
            (conditioning_size + 1) * covariance_degree,
            conditioning_size * covariance_degree,
        )

    @staticmethod
    def conditional_degree_bound(
        bound: DegreeBound,
        conditioning_size: int,
    ) -> DegreeBound:
        m_n, m_d, s_n, s_d = bound
        b = conditioning_size
        return DegreeBound(
            b * s_n + max(m_n, m_d),
            m_d + b * s_n,
            (b + 1) * s_n,
            s_d + b * s_n,
        )

    @staticmethod
    def product_degree_bound(
        left: DegreeBound,
        right: DegreeBound,
    ) -> DegreeBound:
        m_e_n, m_e_d, s_e_n, s_e_d = left
        m_f_n, m_f_d, s_f_n, s_f_d = right
        return DegreeBound(
            max(
                m_e_n + m_f_d,
                m_f_n + max(m_e_n, m_e_d),
            ),
            m_e_d + m_f_d,
            max(
                s_e_n + s_f_d + 2 * m_f_d,
                s_e_n + m_f_n + s_f_d + m_f_d,
                s_e_n + 2 * m_f_n + s_f_d,
                s_f_n + s_e_d + 2 * m_f_d,
            ),
            s_e_d + s_f_d + 2 * m_f_d,
        )
