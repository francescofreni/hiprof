from __future__ import annotations

import warnings
from dataclasses import dataclass
from fractions import Fraction
from random import getrandbits
from typing import Sequence

from flint import fmpq_mat, fmpz

from hiprof.base.graph import Graph, parse_graph
from hiprof.formula.formula import Variable
from hiprof.formula.validation import ValidationResult, parse_and_validate
from hiprof.utils import format_variables

from .degree import DegreeBound, DegreeBoundEvaluator
from .gaussian import GaussianDistribution, GaussianEvaluator, GaussianKernel
from .utils import align_columns, submatrix


# Suppress some ananke-related warnings that are safe to ignore.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="google.api_core",
)
warnings.filterwarnings(
    "ignore",
    message=".*IProgress not found.*",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"pgmpy\..*",
)

from ananke import graphs, identification


_DEFAULT_ENTROPY_BITS = 64
_DEFAULT_TARGET_BOUND = Fraction(1, 10**14)


@dataclass(frozen=True)
class CheckResult:
    accepted: bool
    false_acceptance_bound: Fraction | None = None
    degree: int | None = None
    entropy_bits: int | None = None
    repetitions: int = 0

    def __bool__(self) -> bool:
        return self.accepted

    def __str__(self) -> str:
        if not self.accepted:
            return "False"

        # formula=None case.
        if self.degree is None:
            return "True"

        return (
            "True\n"
            "False-acceptance bound: "
            f"{float(self.false_acceptance_bound):.3e}"
        )

    def __repr__(self) -> str:
        return str(self)


@dataclass(frozen=True)
class _LinearGaussianSCM:
    variables: tuple[str, ...]
    coefficients: fmpq_mat  # (child, parent)
    intercepts: fmpq_mat  # (n, 1)
    noise_covariance: fmpq_mat  # diagonal (n, n)


class HPFalsifier:
    def __init__(
        self,
        graph: str,
        treatments: str | Sequence[str],
        outcomes: str | Sequence[str],
    ) -> None:
        self.graph = parse_graph(graph)

        self.treatments = _validate_variables(
            treatments,
            name="Treatments",
            graph=self.graph,
        )
        self.outcomes = _validate_variables(
            outcomes,
            name="Outcomes",
            graph=self.graph,
        )

        overlap = sorted(set(self.treatments) & set(self.outcomes))
        if overlap:
            raise ValueError(
                "Treatments and outcomes must be disjoint. "
                f"Overlapping variables: {', '.join(overlap)}."
            )

    def check(
        self,
        formula: str | None,
        target_bound: Fraction | float = _DEFAULT_TARGET_BOUND,
    ) -> CheckResult:
        target = _validate_target_bound(target_bound)

        if formula is None:
            return CheckResult(
                accepted=not self._is_identifiable(),
            )

        if not isinstance(formula, str):
            raise TypeError("formula must be a string or None.")

        if not self._is_identifiable():
            return CheckResult(accepted=False)

        validated = parse_and_validate(formula)
        self._validate_formula_variables(validated)
        self._validate_formula_signature(validated)

        number_of_variables = len(self.graph.nodes)
        degree_bound = DegreeBoundEvaluator(
            number_of_variables=number_of_variables,
        ).evaluate(validated)
        equality_degree = _equality_test_degree(
            degree_bound,
            number_of_variables,
        )
        entropy_bits = _DEFAULT_ENTROPY_BITS
        one_run_bound = _zippel_ratio(
            equality_degree,
            entropy_bits,
        )

        if one_run_bound <= target:
            repetitions = 1
            false_acceptance_bound = one_run_bound
        elif one_run_bound < 1:
            repetitions, false_acceptance_bound = _repeat_until_target(
                one_run_bound,
                target,
            )
        else:
            entropy_bits = _minimum_bits_below_one(
                degree=equality_degree,
                minimum_bits=_DEFAULT_ENTROPY_BITS + 1,
            )
            one_run_bound = _zippel_ratio(
                equality_degree,
                entropy_bits,
            )

            if one_run_bound <= target:
                repetitions = 1
                false_acceptance_bound = one_run_bound
            else:
                repetitions, false_acceptance_bound = _repeat_until_target(
                    one_run_bound,
                    target,
                )

        target_outputs = tuple(Variable(name) for name in self.outcomes)
        target_inputs = tuple(Variable(name) for name in self.treatments)

        for repetition in range(1, repetitions + 1):
            scm = self._sample_scm(entropy_bits)
            joint = self._build_joint(scm)

            candidate = GaussianEvaluator(joint).evaluate(validated)
            target_kernel = self._build_interventional_kernel(scm)

            # The formula may have additional free inputs, as in the
            # Napkin formula. We discard such inputs only if the
            # evaluated kernel is invariant with respect to them.
            extra_input_indices = tuple(
                index
                for index, variable in enumerate(candidate.inputs)
                if variable not in target_inputs
            )

            if any(
                candidate.mean_linear[row, column] != 0
                for row in range(candidate.mean_linear.nrows())
                for column in extra_input_indices
            ):
                return CheckResult(
                    accepted=False,
                    degree=equality_degree,
                    entropy_bits=entropy_bits,
                    repetitions=repetition,
                )

            candidate = _align_kernel(
                candidate,
                outputs=target_outputs,
                inputs=target_inputs,
            )

            if not _kernels_equal(candidate, target_kernel):
                return CheckResult(
                    accepted=False,
                    degree=equality_degree,
                    entropy_bits=entropy_bits,
                    repetitions=repetition,
                )

        return CheckResult(
            accepted=True,
            false_acceptance_bound=false_acceptance_bound,
            degree=equality_degree,
            entropy_bits=entropy_bits,
            repetitions=repetitions,
        )

    def _validate_formula_variables(
        self,
        validated: ValidationResult,
    ) -> None:
        observed_variables = frozenset(
            Variable(name)
            for name, node in self.graph.nodes.items()
            if node.observed
        )

        invalid_variables = frozenset(
            variable
            for variable in validated.used_variables
            if variable.original not in observed_variables
        )

        if invalid_variables:
            raise ValueError(
                "The formula must use only observed variables from the graph. "
                f"Invalid variables: {format_variables(invalid_variables)}."
            )

    def _validate_formula_signature(
        self,
        validated: ValidationResult,
    ) -> None:
        expected_outputs = frozenset(Variable(name) for name in self.outcomes)

        if validated.signature.outputs != expected_outputs:
            raise ValueError(
                "The formula must yield exactly the outputs "
                f"{format_variables(expected_outputs)}, but yielded "
                f"{format_variables(validated.signature.outputs)}."
            )

    def _is_identifiable(self) -> bool:
        if all(node.observed for node in self.graph.nodes.values()):
            return True

        observed_nodes = [
            name for name, node in self.graph.nodes.items() if node.observed
        ]
        directed_edges = [
            (parent.name, child.name)
            for parent in self.graph.nodes.values()
            if parent.observed
            for child in parent.children
            if child.observed
        ]
        bidirected_edges = sorted(
            {
                tuple(sorted(child.name for child in latent.children))
                for latent in self.graph.nodes.values()
                if not latent.observed
            }
        )

        graph = graphs.ADMG(
            observed_nodes,
            di_edges=directed_edges,
            bi_edges=bidirected_edges,
        )
        return identification.OneLineID(
            graph,
            treatments=self.treatments,
            outcomes=self.outcomes,
        ).id()

    def _sample_scm(self, entropy_bits: int) -> _LinearGaussianSCM:
        if entropy_bits < 1:
            raise ValueError("entropy_bits must be positive.")

        variables = tuple(self.graph.nodes)
        index = {
            variable: position for position, variable in enumerate(variables)
        }
        n = len(variables)

        coefficients = [fmpz(0) for _ in range(n * n)]
        for child_name, child in self.graph.nodes.items():
            child_index = index[child_name]
            for parent in child.parents:
                parent_index = index[parent.name]
                coefficients[child_index * n + parent_index] = _sample_fmpz(
                    entropy_bits, signed=True
                )

        intercepts = [
            _sample_fmpz(entropy_bits, signed=True) for _ in variables
        ]
        variances = [_sample_fmpz(entropy_bits) for _ in variables]

        return _LinearGaussianSCM(
            variables=variables,
            coefficients=fmpq_mat(n, n, coefficients),
            intercepts=fmpq_mat(n, 1, intercepts),
            noise_covariance=fmpq_mat(
                n,
                n,
                [
                    variances[i] if i == j else fmpz(0)
                    for i in range(n)
                    for j in range(n)
                ],
            ),
        )

    @staticmethod
    def _build_joint(
        scm: _LinearGaussianSCM,
    ) -> GaussianDistribution:
        mean, covariance = _solve_linear_gaussian(
            scm.coefficients,
            scm.intercepts,
            scm.noise_covariance,
        )

        return GaussianDistribution(
            variables=tuple(Variable(name) for name in scm.variables),
            mean=mean,
            covariance=covariance,
        )

    def _build_interventional_kernel(
        self,
        scm: _LinearGaussianSCM,
    ) -> GaussianKernel:
        index = {
            variable: position
            for position, variable in enumerate(scm.variables)
        }
        treatment_indices = tuple(index[name] for name in self.treatments)
        treatment_set = set(treatment_indices)
        non_treatment_indices = tuple(
            i for i in range(len(scm.variables)) if i not in treatment_set
        )

        coefficients_non_treatment = submatrix(
            scm.coefficients,
            non_treatment_indices,
            non_treatment_indices,
        )
        coefficients_treatment = submatrix(
            scm.coefficients,
            non_treatment_indices,
            treatment_indices,
        )
        intercepts = submatrix(
            scm.intercepts,
            non_treatment_indices,
            (0,),
        )
        noise_covariance = submatrix(
            scm.noise_covariance,
            non_treatment_indices,
            non_treatment_indices,
        )

        identity = _identity(len(non_treatment_indices))
        system = identity - coefficients_non_treatment
        inverse = system.solve(identity)

        mean_intercept = inverse * intercepts
        mean_linear = inverse * coefficients_treatment
        covariance = inverse * noise_covariance * inverse.transpose()

        non_treatment_position = {
            original_index: position
            for position, original_index in enumerate(non_treatment_indices)
        }
        output_indices = tuple(
            non_treatment_position[index[name]] for name in self.outcomes
        )

        return GaussianKernel(
            outputs=tuple(Variable(name) for name in self.outcomes),
            inputs=tuple(Variable(name) for name in self.treatments),
            mean_intercept=submatrix(
                mean_intercept,
                output_indices,
                (0,),
            ),
            mean_linear=submatrix(
                mean_linear,
                output_indices,
                range(len(self.treatments)),
            ),
            covariance=submatrix(
                covariance,
                output_indices,
                output_indices,
            ),
        )


def _validate_variables(
    variables: str | Sequence[str],
    name: str,
    graph: Graph,
) -> tuple[str, ...]:
    if isinstance(variables, str):
        variables = (variables,)
    else:
        variables = tuple(variables)

    if not variables:
        raise ValueError(f"{name} must not be empty.")

    if any(not isinstance(variable, str) for variable in variables):
        raise TypeError(f"{name} must contain only variable names as strings.")

    duplicates = sorted(
        {variable for variable in variables if variables.count(variable) > 1}
    )
    if duplicates:
        raise ValueError(
            f"{name} contains duplicate variables: "
            f"{', '.join(duplicates)}."
        )

    unknown = sorted(
        variable for variable in variables if variable not in graph.nodes
    )
    if unknown:
        raise ValueError(
            f"{name} contains variables not present in the graph: "
            f"{', '.join(unknown)}."
        )

    unobserved = sorted(
        variable
        for variable in variables
        if not graph.nodes[variable].observed
    )
    if unobserved:
        raise ValueError(
            f"{name} must contain only observed variables. "
            f"Unobserved variables: {', '.join(unobserved)}."
        )

    return variables


def _validate_target_bound(
    target_bound: Fraction | float,
) -> Fraction:
    if isinstance(target_bound, Fraction):
        bound = target_bound
    elif isinstance(target_bound, float):
        bound = Fraction(str(target_bound))
    else:
        raise TypeError("target_bound must be a Fraction or float.")

    if not 0 < bound < 1:
        raise ValueError("target_bound must lie strictly between 0 and 1.")

    return bound


def _equality_test_degree(
    candidate: DegreeBound,
    number_of_variables: int,
) -> int:
    target_mean_degree = number_of_variables
    target_covariance_degree = 2 * number_of_variables - 1

    return max(
        candidate.mean_numerator,
        candidate.mean_denominator + target_mean_degree,
        candidate.covariance_numerator,
        candidate.covariance_denominator + target_covariance_degree,
    )


def _zippel_ratio(
    degree: int,
    entropy_bits: int,
) -> Fraction:
    if degree < 0:
        raise ValueError("degree must be non-negative.")
    if entropy_bits < 1:
        raise ValueError("entropy_bits must be positive.")

    return Fraction(degree, 1 << entropy_bits)


def _repeat_until_target(
    one_run_bound: Fraction,
    target_bound: Fraction,
) -> tuple[int, Fraction]:
    if not 0 <= one_run_bound < 1:
        raise ValueError("one_run_bound must lie in [0, 1).")

    repetitions = 1
    repeated_bound = one_run_bound

    while repeated_bound > target_bound:
        repetitions += 1
        repeated_bound *= one_run_bound

    return repetitions, repeated_bound


def _minimum_bits_below_one(
    degree: int,
    minimum_bits: int,
) -> int:
    if degree < 0:
        raise ValueError("degree must be non-negative.")
    if minimum_bits < 1:
        raise ValueError("minimum_bits must be positive.")

    return max(minimum_bits, degree.bit_length())


def _sample_fmpz(
    entropy_bits: int,
    signed: bool = False,
) -> fmpz:
    if entropy_bits < 1:
        raise ValueError("entropy_bits must be at least 1")

    if not signed:
        return fmpz(getrandbits(entropy_bits)) + 1

    magnitude = fmpz(getrandbits(entropy_bits - 1)) + 1
    sign = -1 if getrandbits(1) else 1
    return sign * magnitude


def _solve_linear_gaussian(
    coefficients: fmpq_mat,
    intercepts: fmpq_mat,
    noise_covariance: fmpq_mat,
) -> tuple[fmpq_mat, fmpq_mat]:
    identity = _identity(coefficients.nrows())

    system = identity - coefficients
    inverse = system.solve(identity)

    return (
        inverse * intercepts,
        inverse * noise_covariance * inverse.transpose(),
    )


def _identity(size: int) -> fmpq_mat:
    return fmpq_mat(
        size,
        size,
        [1 if i == j else 0 for i in range(size) for j in range(size)],
    )


def _align_kernel(
    kernel: GaussianKernel,
    outputs: tuple[Variable, ...],
    inputs: tuple[Variable, ...],
) -> GaussianKernel:
    output_index = {variable: i for i, variable in enumerate(kernel.outputs)}
    indices = tuple(output_index[variable] for variable in outputs)

    return GaussianKernel(
        outputs=outputs,
        inputs=inputs,
        mean_intercept=submatrix(
            kernel.mean_intercept,
            indices,
            (0,),
        ),
        mean_linear=align_columns(
            submatrix(
                kernel.mean_linear,
                indices,
                range(len(kernel.inputs)),
            ),
            kernel.inputs,
            inputs,
        ),
        covariance=submatrix(
            kernel.covariance,
            indices,
            indices,
        ),
    )


def _kernels_equal(
    left: GaussianKernel,
    right: GaussianKernel,
) -> bool:
    return (
        left.outputs == right.outputs
        and left.inputs == right.inputs
        and _matrices_equal(left.mean_intercept, right.mean_intercept)
        and _matrices_equal(left.mean_linear, right.mean_linear)
        and _matrices_equal(left.covariance, right.covariance)
    )


def _matrices_equal(left: fmpq_mat, right: fmpq_mat) -> bool:
    return (
        left.nrows() == right.nrows()
        and left.ncols() == right.ncols()
        and all(
            left[i, j] == right[i, j]
            for i in range(left.nrows())
            for j in range(left.ncols())
        )
    )
