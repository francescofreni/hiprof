from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from random import getrandbits
from typing import Sequence

from flint import fmpq_mat, fmpz

from hiprof.formula.formula import Variable
from hiprof.formula.validation import ValidationResult, parse_and_validate
from hiprof.graph import parse_graph
from hiprof.utils import format_variables

from ..identification import _latent_projection
from ..utils import validate_variables
from .degree import DegreeBound, DegreeBoundEvaluator
from .gaussian import GaussianDistribution, GaussianEvaluator, GaussianKernel
from .utils import submatrix

_DEFAULT_ENTROPY_BITS = 64
_DEFAULT_TARGET_BOUND = Fraction(1, 10**14)


@dataclass(frozen=True)
class CheckResult:
    """Result returned by :meth:`HPFalsifier.check`.

    :param accepted: Whether the formula passed the falsification check.
    :param false_acceptance_bound: Upper bound on the probability of
        accepting an incorrect formula, when available.
    :param degree: Polynomial degree used for Schwarz-Zippel's bound.
    :param entropy_bits: Number of random bits used per sampled coefficient.
    :param repetitions: Number of independent checks that were run.
    """

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

        assert self.false_acceptance_bound is not None

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

    def joint_distribution(self) -> GaussianDistribution:
        identity = _identity(len(self.variables))
        system = identity - self.coefficients
        inverse = system.solve(identity)

        return GaussianDistribution(
            variables=tuple(Variable(name) for name in self.variables),
            mean=inverse * self.intercepts,
            covariance=(inverse * self.noise_covariance * inverse.transpose()),
        )

    def interventional_kernel(
        self,
        treatments: tuple[str, ...],
        outcomes: tuple[str, ...],
        conditioning_set: tuple[str, ...] = (),
    ) -> GaussianKernel:
        index = {
            variable: position
            for position, variable in enumerate(self.variables)
        }
        treatment_indices = tuple(index[name] for name in treatments)
        treatment_set = set(treatment_indices)
        non_treatment_indices = tuple(
            i for i in range(len(self.variables)) if i not in treatment_set
        )

        coefficients_non_treatment = submatrix(
            self.coefficients,
            non_treatment_indices,
            non_treatment_indices,
        )
        coefficients_treatment = submatrix(
            self.coefficients,
            non_treatment_indices,
            treatment_indices,
        )
        intercepts = submatrix(
            self.intercepts,
            non_treatment_indices,
            (0,),
        )
        noise_covariance = submatrix(
            self.noise_covariance,
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
        output_variables = outcomes + conditioning_set
        output_indices = tuple(
            non_treatment_position[index[name]] for name in output_variables
        )

        kernel = GaussianKernel(
            outputs=tuple(Variable(name) for name in output_variables),
            inputs=tuple(Variable(name) for name in treatments),
            mean_intercept=submatrix(
                mean_intercept,
                output_indices,
                (0,),
            ),
            mean_linear=submatrix(
                mean_linear,
                output_indices,
                range(len(treatments)),
            ),
            covariance=submatrix(
                covariance,
                output_indices,
                output_indices,
            ),
        )

        if not conditioning_set:
            return kernel

        return kernel.condition_on(
            tuple(Variable(name) for name in conditioning_set)
        )


class HPFalsifier:
    """High-probability falsifier of observational
    formulas for interventional distributions.

    The falsifier tests a candidate formula by evaluating it
    on randomly sampled linear Gaussian models and then
    comparing it with the target interventional distribution.
    """

    def __init__(
        self,
        graph: str,
        treatments: str | Sequence[str],
        outcomes: str | Sequence[str],
        latents: str | Sequence[str] | None = None,
        conditioning_set: str | Sequence[str] | None = None,
    ) -> None:
        """Initialise a falsifier for a causal query.

        :param graph: Graph specification using ``->`` for directed edges and
            ``<->`` for bidirected edges.
        :param treatments: Treatment variable name, or sequence of treatment
            variable names.
        :param outcomes: Outcome variable name, or sequence of outcome
            variable names.
        :param latents: Variable name, or sequence of variable names, to treat
            as unobserved. Every latent variable must be a node in ``graph``.
        :param conditioning_set: Variable name, or sequence of variable
            names, to additionally condition on, that is, the target becomes
            ``p(outcomes | do(treatments), conditioning_set)``. Must be
            disjoint from treatments and outcomes.
        :raises TypeError: If treatments, outcomes, latents, or
            conditioning_set have invalid types.
        :raises ValueError: If the graph, treatments, outcomes, latents, or
            conditioning_set are invalid.
        """
        self.graph = parse_graph(graph)

        if latents is None:
            self.latents: tuple[str, ...] = ()
        else:
            self.latents = validate_variables(
                latents,
                name="Latents",
                graph=self.graph,
            )

        for latent in self.latents:
            self.graph.nodes[latent].observed = False

        self.treatments = validate_variables(
            treatments,
            name="Treatments",
            graph=self.graph,
        )
        self.outcomes = validate_variables(
            outcomes,
            name="Outcomes",
            graph=self.graph,
        )

        if conditioning_set is None:
            self.conditioning_set: tuple[str, ...] = ()
        else:
            self.conditioning_set = validate_variables(
                conditioning_set,
                name="Conditioning set",
                graph=self.graph,
            )

        for label_a, set_a, label_b, set_b in (
            ("Treatments", self.treatments, "outcomes", self.outcomes),
            (
                "Treatments",
                self.treatments,
                "the conditioning set",
                self.conditioning_set,
            ),
            (
                "Outcomes",
                self.outcomes,
                "the conditioning set",
                self.conditioning_set,
            ),
        ):
            overlap = sorted(set(set_a) & set(set_b))
            if overlap:
                raise ValueError(
                    f"{label_a} and {label_b} must be disjoint. "
                    f"Overlapping variables: {', '.join(overlap)}."
                )

    def check(
        self,
        formula: str | None,
        target_bound: Fraction | float = _DEFAULT_TARGET_BOUND,
        redundant_inputs: str | Sequence[str] | None = None,
    ) -> CheckResult:
        """Check a formula for the target interventional distribution.

        If ``formula`` is ``None``, the check accepts when the target
        interventional distribution is non-identifiable.

        :param formula: formula string to check, or ``None`` to check a
            non-identifiability claim.
        :param target_bound: Desired upper bound on the false-acceptance
            probability.
        :param redundant_inputs: Variables that may appear as redundant
            free inputs of the final kernel, but are declared to be
            redundant because the kernel is constant with respect to them.
            They are not marginalized or removed semantically.
        :returns: CheckResult containing the acceptance decision and, when
            applicable, the false-acceptance bound.
        :raises TypeError: If ``formula`` or ``target_bound`` has an invalid
            type.
        :raises ValueError: If the formula is invalid for the configured graph
            or causal query.
        :raises ImportError: If ``formula`` is ``None`` and the optional
            identification dependencies are not installed.
        """
        if formula is not None and not isinstance(formula, str):
            raise TypeError("formula must be a string or None.")

        declared_redundant_inputs = self._validate_redundant_inputs(
            redundant_inputs,
        )

        if formula is None:
            if declared_redundant_inputs:
                raise ValueError(
                    "redundant_inputs cannot be used when formula is None."
                )

            return CheckResult(
                accepted=not self._is_identifiable(),
            )

        target = _validate_target_bound(target_bound)

        validated = parse_and_validate(formula)
        self._validate_formula_variables(validated)
        self._validate_formula_signature(
            validated,
            declared_redundant_inputs,
        )

        number_of_variables = len(self.graph.nodes)
        degree_bound = DegreeBoundEvaluator(
            number_of_variables=number_of_variables,
        ).evaluate(validated)
        equality_degree = _equality_test_degree(
            degree_bound,
            number_of_variables,
            len(self.conditioning_set),
        )
        entropy_bits = _DEFAULT_ENTROPY_BITS
        one_run_bound = _zippel_ratio(
            equality_degree,
            entropy_bits,
        )

        if one_run_bound >= Fraction(1, 2):
            entropy_bits = _minimum_bits_below_half(
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
        target_inputs = tuple(
            Variable(name) for name in self.treatments
        ) + tuple(Variable(name) for name in self.conditioning_set)

        for repetition in range(1, repetitions + 1):
            scm = self._sample_scm(entropy_bits)
            joint = scm.joint_distribution()

            candidate = GaussianEvaluator(joint).evaluate(validated)
            target_kernel = scm.interventional_kernel(
                self.treatments,
                self.outcomes,
                self.conditioning_set,
            )

            if declared_redundant_inputs:
                # check invariance before dropping redundant input
                # columns to allow early rejection if the linear
                # coefficient is non-zero (which means that the
                # inputs are not redundant)
                redundant_input_indices = tuple(
                    index
                    for index, variable in enumerate(candidate.inputs)
                    if variable in declared_redundant_inputs
                )

                has_nonzero_redundant_coefficient = any(
                    candidate.mean_linear[row, column] != 0
                    for row in range(candidate.mean_linear.nrows())
                    for column in redundant_input_indices
                )

                if has_nonzero_redundant_coefficient:
                    return CheckResult(
                        accepted=False,
                        degree=equality_degree,
                        entropy_bits=entropy_bits,
                        repetitions=repetition,
                    )

            candidate = candidate.align(
                outputs=target_outputs,
                inputs=target_inputs,
            )

            if candidate != target_kernel:
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

    def _validate_redundant_inputs(
        self,
        redundant_inputs: str | Sequence[str] | None,
    ) -> frozenset[Variable]:
        if redundant_inputs is None:
            return frozenset()

        names = validate_variables(
            redundant_inputs,
            name="Redundant inputs",
            graph=self.graph,
        )

        forbidden = frozenset(names) & (
            frozenset(self.treatments)
            | frozenset(self.outcomes)
            | frozenset(self.conditioning_set)
        )
        if forbidden:
            raise ValueError(
                "Redundant inputs must be distinct from treatments, "
                "outcomes, and the conditioning set. Invalid variables: "
                f"{', '.join(sorted(forbidden))}."
            )

        return frozenset(Variable(name) for name in names)

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
        declared_redundant_inputs: frozenset[Variable],
    ) -> None:
        expected_outputs = frozenset(Variable(name) for name in self.outcomes)

        if validated.signature.outputs != expected_outputs:
            raise ValueError(
                "The formula must yield exactly the outputs "
                f"{format_variables(expected_outputs)}, but yielded "
                f"{format_variables(validated.signature.outputs)}."
            )

        allowed_inputs = frozenset(
            Variable(name) for name in self.treatments
        ) | frozenset(Variable(name) for name in self.conditioning_set)
        extra_inputs = validated.signature.inputs - allowed_inputs

        undeclared_inputs = extra_inputs - declared_redundant_inputs
        if undeclared_inputs:
            names = sorted(variable.name for variable in undeclared_inputs)
            raise ValueError(
                "The formula has free inputs other than the treatments "
                "and the conditioning set: "
                f"{format_variables(undeclared_inputs)}. These inputs must "
                "either be eliminated from the final formula or explicitly "
                "declared for invariance checking, for example "
                f"`redundant_inputs={names!r}`."
            )

        unused_declarations = declared_redundant_inputs - extra_inputs
        if unused_declarations:
            raise ValueError(
                "The following declared redundant inputs are not free inputs "
                "of the formula: "
                f"{format_variables(unused_declarations)}."
            )

    def _is_identifiable(self) -> bool:
        if all(node.observed for node in self.graph.nodes.values()):
            return True

        try:
            from y0.algorithm.identify import identify_outcomes
            from y0.dsl import Variable as Y0Variable
            from y0.graph import NxMixedGraph
        except ImportError as error:
            raise ImportError(
                "Verifying non-identifiability claims (`formula=None`) "
                "requires the optional `y0` dependency. Install it with "
                '`pip install "hiprof[identification]"`.'
            ) from error

        variables = {
            name: Y0Variable(name)
            for name, node in self.graph.nodes.items()
            if node.observed
        }
        directed_edges, bidirected_edges = _latent_projection(self.graph)
        admg = NxMixedGraph.from_edges(
            directed=[
                (variables[parent], variables[child])
                for parent, child in sorted(directed_edges)
            ],
            undirected=[
                (variables[left], variables[right])
                for left, right in sorted(bidirected_edges)
            ],
        )
        for variable in variables.values():
            admg.add_node(variable)

        conditions = {variables[name] for name in self.conditioning_set}
        formula = identify_outcomes(
            admg,
            treatments={variables[name] for name in self.treatments},
            outcomes={variables[name] for name in self.outcomes},
            conditions=conditions or None,
        )
        return formula is not None

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
    conditioning_size: int,
) -> int:
    n = number_of_variables
    target = DegreeBoundEvaluator.conditional_degree_bound(
        DegreeBound(n, 0, 2 * n - 1, 0),
        conditioning_size,
    )

    return max(
        candidate.mean_numerator + target.mean_denominator,
        target.mean_numerator + candidate.mean_denominator,
        candidate.covariance_numerator + target.covariance_denominator,
        target.covariance_numerator + candidate.covariance_denominator,
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


def _minimum_bits_below_half(
    degree: int,
    minimum_bits: int,
) -> int:
    if degree < 0:
        raise ValueError("degree must be non-negative.")
    if minimum_bits < 1:
        raise ValueError("minimum_bits must be positive.")

    return max(minimum_bits, degree.bit_length() + 1)


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


def _identity(size: int) -> fmpq_mat:
    return fmpq_mat(
        size,
        size,
        [1 if i == j else 0 for i in range(size) for j in range(size)],
    )
