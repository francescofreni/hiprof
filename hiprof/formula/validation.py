from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .formula import (
    BaseKernel,
    BaseQuotient,
    Formula,
    InternalConditionalDivision,
    KernelSignature,
    Marginalisation,
    Product,
    Variable,
)
from .parser import parse


@dataclass(frozen=True)
class ValidationResult:
    formula: Formula
    signature: KernelSignature


class ValidationError(ValueError):
    pass


def _format_variables(variables: Iterable[Variable]) -> str:
    return ", ".join(str(variable) for variable in sorted(variables, key=str))


def _require_distinct(
    variables: tuple[Variable, ...],
    context: str,
) -> None:
    seen: set[Variable] = set()

    for variable in variables:
        if variable in seen:
            raise ValidationError(
                f"Repeated variable in {context}: {variable}"
            )

        seen.add(variable)


def _validate_base_kernel(kernel: BaseKernel) -> ValidationResult:
    seen: set[Variable] = set()

    for variable in kernel.outputs + kernel.inputs:
        original = variable.original

        if original in seen:
            raise ValidationError(
                "A base kernel contains multiple versions "
                f"of variable {original}"
            )

        seen.add(original)

    return ValidationResult(
        formula=kernel,
        signature=KernelSignature(
            outputs=frozenset(kernel.outputs),
            inputs=frozenset(kernel.inputs),
        ),
    )


def _validate_base_quotient(
    quotient: BaseQuotient,
) -> ValidationResult:
    _validate_base_kernel(quotient.numerator)
    _validate_base_kernel(quotient.denominator)

    # A base quotient is convenience syntax for the corresponding ICD.
    # The denominator expression supplies the derived kernel's signature.
    icd = InternalConditionalDivision(
        denominator_outputs=quotient.denominator.outputs,
        denominator_inputs=quotient.denominator.inputs,
        body=quotient.numerator,
    )

    return _validate_icd(icd)


def _validate_product(
    product: Product,
) -> ValidationResult:
    factors = [validate(factor) for factor in product.factors]

    all_outputs: set[Variable] = set()

    for factor in factors:
        overlap = all_outputs & factor.signature.outputs

        if overlap:
            raise ValidationError(
                "Product factors repeat outputs: " + _format_variables(overlap)
            )

        all_outputs.update(factor.signature.outputs)

    all_inputs = set().union(*(factor.signature.inputs for factor in factors))

    external_inputs = all_inputs - all_outputs

    available = set(external_inputs)
    remaining = factors.copy()
    ordered: list[ValidationResult] = []

    while remaining:
        for index, factor in enumerate(remaining):
            if factor.signature.inputs <= available:
                ordered.append(factor)
                available.update(factor.signature.outputs)
                remaining.pop(index)
                break
        else:
            blocked = "; ".join((
                f"({_format_variables(factor.signature.outputs)}) "
                f"needs "
                f"({_format_variables(factor.signature.inputs - available)})")
                for factor in remaining
            )

            raise ValidationError(
                "Product has no valid sequential ordering; "
                f"blocked factors: {blocked}"
            )

    normalised_product = Product(
        factors=tuple(factor.formula for factor in ordered)
    )

    return ValidationResult(
        formula=normalised_product,
        signature=KernelSignature(
            outputs=frozenset(all_outputs),
            inputs=frozenset(external_inputs),
        ),
    )


def _validate_marginalisation(
    marginalisation: Marginalisation,
) -> ValidationResult:
    _require_distinct(
        marginalisation.variables,
        "marginalisation subscript",
    )

    body = validate(marginalisation.body)
    variables = frozenset(marginalisation.variables)

    missing = variables - body.signature.outputs

    if missing:
        raise ValidationError(
            "Cannot marginalise non-outputs: " + _format_variables(missing)
        )

    normalised_marginalisation = Marginalisation(
        variables=marginalisation.variables,
        body=body.formula,
    )

    return ValidationResult(
        formula=normalised_marginalisation,
        signature=KernelSignature(
            outputs=(body.signature.outputs - variables),
            inputs=body.signature.inputs,
        ),
    )


def _validate_icd(
    icd: InternalConditionalDivision,
) -> ValidationResult:
    _require_distinct(
        icd.denominator_outputs,
        "ICD denominator outputs",
    )
    _require_distinct(
        icd.denominator_inputs,
        "ICD denominator inputs",
    )

    body = validate(icd.body)

    denominator_outputs = frozenset(icd.denominator_outputs)
    denominator_inputs = frozenset(icd.denominator_inputs)

    overlap = denominator_outputs & denominator_inputs

    if overlap:
        raise ValidationError(
            "ICD denominator outputs and inputs overlap: "
            + _format_variables(overlap)
        )

    missing_outputs = denominator_outputs - body.signature.outputs

    if missing_outputs:
        raise ValidationError(
            "ICD denominator outputs must be body outputs; "
            "not outputs: " + _format_variables(missing_outputs)
        )

    omitted_body_inputs = body.signature.inputs - denominator_inputs

    if omitted_body_inputs:
        raise ValidationError(
            "ICD denominator must retain every body input; "
            "omitted: " + _format_variables(omitted_body_inputs)
        )

    remaining_outputs = body.signature.outputs - denominator_outputs

    invalid_inputs = denominator_inputs - (
        body.signature.inputs | remaining_outputs
    )

    if invalid_inputs:
        raise ValidationError(
            "ICD denominator inputs must be body inputs "
            "or remaining body outputs; invalid: "
            + _format_variables(invalid_inputs)
        )

    normalised_icd = InternalConditionalDivision(
        denominator_outputs=icd.denominator_outputs,
        denominator_inputs=icd.denominator_inputs,
        body=body.formula,
    )

    return ValidationResult(
        formula=normalised_icd,
        signature=KernelSignature(
            outputs=remaining_outputs,
            inputs=(body.signature.inputs | denominator_outputs),
        ),
    )


def validate(
    formula: Formula,
) -> ValidationResult:
    """Validate a formula and return its recursively normalised AST."""
    if isinstance(formula, BaseKernel):
        return _validate_base_kernel(formula)

    if isinstance(formula, BaseQuotient):
        return _validate_base_quotient(formula)

    if isinstance(formula, Product):
        return _validate_product(formula)

    if isinstance(formula, Marginalisation):
        return _validate_marginalisation(formula)

    if isinstance(
        formula,
        InternalConditionalDivision,
    ):
        return _validate_icd(formula)

    raise TypeError("Unknown formula node: " f"{type(formula).__name__}")


def parse_and_validate(source: str) -> ValidationResult:
    return validate(parse(source))
