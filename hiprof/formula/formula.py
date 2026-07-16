from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Variable:
    """A user-chosen name, optionally carrying a copy index.

    Names are opaque, only the suffix after a prime has structure,
    so X' and X'2 are copies of X.
    """

    name: str
    copy_index: int | None = None

    @classmethod
    def from_token(cls, text: str) -> Variable:
        # Lark has already checked:
        # [A-Z]+(?:0|[1-9][0-9]*)?(?:'(?:0|[1-9][0-9]*)?)?

        if "'" not in text:
            return cls(text)
        name, suffix = text.split("'", maxsplit=1)
        return cls(name, 0 if suffix == "" else int(suffix))

    @property
    def original(self) -> Variable:
        return Variable(self.name)

    def __str__(self) -> str:
        if self.copy_index is None:
            return self.name
        suffix = "" if self.copy_index == 0 else str(self.copy_index)
        return f"{self.name}'{suffix}"


class Formula:
    pass


@dataclass(frozen=True)
class BaseKernel(Formula):
    outputs: tuple[Variable, ...]
    inputs: tuple[Variable, ...] = ()


@dataclass(frozen=True)
class BaseQuotient(Formula):
    """Parser-level syntax; validation normalises this to an ICD node."""

    numerator: BaseKernel
    denominator: BaseKernel


@dataclass(frozen=True)
class Product(Formula):
    # After validation, factors are stored in a valid ltr sequential order.
    # Evaluators may use the left fold ((E1 E2) E3) ... Ek.
    factors: tuple[Formula, ...]


@dataclass(frozen=True)
class Marginalisation(Formula):
    variables: tuple[Variable, ...]
    body: Formula


@dataclass(frozen=True)
class InternalConditionalDivision(Formula):
    denominator_outputs: tuple[Variable, ...]
    denominator_inputs: tuple[Variable, ...]
    body: Formula


@dataclass(frozen=True)
class KernelSignature:
    outputs: frozenset[Variable]
    inputs: frozenset[Variable]

    def __str__(self) -> str:
        return f"{_format_set(self.outputs)} | {_format_set(self.inputs)}"


def _format_set(variables: Iterable[Variable]) -> str:
    text = ", ".join(str(variable) for variable in sorted(variables, key=str))
    return text or "empty"


def _tuple_text(variables: tuple[Variable, ...]) -> str:
    if not variables:
        return "()"
    text = ", ".join(map(str, variables))
    return f"({text + ',' if len(variables) == 1 else text})"


def format_ast(formula: Formula, indent: int = 0) -> str:
    prefix = " " * indent
    field = " " * (indent + 4)

    if isinstance(formula, BaseKernel):
        return (
            f"{prefix}BaseKernel(\n"
            f"{field}outputs={_tuple_text(formula.outputs)},\n"
            f"{field}inputs={_tuple_text(formula.inputs)},\n"
            f"{prefix})"
        )

    if isinstance(formula, BaseQuotient):
        numerator = format_ast(formula.numerator, indent + 8)
        denominator = format_ast(formula.denominator, indent + 8)
        return (
            f"{prefix}BaseQuotient(\n"
            f"{field}numerator=(\n{numerator}\n{field}),\n"
            f"{field}denominator=(\n{denominator}\n{field}),\n"
            f"{prefix})"
        )

    if isinstance(formula, Product):
        factors = ",\n".join(
            format_ast(factor, indent + 8) for factor in formula.factors
        )
        return (
            f"{prefix}Product(\n"
            f"{field}factors=(\n{factors},\n"
            f"{field}),\n{prefix})"
        )

    if isinstance(formula, Marginalisation):
        body = format_ast(formula.body, indent + 8)
        return (
            f"{prefix}Marginalisation(\n"
            f"{field}variables={_tuple_text(formula.variables)},\n"
            f"{field}body=(\n{body}\n{field}),\n"
            f"{prefix})"
        )

    if isinstance(formula, InternalConditionalDivision):
        body = format_ast(formula.body, indent + 8)
        return (
            f"{prefix}InternalConditionalDivision(\n"
            f"{field}denominator_outputs="
            f"{_tuple_text(formula.denominator_outputs)},\n"
            f"{field}denominator_inputs="
            f"{_tuple_text(formula.denominator_inputs)},\n"
            f"{field}body=(\n{body}\n{field}),\n"
            f"{prefix})"
        )

    raise TypeError(f"Unknown formula node: {type(formula).__name__}")
