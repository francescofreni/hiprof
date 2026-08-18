"""Verify the napkin formula and redundant-input handling."""

from __future__ import annotations

from textwrap import dedent
from typing import Sequence

from hiprof import HPFalsifier


def report(
    falsifier: HPFalsifier,
    label: str,
    formula: str,
    redundant_inputs: str | Sequence[str] | None = None,
) -> None:
    formula = dedent(formula).strip()

    print(f"\n{label}")
    print("Formula:")
    print(formula)

    if redundant_inputs is not None:
        print(f"Declared redundant inputs: {redundant_inputs}")

    try:
        result = falsifier.check(
            formula,
            redundant_inputs=redundant_inputs,
        )
    except ValueError as error:
        print(f"Validation error: {error}")
        return

    print(f"Result: {result.accepted}")


def main() -> None:
    falsifier = HPFalsifier(
        graph="X -> Y; W -> Z; Z -> X; X <-> W; W <-> Y",
        treatments="X",
        outcomes="Y",
    )

    summed_formula = """
        sum_{Z} {
            icd_{X | Z} {
                sum_{W} { p(Y, X | Z, W) p(W) }
            }
            p(Z)
        }
    """

    conditional_formula = """
        icd_{X | Z} {
            sum_{W} { p(Y, X | Z, W) p(W) }
        }
    """

    print("Napkin graph")
    print("X -> Y; W -> Z; Z -> X; X <-> W; W <-> Y")

    report(
        falsifier,
        "Napkin formula marginalized over Z",
        summed_formula,
    )
    report(
        falsifier,
        "Napkin formula retaining undeclared free input Z",
        conditional_formula,
    )
    report(
        falsifier,
        "Napkin formula with Z declared redundant",
        conditional_formula,
        redundant_inputs="Z",
    )
    report(
        falsifier,
        "Observational conditional",
        "p(Y | X)",
    )


if __name__ == "__main__":
    main()
