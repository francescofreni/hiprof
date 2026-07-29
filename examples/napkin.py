"""Verify the napkin formula."""

from textwrap import dedent

from hiprof import HPFalsifier


def report(
    falsifier: HPFalsifier,
    label: str,
    formula: str,
) -> None:
    formula = dedent(formula).strip()

    print(f"\n{label}")
    print("Formula:")
    print(formula)
    print(f"Result: {falsifier.check(formula).accepted}")


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
        "Napkin formula retaining Z as a free input",
        conditional_formula,
    )
    report(
        falsifier,
        "Observational conditional",
        "p(Y | X)",
    )


if __name__ == "__main__":
    main()
