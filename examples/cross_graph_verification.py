"""Compare formulas across graphs to reproduce Figure 1 of
F. Freni, L. Henckel, and S. Weichwald.
"Verifying formulas for interventional distributions".
arXiv preprint arXiv:2607.13883, 2026.
"""

from collections.abc import Sequence
from textwrap import dedent

from hiprof import HPFalsifier, IDAlgorithm


def print_formula(label: str, formula: str) -> None:
    print(f"\n{label}")
    print(dedent(formula).strip())


def verify_formula(
    label: str,
    formula: str,
    checks: Sequence[tuple[str, HPFalsifier]],
) -> None:
    formula = dedent(formula).strip()

    print(f"\n{label}")
    print("Formula:")
    print(formula)

    for graph_label, falsifier in checks:
        print(f"{graph_label}: {falsifier.check(formula).accepted}")


def main() -> None:
    graph_1 = "T -> M; M -> Y; C -> T; C -> Y"
    graph_2 = "T -> M; M -> Y; T -> C; Y -> C"

    id_formula_1 = IDAlgorithm(
        graph=graph_1,
        treatments="T",
        outcomes="Y",
    ).run()
    id_formula_2 = IDAlgorithm(
        graph=graph_2,
        treatments="T",
        outcomes="Y",
    ).run()

    print("Graph 1")
    print(graph_1)
    print_formula("Formula returned by ID", str(id_formula_1))

    print("\nGraph 2")
    print(graph_2)
    print_formula("Formula returned by ID", str(id_formula_2))

    adjustment = """
        sum_{C} {
            p(Y | C, T) p(C)
        }
    """
    conditional = "p(Y | T)"
    front_door = """
        sum_{M} {
            p(M | T)
            sum_{T'} {
                p(Y | M, T') p(T')
            }
        }
    """

    falsifier_1 = HPFalsifier(
        graph=graph_1,
        treatments="T",
        outcomes="Y",
    )
    falsifier_2 = HPFalsifier(
        graph=graph_2,
        treatments="T",
        outcomes="Y",
    )

    print("\nGraph-specific formulas checked on the other graph")
    verify_formula(
        "Observational conditional from graph 2, checked on graph 1",
        conditional,
        [("Graph 1", falsifier_1)],
    )
    verify_formula(
        "Adjustment formula from graph 1, checked on graph 2",
        adjustment,
        [("Graph 2", falsifier_2)],
    )

    print("\nA front-door formula that identifies in both graphs")
    verify_formula(
        "Front-door formula",
        front_door,
        [
            ("Graph 1", falsifier_1),
            ("Graph 2", falsifier_2),
        ],
    )


if __name__ == "__main__":
    main()
