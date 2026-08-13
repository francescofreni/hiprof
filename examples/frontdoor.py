"""Verify different front-door formulas.

Examples are based on Figure 1 of:
Wienöbst et al., "Linear-Time Algorithms for Front-Door Adjustment in
Causal Graphs", AAAI 2024.
"""

from collections.abc import Sequence

from hiprof import HPFalsifier


def front_door_formula(variables: Sequence[str]) -> str:
    names = ", ".join(variables)
    return (
        f"sum_{{{names}}} {{ "
        f"p({names} | X) "
        f"sum_{{X'}} {{ p(Y | X', {names}) p(X') }} "
        f"}}"
    )


def report_sets(
    falsifier: HPFalsifier,
    cases: Sequence[tuple[Sequence[str], bool]],
) -> None:
    print(
        f"{'variable set':<24}"
        f"{'front-door criterion':<24}"
        "formula accepted"
    )
    print("-" * 64)

    for variables, satisfies_criterion in cases:
        label = "{" + ", ".join(variables) + "}"
        formula = front_door_formula(variables)
        accepted = falsifier.check(formula).accepted
        yes_no_satisfied = "yes" if satisfies_criterion else "no"
        yes_no_accepted = "yes" if accepted else "no"
        print(f"{label:<24}" f"{yes_no_satisfied:<24}" f"{yes_no_accepted}")


def main() -> None:
    print("ADMG (i): standard front-door graph")
    falsifier = HPFalsifier(
        graph="Z -> Y; X -> Z; X <-> Y",
        treatments="X",
        outcomes="Y",
    )
    report_sets(
        falsifier,
        [
            (("Z",), True),
        ],
    )

    print("\nADMG (ii): a longer directed mediator structure")
    falsifier = HPFalsifier(
        graph="X -> A; A -> B; A -> C; B -> D; " "C -> D; D -> Y; X <-> Y",
        treatments="X",
        outcomes="Y",
    )
    report_sets(
        falsifier,
        [
            (("A",), True),
            (("A", "B", "C", "D"), True),
            (("C",), False),
        ],
    )

    print(
        "\nADMG (iii): the front-door criterion is sound, "
        "but not exhaustively complete"
    )
    print(
        "The sets {A, B} and {A, C} fail the front-door criterion, "
        "but the front-door formulas are still valid."
    )
    falsifier = HPFalsifier(
        graph="X -> A; A -> Y; X <-> Y; B -> Y; C -> Y; " "D -> B; D -> C",
        treatments="X",
        outcomes="Y",
    )
    report_sets(
        falsifier,
        [
            (("A", "B", "C"), True),
            (("A",), True),
            (("A", "B"), False),
            (("A", "C"), False),
            (("B", "C"), False),
        ],
    )

    print("\nADMG (iv): valid formulas beyond front-door sets")
    falsifier = HPFalsifier(
        graph=(
            "X -> A; A -> B; B -> C; B -> Y; C -> Y; "
            "D -> Y; E -> C; F -> E; F -> G; G -> Y; "
            "X <-> Y; A <-> D"
        ),
        treatments="X",
        outcomes="Y",
    )

    print("\nThree sets satisfying the front-door criterion:")
    report_sets(
        falsifier,
        [
            (("A", "D"), True),
            (("B", "D"), True),
            (("A", "B", "D"), True),
        ],
    )

    print("\nTwo insufficient sets:")
    report_sets(
        falsifier,
        [
            (("A",), False),
            (("B",), False),
        ],
    )

    print(
        "\nSets that fail the front-door criterion but whose "
        "front-door formulas are accepted:"
    )
    non_front_door_sets = [
        ("A", "D", "C"),
        ("A", "D", "E"),
        ("B", "D", "E"),
        ("B", "D", "F"),
        ("A", "D", "G"),
        ("B", "D", "G"),
        ("A", "D", "C", "E"),
        ("B", "D", "C", "E"),
        ("A", "D", "C", "F"),
        ("A", "D", "C", "G"),
        ("A", "B", "D", "E"),
        ("A", "B", "D", "G"),
        ("A", "B", "D", "C", "E"),
        ("A", "D", "C", "E", "F"),
        ("A", "D", "C", "E", "G"),
        ("A", "D", "C", "F", "G"),
    ]
    report_sets(
        falsifier,
        [(variables, False) for variables in non_front_door_sets],
    )


if __name__ == "__main__":
    main()
