from typing import Iterable, Sequence

from hiprof.graph import Graph

from .formula.formula import Variable


def format_variables(variables: Iterable[Variable]) -> str:
    return ", ".join(str(variable) for variable in sorted(variables, key=str))


def validate_variables(
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
