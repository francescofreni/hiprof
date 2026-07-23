from typing import Iterable

from .formula.formula import Variable


def format_variables(variables: Iterable[Variable]) -> str:
    return ", ".join(str(variable) for variable in sorted(variables, key=str))
