from fractions import Fraction
import re
from typing import Tuple
from lark import Transformer, v_args
from hiprof.formula.formula import Distribution, Integral, p

_PRIME_SUFFIX = re.compile(r"'+[0-9]*$")


def normalize_var_name(name: str) -> str:
    """Map X', X'', X'2, ... to X."""
    return _PRIME_SUFFIX.sub("", name)


@v_args(inline=True)
class TreeToFormula(Transformer):
    """Transform the parse tree into formula AST nodes."""

    def __init__(self, exact: bool = False):
        super().__init__()
        self.exact = exact

    @staticmethod
    def VAR(token):
        return str(token)

    def number(self, token):
        if self.exact:
            return Fraction(str(token))
        return float(token)

    @staticmethod
    def var_eq(name, value):
        return (normalize_var_name(name), value)

    @staticmethod
    def var_name(name):
        return (normalize_var_name(name), None)

    @staticmethod
    def item_list(*items):
        return tuple(items)

    @staticmethod
    def cond(cond_list):
        return cond_list

    @staticmethod
    def start(expr):
        return expr

    @staticmethod
    def mul(a, b):
        return a * b

    @staticmethod
    def div(a, b):
        return a / b

    @staticmethod
    def dist(targets, cond_list=None) -> Distribution:
        given = () if cond_list is None else cond_list
        return p(targets, given)

    @staticmethod
    def name_list(*names) -> Tuple[str, ...]:
        return tuple(normalize_var_name(name) for name in names)

    @staticmethod
    def integral(_op, vars, integrand) -> Integral:
        return Integral(integrand, vars)

    @staticmethod
    def summation(_op, vars, integrand) -> Integral:
        return Integral(integrand, vars)
