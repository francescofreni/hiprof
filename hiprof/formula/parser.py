from pathlib import Path
from typing import Any, cast

from lark import Lark, Token, Transformer, v_args

from .formula import (
    BaseKernel,
    BaseQuotient,
    Formula,
    InternalConditionalDivision,
    Marginalisation,
    Product,
    Variable,
)


@v_args(inline=True)
class ToAST(Transformer):
    @staticmethod
    def VARIABLE(token: Token) -> Variable:
        return Variable.from_token(str(token))

    @staticmethod
    def variables(*variables: Variable) -> tuple[Variable, ...]:
        return variables

    @staticmethod
    def base_kernel(
        outputs: tuple[Variable, ...],
        inputs: tuple[Variable, ...] = (),
    ) -> BaseKernel:
        return BaseKernel(outputs=outputs, inputs=inputs)

    @staticmethod
    def base_quotient(
        numerator: BaseKernel,
        denominator: BaseKernel,
    ) -> BaseQuotient:
        return BaseQuotient(numerator, denominator)

    @staticmethod
    def product(*factors: Formula) -> Product:
        return Product(factors)

    @staticmethod
    def marginalisation(
        _operator: Token,
        variables: tuple[Variable, ...],
        body: Formula,
    ) -> Marginalisation:
        return Marginalisation(variables, body)

    @staticmethod
    def internal_conditional_division(
        *items: Any,
    ) -> InternalConditionalDivision:
        if len(items) == 2:
            denominator_outputs, body = items
            denominator_inputs = ()
        else:
            denominator_outputs, denominator_inputs, body = items
        return InternalConditionalDivision(
            denominator_outputs=denominator_outputs,
            denominator_inputs=denominator_inputs,
            body=body,
        )

    @staticmethod
    def grouped(formula: Formula) -> Formula:
        return formula


PARSER = Lark.open(
    str(Path(__file__).with_name("grammar.lark")),
    parser="lalr",
    lexer="contextual",
    start="start",
    transformer=ToAST(),
    maybe_placeholders=False,
)


def parse(source: str) -> Formula:
    """Parse a hiprof formula into an abstract syntax tree.

    This function checks only syntax. Use
    :func:`hiprof.formula.validation.parse_and_validate` to also validate and
    normalise the formula.

    :param source: Formula source string.
    :returns: Parsed formula AST.
    :raises lark.exceptions.UnexpectedInput: If ``source`` is syntactically
        invalid.
    """
    return cast(Formula, PARSER.parse(source))
