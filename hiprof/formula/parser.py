from pathlib import Path

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
    def VARIABLE(self, token: Token) -> Variable:
        return Variable.from_token(str(token))

    def variables(self, *variables: Variable) -> tuple[Variable, ...]:
        return variables

    def base_kernel(
        self,
        outputs: tuple[Variable, ...],
        inputs: tuple[Variable, ...] = (),
    ) -> BaseKernel:
        return BaseKernel(outputs=outputs, inputs=inputs)

    def base_quotient(
        self,
        numerator: BaseKernel,
        denominator: BaseKernel,
    ) -> BaseQuotient:
        return BaseQuotient(numerator, denominator)

    def product(self, *factors: Formula) -> Product:
        return Product(factors)

    def marginalisation(
        self,
        _operator: Token,
        variables: tuple[Variable, ...],
        body: Formula,
    ) -> Marginalisation:
        return Marginalisation(variables, body)

    def internal_conditional_division(
        self, *items
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

    def grouped(self, formula: Formula) -> Formula:
        return formula


PARSER = Lark.open(
    Path(__file__).with_name("grammar.lark"),
    parser="lalr",
    lexer="contextual",
    start="start",
    transformer=ToAST(),
    maybe_placeholders=False,
)


def parse(source: str) -> Formula:
    return PARSER.parse(source)
