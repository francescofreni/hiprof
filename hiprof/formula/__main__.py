from lark import UnexpectedInput

from ..verification import DegreeBoundEvaluator
from .formula import format_ast
from .validation import ValidationError, parse_and_validate


EXAMPLES = [
    "p(A, B) / p(B)",
    "p(X)p(X')",
    "p(A, B | C) / p(B | C)",
    "p(C | F) (p(A, B) / p(B)) p(F)",
    "icd_{A |} { p(A, B) }",
    "icd_{X | Z} { int_{W} { p(Y, X | Z, W) p(W) } }",
    "int_{Z} { p(Z | X) int_{X} { p(Y | X, Z) p(X) } }",
    "int_{Z} { int_{X'} { p(Z | X) p(Y | X', Z) p(X') } }",
]


def main() -> None:
    evaluator = DegreeBoundEvaluator(number_of_observed_variables=10)

    for source in EXAMPLES:
        print("\n" + "=" * 78 + "\n")
        print(source)
        try:
            result = parse_and_validate(source)
            print(format_ast(result.formula))
            print("signature:", result.signature)
            print(evaluator.evaluate(result))
        except (UnexpectedInput, ValidationError) as error:
            print(f"{type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
