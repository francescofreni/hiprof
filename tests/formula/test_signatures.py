from __future__ import annotations

from hiprof.formula.formula import KernelSignature, Variable
from hiprof.formula.validation import parse_and_validate


def names(signature: KernelSignature) -> tuple[set[str], set[str]]:
    return (
        {str(variable) for variable in signature.outputs},
        {str(variable) for variable in signature.inputs},
    )


def test_base_kernel_signature() -> None:
    result = parse_and_validate("p(Y, Z | X)")

    assert names(result.signature) == ({"Y", "Z"}, {"X"})
    assert {str(variable) for variable in result.used_variables} == {
        "X",
        "Y",
        "Z",
    }


def test_base_quotient_normalises_to_icd_signature() -> None:
    result = parse_and_validate("p(A, B | C) / p(B | C)")

    assert names(result.signature) == ({"A"}, {"B", "C"})


def test_product_signature_discards_inputs_generated_by_other_factors() -> (
    None
):
    result = parse_and_validate("p(Y | X, Z) p(Z | X)")

    assert names(result.signature) == ({"Y", "Z"}, {"X"})


def test_marginalisation_removes_only_outputs() -> None:
    result = parse_and_validate("sum_{Z} { p(Y | X, Z) p(Z | X) }")

    assert names(result.signature) == ({"Y"}, {"X"})


def test_icd_signature_retains_denominator_outputs_as_inputs() -> None:
    result = parse_and_validate("icd_{X | Z} { p(Y, X | Z) }")

    assert names(result.signature) == ({"Y"}, {"X", "Z"})


def test_kernel_signature_text_formats_empty_sets() -> None:
    signature = KernelSignature(
        outputs=frozenset({Variable("Y")}),
        inputs=frozenset(),
    )

    assert str(signature) == "Y | \u2205"
