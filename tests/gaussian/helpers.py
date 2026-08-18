from __future__ import annotations

from collections.abc import Sequence

from flint import fmpq, fmpq_mat

from hiprof.formula.formula import Variable


def matrix_entries(matrix: fmpq_mat) -> tuple[tuple[fmpq, ...], ...]:
    return tuple(
        tuple(matrix[i, j] for j in range(matrix.ncols()))
        for i in range(matrix.nrows())
    )


def matrix_shape(matrix: fmpq_mat) -> tuple[int, int]:
    return matrix.nrows(), matrix.ncols()


def variable_names(variables: Sequence[Variable]) -> tuple[str, ...]:
    return tuple(str(variable) for variable in variables)


def q(numerator: int, denominator: int = 1) -> fmpq:
    return fmpq(numerator, denominator)
