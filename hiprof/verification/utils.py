from typing import Sequence

from flint import fmpq_mat

from ..formula.formula import Variable


def submatrix(
    matrix: fmpq_mat,
    rows: Sequence[int],
    columns: Sequence[int],
) -> fmpq_mat:
    return fmpq_mat(
        len(rows),
        len(columns),
        [matrix[i, j] for i in rows for j in columns],
    )


def align_columns(
    matrix: fmpq_mat,
    old_variables: tuple[Variable, ...],
    new_variables: tuple[Variable, ...],
) -> fmpq_mat:
    old_index = {
        variable: index for index, variable in enumerate(old_variables)
    }

    return fmpq_mat(
        matrix.nrows(),
        len(new_variables),
        [
            (matrix[i, old_index[variable]] if variable in old_index else 0)
            for i in range(matrix.nrows())
            for variable in new_variables
        ],
    )
