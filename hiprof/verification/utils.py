from typing import Sequence

from flint import fmpq_mat


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
