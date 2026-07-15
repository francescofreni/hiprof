import numpy as np
import pandas as pd
from fractions import Fraction
from flint import fmpq, fmpq_mat
from typing import Any

# =============================
# Floating point linear algebra
# =============================


def solve_spd(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Solve A X = B for symmetric positive-definite A without forming inv(A).
    """
    L = np.linalg.cholesky(A)
    y = np.linalg.solve(L, B)
    return np.linalg.solve(L.T, y)


def logdet_spd(A: np.ndarray) -> float:
    """
    Compute log(det(A)) for symmetric positive-definite A.
    """
    L = np.linalg.cholesky(A)
    return float(2.0 * np.sum(np.log(np.diag(L))))


# =============================
# Exact rational helper routines
# =============================


def to_fmpq(x: Any) -> fmpq:
    """
    Convert a Python value to an exact ``flint.fmpq`` rational.
    Floats are converted through their shortest decimal representation.
    """
    if isinstance(x, fmpq):
        return x

    if isinstance(x, bool):
        raise TypeError("Boolean values are not valid rational numbers here.")

    if isinstance(x, int):
        return fmpq(x)

    if isinstance(x, Fraction):
        return fmpq(x.numerator, x.denominator)

    if isinstance(x, float):
        if not np.isfinite(x):
            raise ValueError("Cannot convert NaN or infinity to fmpq.")
        frac = Fraction(str(x))
        return fmpq(frac.numerator, frac.denominator)

    if isinstance(x, str):
        frac = Fraction(x.strip())
        return fmpq(frac.numerator, frac.denominator)

    if isinstance(x, np.integer):
        return fmpq(int(x))

    if isinstance(x, np.floating):
        xf = float(x)
        if not np.isfinite(xf):
            raise ValueError("Cannot convert NaN or infinity to fmpq.")
        frac = Fraction(str(xf))
        return fmpq(frac.numerator, frac.denominator)

    raise TypeError(f"Cannot convert value of type {type(x)!r} to fmpq.")


def zero_df(index, columns) -> pd.DataFrame:
    """
    Create a pandas DataFrame filled with exact rational zeros.
    """
    index = list(index)
    columns = list(columns)
    return pd.DataFrame(
        [[fmpq(0) for _ in columns] for _ in index],
        index=index,
        columns=columns,
        dtype=object,
    )


def zero_series(index) -> pd.Series:
    """
    Create a pandas Series filled with exact rational zeros.
    """
    index = list(index)
    return pd.Series([fmpq(0) for _ in index], index=index, dtype=object)


def eye_df(index) -> pd.DataFrame:
    """
    Create an exact rational identity matrix as a pandas DataFrame.
    """
    index = list(index)
    out = zero_df(index, index)
    for v in index:
        out.loc[v, v] = fmpq(1)
    return out


def as_fmpq_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert every entry of a DataFrame to ``fmpq``.
    """
    return pd.DataFrame(
        [[to_fmpq(df.loc[i, j]) for j in df.columns] for i in df.index],
        index=list(df.index),
        columns=list(df.columns),
        dtype=object,
    )


def as_fmpq_series(s: pd.Series) -> pd.Series:
    """
    Convert every entry of a Series to ``fmpq``.
    """
    return pd.Series(
        [to_fmpq(s.loc[i]) for i in s.index],
        index=list(s.index),
        dtype=object,
    )


def df_to_fmpq_mat(df: pd.DataFrame) -> fmpq_mat:
    """
    Convert a pandas DataFrame to a FLINT rational matrix.
    """
    rows = list(df.index)
    cols = list(df.columns)
    entries = [to_fmpq(df.loc[i, j]) for i in rows for j in cols]
    return fmpq_mat(len(rows), len(cols), entries)


def series_to_fmpq_mat(s: pd.Series) -> fmpq_mat:
    """
    Convert a pandas Series to a FLINT rational column matrix.
    """
    index = list(s.index)
    entries = [to_fmpq(s.loc[i]) for i in index]
    return fmpq_mat(len(index), 1, entries)


def fmpq_mat_to_df(
    M: fmpq_mat,
    index,
    columns,
) -> pd.DataFrame:
    """
    Convert a FLINT rational matrix to a pandas DataFrame.
    """
    index = list(index)
    columns = list(columns)

    if M.nrows() != len(index) or M.ncols() != len(columns):
        raise ValueError("Matrix shape does not match the supplied labels.")

    values = M.tolist()
    return pd.DataFrame(values, index=index, columns=columns, dtype=object)


def fmpq_mat_to_series(
    M: fmpq_mat,
    index,
) -> pd.Series:
    """
    Convert a single-column FLINT rational matrix to a pandas Series.
    """
    index = list(index)

    if M.ncols() != 1:
        raise ValueError(
            "Can only convert a single-column fmpq_mat to a Series."
        )
    if M.nrows() != len(index):
        raise ValueError("Matrix shape does not match the supplied index.")

    values = M.tolist()
    return pd.Series([row[0] for row in values], index=index, dtype=object)


def solve_exact(
    A: pd.DataFrame,
    B: pd.DataFrame | pd.Series,
) -> pd.DataFrame | pd.Series:
    """
    Solve A X = B exactly over rationals using ``flint.fmpq_mat.solve``.

    Parameters
    ----------
    A : pandas.DataFrame
        Square matrix with identical row and column labels.
    B : pandas.DataFrame or pandas.Series
        Right-hand side. Its index must match `A.index`.

    Returns
    -------
    pandas.DataFrame or pandas.Series
        Exact rational solution with the same type as `B`.

    Raises
    ------
    ZeroDivisionError
        If ``A`` is singular.
    ValueError
        If dimensions or labels are incompatible.
    """
    rows = list(A.index)
    cols = list(A.columns)

    if rows != cols:
        raise ValueError("A must be square with identical row/column labels.")

    if isinstance(B, pd.Series):
        if list(B.index) != rows:
            raise ValueError("B must have the same index as A.")
        B_mat = series_to_fmpq_mat(B)
        return_series = True
    else:
        if list(B.index) != rows:
            raise ValueError("B must have the same row index as A.")
        B_mat = df_to_fmpq_mat(B)
        return_series = False
        rhs_columns = list(B.columns)

    A_mat = df_to_fmpq_mat(A)

    X_mat = A_mat.solve(B_mat)

    if return_series:
        return fmpq_mat_to_series(X_mat, index=cols)

    return fmpq_mat_to_df(X_mat, index=cols, columns=rhs_columns)


def matmul_exact(
    A: pd.DataFrame,
    B: pd.DataFrame | pd.Series,
) -> pd.DataFrame | pd.Series:
    A_mat = df_to_fmpq_mat(A)

    if isinstance(B, pd.Series):
        if list(B.index) != list(A.columns):
            raise ValueError("B index must match A columns.")
        B_mat = series_to_fmpq_mat(B)
        C_mat = A_mat * B_mat
        return fmpq_mat_to_series(C_mat, index=A.index)

    if list(B.index) != list(A.columns):
        raise ValueError("B index must match A columns.")

    B_mat = df_to_fmpq_mat(B)
    C_mat = A_mat * B_mat
    return fmpq_mat_to_df(C_mat, index=A.index, columns=B.columns)


def subtract_exact(
    A: pd.DataFrame | pd.Series,
    B: pd.DataFrame | pd.Series,
) -> pd.DataFrame | pd.Series:
    if isinstance(A, pd.Series) and isinstance(B, pd.Series):
        if list(A.index) != list(B.index):
            raise ValueError("Series indices must match.")
        return pd.Series(
            [to_fmpq(A.loc[i]) - to_fmpq(B.loc[i]) for i in A.index],
            index=list(A.index),
            dtype=object,
        )

    if isinstance(A, pd.DataFrame) and isinstance(B, pd.DataFrame):
        if list(A.index) != list(B.index) or list(A.columns) != list(
            B.columns
        ):
            raise ValueError("DataFrame labels must match.")
        return pd.DataFrame(
            [
                [
                    to_fmpq(A.loc[i, j]) - to_fmpq(B.loc[i, j])
                    for j in A.columns
                ]
                for i in A.index
            ],
            index=list(A.index),
            columns=list(A.columns),
            dtype=object,
        )

    raise TypeError("subtract_exact expects two Series or two DataFrames.")
