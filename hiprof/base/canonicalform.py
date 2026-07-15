# code modified from https://github.com/Whatisthisname/gaussian-canonical-forms/blob/master/CanonicalForms.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple
from numbers import Real
from flint import fmpq
from hiprof.base.utils import (
    as_fmpq_df,
    as_fmpq_series,
    eye_df,
    logdet_spd,
    matmul_exact,
    solve_exact,
    solve_spd,
    subtract_exact,
    to_fmpq,
    zero_df,
    zero_series,
)

# =============================
# Floating point implementation
# =============================


class CanonicalForm:
    """
    Canonical (or information) form of a multivariate Gaussian distribution / factor.

    The pdf p(X) is given by:
    p(X; J, h, g) = exp{g + h.T @ X - 0.5 * (X.T @ J @ X)}
    where
    - J is the information (precision) matrix,
    - h is the potential vector,
    - g is the normalization constant.

    The class also supports standard operations such as multiplication,
    division, marginalization, and conditioning, which correspond to
    manipulations of Gaussian factors in canonical form.
    """

    def __init__(
        self,
        J: pd.DataFrame,
        h: pd.Series,
        g: float,
    ) -> None:
        """
        Initialize a canonical form.

        Parameters
        ----------
        J : pandas.DataFrame
            Symmetric positive-definite information (precision) matrix.
            Row/column indices correspond to variable names.
        h : pandas.Series
            Information vector with index aligned to J's rows/columns.
        g : float
            Normalization constant.
        """
        self.J = J
        self.h = h
        self.g = g

    def _drop_zero_information_variables(
        self, atol: float = 1e-10
    ) -> "CanonicalForm":
        """Drop unconstrained variables with zero precision interactions and zero information vector entries."""
        keep = []

        for var in self.scope():
            row_zero = np.allclose(
                self.J.loc[var, :].to_numpy(dtype=float),
                0.0,
                atol=atol,
                rtol=0.0,
            )
            col_zero = np.allclose(
                self.J.loc[:, var].to_numpy(dtype=float),
                0.0,
                atol=atol,
                rtol=0.0,
            )
            h_zero = np.isclose(
                float(self.h.loc[var]), 0.0, atol=atol, rtol=0.0
            )

            if not (row_zero and col_zero and h_zero):
                keep.append(var)

        return CanonicalForm(
            self.J.loc[keep, keep].copy(),
            self.h.loc[keep].copy(),
            self.g,
        )

    def __mul__(
        self,
        other: CanonicalForm | float,
    ) -> CanonicalForm:
        """
        Multiply two canonical forms (factor product).

        Parameters
        ----------
        other : CanonicalForm | float
            The factor to be multiplied with this one.

        Returns
        -------
        CanonicalForm
            New canonical factor representing the product.
        """
        if isinstance(other, Real):
            if other <= 0:
                raise ValueError("Can only multiply by positive constants.")
            return CanonicalForm(
                self.J.copy(), self.h.copy(), self.g + float(np.log(other))
            )
        J = pd.DataFrame.add(self.J, other.J, fill_value=0.0).fillna(0.0)
        h = pd.Series.add(self.h, other.h, fill_value=0.0).fillna(0.0)
        g = self.g + other.g
        return CanonicalForm(J, h, g)._drop_zero_information_variables()

    def __rmul__(self, other: float) -> CanonicalForm:
        return self.__mul__(other)

    def __truediv__(
        self,
        other: CanonicalForm | float,
    ) -> CanonicalForm:
        """
        Divide two canonical forms (factor quotient).

        Parameters
        ----------
        other : CanonicalForm | float
            The factor by which this one is divided.

        Returns
        -------
        CanonicalForm
            New canonical factor representing the quotient.
        """
        if isinstance(other, Real):
            if other <= 0:
                raise ValueError("Can only divide by positive constants.")
            return CanonicalForm(
                self.J.copy(), self.h.copy(), self.g - float(np.log(other))
            )
        J = pd.DataFrame.subtract(self.J, other.J, fill_value=0.0).fillna(0.0)
        h = pd.Series.subtract(self.h, other.h, fill_value=0.0).fillna(0.0)
        g = self.g - other.g
        return CanonicalForm(J, h, g)._drop_zero_information_variables()

    def scope(self) -> List[str]:
        """
        Get the list of variable names in the factor's scope.

        Returns
        -------
        list of str
            Variable names corresponding to the indices of J.
        """
        return list(self.J.index)

    def marginalization(
        self,
        vars_to_marginalize: Tuple[str, ...],
    ) -> CanonicalForm:
        """
        Marginalize out a subset of variables.

        Parameters
        ----------
        vars_to_marginalize : tuple of str
            Variables to integrate out.

        Returns
        -------
        CanonicalForm
            New factor over the remaining variables.
        """
        scope = self.scope()
        Y = [v for v in vars_to_marginalize if v in scope]
        if not Y:
            return self

        X = [x for x in scope if x not in Y]

        J_YY = self.J.loc[Y, Y].to_numpy(dtype=float)
        h_Y = self.h[Y].to_numpy(dtype=float)

        # Solve J_YY z = h_Y, that is z = J_YY^{-1} h_Y
        solve_h_Y = solve_spd(J_YY, h_Y)

        # log det(2*pi*J_YY^{-1}) = len(Y)*log(2*pi) - log det(J_YY)
        logdet_term = len(Y) * np.log(2 * np.pi) - logdet_spd(J_YY)

        if not X:
            g = self.g + 0.5 * (logdet_term + h_Y.T @ solve_h_Y)
            J = pd.DataFrame(index=[], columns=[], dtype=float)
            h = pd.Series(dtype=float)
            return CanonicalForm(J, h, float(g))

        J_XX = self.J.loc[X, X].to_numpy(dtype=float)
        J_XY = self.J.loc[X, Y].to_numpy(dtype=float)
        J_YX = self.J.loc[Y, X].to_numpy(dtype=float)
        h_X = self.h[X].to_numpy(dtype=float)

        # Need J_XY @ J_YY^{-1} @ J_YX.
        # Solve J_YY Z = J_YX instead of forming J_YY^{-1}.
        solve_J_YX = solve_spd(J_YY, J_YX)

        J_new = J_XX - J_XY @ solve_J_YX
        h_new = h_X - J_XY @ solve_h_Y
        g_new = self.g + 0.5 * (logdet_term + h_Y.T @ solve_h_Y)

        J_new = pd.DataFrame(J_new, index=X, columns=X)
        h_new = pd.Series(h_new, index=X)

        return CanonicalForm(J_new, h_new, float(g_new))

    def reduction(
        self,
        evidence: Dict[str, float],
    ) -> CanonicalForm:
        """
        Fix some variables to given values and return
        the resulting factor over the remaining variables.

        Parameters
        ----------
        evidence : dict of str to float
            Mapping from variable names to observed values.

        Returns
        -------
        CanonicalForm
            New factor with evidence variables eliminated.
            If none of the evidence variables are in the scope, returns self.
        """
        scope = self.scope()
        if np.all([key not in scope for key in evidence.keys()]):
            return self

        X = list(filter(lambda x: x not in evidence.keys(), scope))
        Y = list(filter(lambda x: x in scope, evidence.keys()))
        y = np.array([evidence[y] for y in Y])

        J_YY = self.J.loc[Y, Y].to_numpy()
        h_Y = self.h[Y].to_numpy()

        if len(X) == 0:
            g = self.g + h_Y.T @ y - 0.5 * (y.T @ J_YY @ y)
            J = pd.DataFrame(index=[], columns=[], dtype=float)
            h = pd.Series(dtype=float)
            return CanonicalForm(J, h, float(g))

        J_XX = self.J.loc[X, X]
        J_XY = self.J.loc[X, Y].to_numpy()
        h_X = self.h[X].to_numpy()

        J = J_XX
        h = h_X - J_XY @ y
        g = self.g + h_Y.T @ y - 0.5 * (y.T @ J_YY @ y)
        h = pd.Series(h, index=X)

        return CanonicalForm(J, h, g)

    def evaluate_pdf(
        self,
        state: Dict[str, float],
    ) -> float:
        """
        Evaluate the probability density at a given state.

        Parameters
        ----------
        state : dict of str to float
            Mapping from variable names to values. The order of values in
            the dictionary should be consistent with the index order of J.

        Returns
        -------
        float
            The probability density at the given state.
        """
        x = np.array(list(state.values()))
        return np.exp(-0.5 * x.T @ self.J @ x + self.h.T @ x + self.g)

    def covariance(self) -> pd.DataFrame:
        """
        Compute the covariance matrix of the corresponding Gaussian.

        Returns
        -------
        pandas.DataFrame
            Covariance matrix with indices aligned to J.
        """
        J_np = self.J.to_numpy(dtype=float)
        I = np.eye(J_np.shape[0])

        Sigma = solve_spd(J_np, I)
        Sigma = 0.5 * (Sigma + Sigma.T)

        return pd.DataFrame(Sigma, index=self.J.index, columns=self.J.columns)

    def mean(self) -> pd.Series:
        """
        Compute the mean vector of the corresponding Gaussian.

        Returns
        -------
        pandas.Series
            Mean vector with index aligned to J and h.
        """
        J_np = self.J.to_numpy(dtype=float)
        h_np = self.h.to_numpy(dtype=float)

        mu = solve_spd(J_np, h_np)

        return pd.Series(mu, index=self.h.index)


def marginal(
    name: str,
    mean: float = 0.0,
    variance: float = 1.0,
) -> CanonicalForm:
    """
    Construct a one-dimensional Gaussian factor in canonical form.

    Parameters
    ----------
    name : str
        Name of the variable.
    mean : float, optional
        Mean of the Gaussian, by default 0.0.
    variance : float, optional
        Variance of the Gaussian, by default 1.0.

    Returns
    -------
    CanonicalForm
        Canonical form of the univariate Gaussian over `name`.
    """
    J = pd.DataFrame([1 / variance], index=[name], columns=[name])
    h = pd.Series(mean / variance, index=[name])
    g = -0.5 * mean**2 / variance - np.log(np.sqrt(2 * np.pi * variance))
    return CanonicalForm(J, h, g)


def conditional(
    name: str,
    parents: Tuple[str, ...],
    w: Tuple[float, ...],
    b: float = 0.0,
    variance: float = 1.0,
) -> CanonicalForm:
    """
    Construct a conditional linear Gaussian factor in canonical form.

    Parameters
    ----------
    name : str
        Name of the child variable.
    parents : tuple of str
        Names of the parent variables.
    w : tuple of float
        Weights corresponding to each parent.
    b : float, optional
        Intercept term, by default 0.0.
    variance : float, optional
        Noise variance, by default 1.0.

    Returns
    -------
    CanonicalForm
        Canonical form representing the joint Gaussian of the child
        and its parents implied by the conditional model.
    """
    scope = [name] + list(parents)
    w = np.asarray(w, dtype=float).reshape(-1, 1)  # (k, 1)

    J_YY = np.array([[1 / variance]])  # (1, 1)
    J_YX = (-w.T) / variance  # (1, k)
    J_XY = -w / variance  # (k, 1)
    J_XX = w @ w.T / variance  # (k, k)

    top = np.concatenate([J_YY, J_YX], axis=1)  # (1+k, k)
    bottom = np.concatenate([J_XY, J_XX], axis=1)  # (k, 1+k)
    J = np.concatenate([top, bottom], axis=0)  # (1+k, 1+k)
    J = pd.DataFrame(J, index=scope, columns=scope)

    h_Y = np.array([b / variance])  # (1,)
    h_X = (-b * w / variance).flatten()  # (k,)
    h = np.concatenate([h_Y, h_X])  # (1+k,)
    h = pd.Series(h, index=scope)

    g = -(b**2) / (2 * variance) - 0.5 * np.log(2 * np.pi * variance)

    return CanonicalForm(J, h, g)


# ===============================
# Exact arithmetic implementation
# ===============================


class ExactCanonicalForm:
    """
    Exact rational canonical form for Gaussian algebra.

    This represents only the quadratic and linear parts of a Gaussian factor:

        exp{ h.T x - 0.5 x.T J x }

    The log-normalizing constant ``g`` is intentionally omitted because the
    exact falsifier compares means and covariances, not density constants.
    """

    def __init__(
        self,
        J: pd.DataFrame,
        h: pd.Series,
    ) -> None:
        self.J = as_fmpq_df(J)
        self.h = as_fmpq_series(h)

        if list(self.J.index) != list(self.J.columns):
            raise ValueError("J must have identical row and column labels.")
        if list(self.h.index) != list(self.J.index):
            raise ValueError("h index must match J index.")

    def scope(self) -> List[str]:
        """
        Get the list of variable names in the factor's scope.
        """
        return list(self.J.index)

    def _drop_zero_information_variables(self) -> "ExactCanonicalForm":
        """Drop unconstrained variables with zero precision interactions and zero information vector entries."""
        keep = []
        for var in self.scope():
            row_zero = all(
                to_fmpq(self.J.loc[var, col]) == fmpq(0)
                for col in self.J.columns
            )
            col_zero = all(
                to_fmpq(self.J.loc[row, var]) == fmpq(0)
                for row in self.J.index
            )
            h_zero = to_fmpq(self.h.loc[var]) == fmpq(0)

            if not (row_zero and col_zero and h_zero):
                keep.append(var)

        return ExactCanonicalForm(
            self.J.loc[keep, keep],
            self.h.loc[keep],
        )

    def __mul__(self, other: "ExactCanonicalForm") -> "ExactCanonicalForm":
        """
        Multiply two exact canonical forms.
        """
        if not isinstance(other, ExactCanonicalForm):
            raise TypeError(
                "ExactCanonicalForm can only be multiplied by ExactCanonicalForm."
            )

        scope = list(dict.fromkeys(self.scope() + other.scope()))

        J = zero_df(scope, scope)
        h = zero_series(scope)

        for i in scope:
            for j in scope:
                a = (
                    self.J.loc[i, j]
                    if i in self.J.index and j in self.J.columns
                    else fmpq(0)
                )
                b = (
                    other.J.loc[i, j]
                    if i in other.J.index and j in other.J.columns
                    else fmpq(0)
                )
                J.loc[i, j] = to_fmpq(a) + to_fmpq(b)

        for i in scope:
            a = self.h.loc[i] if i in self.h.index else fmpq(0)
            b = other.h.loc[i] if i in other.h.index else fmpq(0)
            h.loc[i] = to_fmpq(a) + to_fmpq(b)

        return ExactCanonicalForm(J, h)._drop_zero_information_variables()

    def __truediv__(self, other: "ExactCanonicalForm") -> "ExactCanonicalForm":
        """
        Divide two exact canonical forms.
        """
        if not isinstance(other, ExactCanonicalForm):
            raise TypeError(
                "ExactCanonicalForm can only be divided by ExactCanonicalForm."
            )

        scope = list(dict.fromkeys(self.scope() + other.scope()))

        J = zero_df(scope, scope)
        h = zero_series(scope)

        for i in scope:
            for j in scope:
                a = (
                    self.J.loc[i, j]
                    if i in self.J.index and j in self.J.columns
                    else fmpq(0)
                )
                b = (
                    other.J.loc[i, j]
                    if i in other.J.index and j in other.J.columns
                    else fmpq(0)
                )
                J.loc[i, j] = to_fmpq(a) - to_fmpq(b)

        for i in scope:
            a = self.h.loc[i] if i in self.h.index else fmpq(0)
            b = other.h.loc[i] if i in other.h.index else fmpq(0)
            h.loc[i] = to_fmpq(a) - to_fmpq(b)

        return ExactCanonicalForm(J, h)._drop_zero_information_variables()

    def marginalization(
        self,
        vars_to_marginalize: Tuple[str, ...],
    ) -> "ExactCanonicalForm":
        """
        Marginalize out variables exactly.

        For variables split as X = kept variables and Y = marginalized variables:

            J'_XX = J_XX - J_XY J_YY^{-1} J_YX
            h'_X  = h_X  - J_XY J_YY^{-1} h_Y

        The omitted constant term is not needed for mean/covariance comparison.
        """
        scope = self.scope()
        Y = [v for v in vars_to_marginalize if v in scope]

        if not Y:
            return self

        X = [v for v in scope if v not in Y]

        if not X:
            J = pd.DataFrame(index=[], columns=[], dtype=object)
            h = pd.Series(dtype=object)
            return ExactCanonicalForm(J, h)

        J_YY = self.J.loc[Y, Y]
        J_XX = self.J.loc[X, X]
        J_XY = self.J.loc[X, Y]
        J_YX = self.J.loc[Y, X]
        h_X = self.h.loc[X]
        h_Y = self.h.loc[Y]

        solve_J_YX = solve_exact(J_YY, J_YX)
        solve_h_Y = solve_exact(J_YY, h_Y)

        J_new = subtract_exact(J_XX, matmul_exact(J_XY, solve_J_YX))
        h_new = subtract_exact(h_X, matmul_exact(J_XY, solve_h_Y))

        return ExactCanonicalForm(J_new, h_new)

    def reduction(
        self,
        evidence: Dict[str, Any],
    ) -> "ExactCanonicalForm":
        """
        Fix variables to exact values and return the remaining factor.

        For remaining variables X and fixed variables Y=y:

            J'_XX = J_XX
            h'_X  = h_X - J_XY y

        The constant term is omitted because exact verification compares means
        and covariances of the resulting Gaussian distribution.
        """
        scope = self.scope()

        if all(key not in scope for key in evidence):
            return self

        X = [v for v in scope if v not in evidence]
        Y = [v for v in scope if v in evidence]

        if not X:
            J = pd.DataFrame(index=[], columns=[], dtype=object)
            h = pd.Series(dtype=object)
            return ExactCanonicalForm(J, h)

        y = pd.Series(
            [to_fmpq(evidence[v]) for v in Y],
            index=Y,
            dtype=object,
        )

        J_XX = self.J.loc[X, X]
        J_XY = self.J.loc[X, Y]
        h_X = self.h.loc[X]

        h_new = subtract_exact(h_X, matmul_exact(J_XY, y))

        return ExactCanonicalForm(J_XX, h_new)

    def mean(self) -> pd.Series:
        """
        Compute the exact mean vector, mu = J^{-1} h.
        """
        return solve_exact(self.J, self.h)

    def covariance(self) -> pd.DataFrame:
        """
        Compute the exact covariance matrix, Sigma = J^{-1}.
        """
        I = eye_df(self.J.index)
        return solve_exact(self.J, I)


def exact_marginal(
    name: str,
    mean: Any = 0,
    variance: Any = 1,
) -> ExactCanonicalForm:
    """
    Construct an exact univariate Gaussian canonical factor.
    """
    mean = to_fmpq(mean)
    variance = to_fmpq(variance)

    if variance == 0:
        raise ZeroDivisionError("Variance cannot be zero.")

    J = pd.DataFrame(
        [[fmpq(1) / variance]],
        index=[name],
        columns=[name],
        dtype=object,
    )
    h = pd.Series([mean / variance], index=[name], dtype=object)

    return ExactCanonicalForm(J, h)


def exact_conditional(
    name: str,
    parents: Tuple[str, ...],
    w: Tuple[Any, ...],
    b: Any = 0,
    variance: Any = 1,
) -> ExactCanonicalForm:
    """
    Construct an exact linear-Gaussian conditional canonical factor.
    """
    scope = [name] + list(parents)

    w = [to_fmpq(x) for x in w]
    b = to_fmpq(b)
    variance = to_fmpq(variance)

    if len(w) != len(parents):
        raise ValueError("There must be one weight per parent.")

    if variance == 0:
        raise ZeroDivisionError("Variance cannot be zero.")

    J = zero_df(scope, scope)
    h = zero_series(scope)

    J.loc[name, name] = fmpq(1) / variance

    for parent, weight in zip(parents, w):
        J.loc[name, parent] = -weight / variance
        J.loc[parent, name] = -weight / variance

    for i, parent_i in enumerate(parents):
        for j, parent_j in enumerate(parents):
            J.loc[parent_i, parent_j] = w[i] * w[j] / variance

    h.loc[name] = b / variance

    for parent, weight in zip(parents, w):
        h.loc[parent] = -b * weight / variance

    return ExactCanonicalForm(J, h)
