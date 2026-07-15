from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, Any, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from hiprof.formula.visitors import FormulaVisitor


class Formula(ABC):
    """
    Abstract base class for formulas, which is the root of the Abstract Syntax Tree (AST).
    """

    @abstractmethod
    def accept(self, visitor: FormulaVisitor) -> Any:
        """
        Dispatch this node to a visitor.

        Parameters
        ----------
        visitor : FormulaVisitor
            A FormulaVisitor instance that implements a
            visit method for this formula type.

        Returns
        -------
        Any
            The result of the visitor's method. The actual type
            depends on the particular visitor.
        """
        pass

    def __mul__(self, other: Formula) -> Formula:
        """
        Multiply two formulas (formula product).

        Parameters
        ----------
        other : Formula
            The formula to be multiplied with this one.

        Returns
        -------
        Formula
            New formula representing the product.
        """
        return Product.from_two(self, other)

    def __truediv__(self, other: Formula) -> Formula:
        """
        Divide two formulas (formula division).

        Parameters
        ----------
        other : Formula
            The formula that this one is divided by.

        Returns
        -------
        Formula
            New formula representing the division.
        """
        return Division(numerator=self, denominator=other)


class Distribution(Formula):
    """
    Represents a (possibly conditional) probability distribution.
    """

    def __init__(
        self,
        targets: Tuple[str, ...],
        given: Tuple[str, ...],
    ) -> None:
        """
        Initialize the distribution class.

        Parameters
        ----------
        targets : Tuple[str, ...]
            Tuple of variable names.
        given : Tuple[str, ...]
            Tuple of variables we condition on.
            Empty for a marginal distribution.
        """
        self.targets = targets
        self.given = given

    def accept(self, visitor: FormulaVisitor) -> Any:
        """
        Dispatch this node to a visitor.visit_distribution.
        """
        return visitor.visit_distribution(self)


class Product(Formula):
    """
    Represents the product of multiple formulas.
    """

    def __init__(self, terms: Tuple[Formula, ...]):
        """
        Initialize the product class.

        Parameters
        ----------
        terms : Tuple[Formula, ...]
            A tuple of Formula instances to be multiplied together.
        """
        self.terms = terms

    def accept(self, visitor: FormulaVisitor) -> Any:
        """
        Dispatch this node to a visitor.visit_product.
        """
        return visitor.visit_product(self)

    @staticmethod
    def from_two(a: Formula, b: Formula) -> Product:
        """
        Build the product from two formulas.

        If either 'a' or 'b' is already a Product, its terms are
        flattened so that nested products do not create nested Product
        nodes in the AST.

        Parameters
        ----------
        a, b : Formula, Formula
            Two formulas to combine.

        Returns
        -------
        Product
            A Product with all terms.
        """

        def to_terms(x: Formula) -> Iterable[Formula]:
            if isinstance(x, Product):
                return x.terms
            return (x,)

        terms = tuple(list(to_terms(a)) + list(to_terms(b)))
        return Product(terms)


class Division(Formula):
    """
    Represents the division of two formulas.
    """

    def __init__(
        self,
        numerator: Formula,
        denominator: Formula,
    ) -> None:
        """
        Initialize the division class.

        Parameters
        ----------
        numerator, denominator : Formula, Formula
            Two formulas to divide.
        """
        self.numerator = numerator
        self.denominator = denominator

    def accept(self, visitor: FormulaVisitor) -> Any:
        """
        Dispatch this node to a visitor.visit_division.
        """
        return visitor.visit_division(self)


class Integral(Formula):
    """
    Represents an integral over one or more variables.
    """

    def __init__(
        self,
        integrand: Formula,
        over: Tuple[str, ...],
    ) -> None:
        """
        Initialize the integral class.

        Parameters
        ----------
        integrand : Formula
            The formula being integrated.
        over : Tuple[str, ...]
            Tuple of variable names to integrate out.
        """
        self.integrand = integrand
        self.over = over

    def accept(self, visitor: FormulaVisitor) -> Any:
        """
        Dispatch this node to a visitor.visit_integral.
        """
        return visitor.visit_integral(self)


def p(
    targets: Tuple[str, ...],
    given: Tuple[str, ...] = (),
) -> Distribution:
    """
    Constructor for Distribution.

    Parameters
    ----------
    targets : Tuple[str, ...]
        Tuple of variables names.
    given : Tuple[str, ...]
        Tuple of variables we condition on. Empty for a marginal distribution.

    Returns
    -------
    Distribution
        A node representing the distribution.
    """
    return Distribution(targets=tuple(targets), given=tuple(given))


def integrate(
    integrand: Formula,
    over: Tuple[str, ...],
) -> Integral:
    """
    Constructor for Integral.

    Parameters
    ----------
    integrand : Formula
        The formula to be integrated.
    over : Tuple[str, ...]
        Variables of integration.

    Returns
    -------
    Integral
        An integral node representing the operation.
    """
    return Integral(integrand=integrand, over=tuple(over))
