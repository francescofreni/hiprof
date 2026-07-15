from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Real
from typing import Any, Tuple

from hiprof.formula.formula import Distribution, Division, Integral, Product
from hiprof.base.canonicalform import CanonicalForm, ExactCanonicalForm


class FormulaVisitor(ABC):
    """
    Abstract base class for visitors over Formula nodes.
    """

    @abstractmethod
    def visit_distribution(self, node: Distribution) -> Any:
        """
        Visit a Distribution node.
        """
        ...

    @abstractmethod
    def visit_product(self, node: Product) -> Any:
        """
        Visit a Product node.
        """
        ...

    @abstractmethod
    def visit_division(self, node: Division) -> Any:
        """
        Visit a Division node.
        """
        ...

    @abstractmethod
    def visit_integral(self, node: Integral) -> Any:
        """
        Visit an Integral node.
        """
        ...


class CanonicalFormVisitor(FormulaVisitor):
    """
    Visitor that interprets the AST in terms of operations on a joint
    Gaussian distribution in canonical form.
    """

    def __init__(self, joint: CanonicalForm | ExactCanonicalForm) -> None:
        """
        Initialize the CanonicalFormVisitor.

        Parameters
        ----------
        joint : CanonicalForm | ExactCanonicalForm
            A canonical form representing the full joint distribution.
        """
        self.joint = joint
        self.var_names = tuple(self.joint.J.index)

    def _marginal_from_joint(
        self, keep: Tuple[str, ...]
    ) -> CanonicalForm | ExactCanonicalForm:
        """
        Compute the marginal over the variables in `keep`.

        Parameters
        ----------
        keep : Tuple[str, ...]
            Tuple of variable names to keep in the resulting marginal.

        Returns
        -------
        CanonicalForm | ExactCanonicalForm
            The marginal canonical form over the specified scope.
        """
        keep_set = set(keep)
        to_marginalize = tuple(v for v in self.var_names if v not in keep_set)
        if not to_marginalize:
            return self.joint
        return self.joint.marginalization(to_marginalize)

    def visit_distribution(self, node: Distribution):
        """
        Evaluate a Distribution node to a canonical form.
        """
        targets = tuple(node.targets)
        given = tuple(node.given)

        target_names = tuple(name for name, _ in targets)
        given_names = tuple(name for name, _ in given)
        scope = target_names + given_names

        missing = [v for v in scope if v not in self.var_names]
        if missing:
            raise KeyError(
                f"Variables {missing} not found in joint scope {self.var_names}."
            )

        joint_marg = self._marginal_from_joint(scope)

        if not given:
            result = joint_marg
        else:
            given_marg = joint_marg.marginalization(target_names)
            result = joint_marg / given_marg

        evidence = {}
        for name, value in targets + given:
            if value is not None and not isinstance(value, Real):
                raise TypeError("Conditioning values must be numeric or None.")
            if value is not None:
                evidence[name] = value

        if evidence:
            result = result.reduction(evidence)

        return result

    def visit_product(self, node: Product):
        """
        Evaluate a product of formulas as a product of canonical forms.
        """
        result = node.terms[0].accept(self)
        for term in node.terms[1:]:
            result = result * term.accept(self)
        return result

    def visit_division(self, node: Division):
        """
        Evaluate a division of formulas as a quotient of canonical forms.
        """
        num = node.numerator.accept(self)
        den = node.denominator.accept(self)
        return num / den

    def visit_integral(self, node: Integral):
        """
        Evaluate an integral as marginalization over specified variables.
        """
        factor = node.integrand.accept(self)

        over = tuple(v for v in node.over if v in factor.scope())
        if not over:
            return factor

        return factor.marginalization(over)
