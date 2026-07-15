import warnings
from tqdm import TqdmWarning

# suppress tqdm warnings
warnings.filterwarnings("ignore", category=TqdmWarning)
from typing import Tuple, Set, List
from y0.graph import NxMixedGraph
from y0.algorithm.identify import identify_outcomes
from y0.mutate import canonicalize
from y0.algorithm.separation import are_d_separated
from y0.dsl import (
    Expression,
    Variable,
    Distribution,
    Probability,
    Sum,
    Product,
    Fraction,
)
from hiprof.base.graph import Graph
from hiprof.base.basetask import BaseCausalTask


class ID(BaseCausalTask):
    """
    ID algorithm as proposed in
    Ilya Shpitser and Judea Pearl. "Identification of Joint Interventional
    Distributions in Recursive Semi-Markovian Causal Models". AAAI, 2006.

    Given a DAG or an ADMG and a query (treatments-outcomes),
    outputs an identifying formula or None if the effect is not identifiable.

    Notes
    -----
    - This class is a wrapper of the y0 implementation
    """

    def __init__(
        self,
        treatments: Tuple[str | int, ...] | List[str | int] | str | int,
        outcomes: Tuple[str | int, ...] | List[str | int] | str | int,
        graph: Graph,
    ) -> None:
        """
        Initialize the ID class.

        Parameters
        ----------
        treatments : Tuple[str | int, ...] | List[str | int]
            Tuple or list of treatments. It could also be a single treatment.
            Each element must correspond to a node in the graph.
        outcomes : list, tuple, str or int
            Tuple or list of outcomes. It could also be a single outcome.
            Each element must correspond to a node in the graph.
        graph : Graph
            The DAG or ADMG.
        """
        super().__init__(treatments, outcomes, graph)

        # adapt to y0 input
        self._vars = {n: Variable(n) for n in self.graph.observed_nodes}
        self._graph = self._build_graph()

    def _build_graph(self) -> NxMixedGraph:
        """
        Construct the graph to pass to y0.

        Returns
        -------
        NxMixedGraph
            Graph used for the ID algorithm.
        """
        di_edges = [
            (self._vars[src], self._vars[dst])
            for src in self.graph.directed.index
            for dst in self.graph.directed.columns
            if self.graph.directed.loc[src, dst] != 0
        ]

        bi_edges = []
        if self.graph.is_admg:
            for i, src in enumerate(self.graph.bidirected.index):
                for dst in self.graph.bidirected.columns[i + 1 :]:
                    if self.graph.bidirected.loc[src, dst] != 0:
                        bi_edges.append((self._vars[src], self._vars[dst]))

        return NxMixedGraph.from_edges(directed=di_edges, undirected=bi_edges)

    @staticmethod
    def _drop_ancestors_by_dsep(
        p: Probability,
        graph: NxMixedGraph,
    ) -> Probability:
        """
        Take a y0 `Probability` term representing a kernel of the form
        `P(children | ancestors)` and simplify it using d-separation.

        For each candidate conditioning variable `z` in `ancestors`, test whether
        every child variable in `children` is d-separated from `z` given the remaining
        conditioning variables. If so, `z` is removed from the conditioning set.

        Parameters
        ----------
        p : Probability
            Kernel of the form `P(children | ancestors)`
        graph : NxMixedGraph
            Graph used by `are_d_separated` to test d-separations.

        Returns
        -------
        Probability
            Either the original `p` (if no conditioning variables can be removed),
            or a new `Probability` with the same children and a reduced set of ancestors.
        """
        children = p.distribution.children
        ancestors = list(p.distribution.parents)

        changed = True
        while changed:
            changed = False
            for z in list(ancestors):
                remaining = [q for q in ancestors if q != z]
                # drop z if every child is d-separated from z given remaining
                if all(
                    are_d_separated(
                        graph, c, z, conditions=remaining
                    ).separated
                    for c in children
                ):
                    ancestors.remove(z)
                    changed = True

        if (
            tuple(ancestors) == p.distribution.parents
        ):  # if nothing has changed
            return p
        return Probability.safe(
            Distribution(children=children, parents=tuple(ancestors))
        )

    def _drop_unused_conditioning_ancestors(
        self,
        expr: Expression | Sum | Product | Fraction | Probability,
        graph: NxMixedGraph,
    ):
        """
        Walk a y0 expression tree (sums/products/fractions/probability kernels)
        and simplify every `Probability` term it finds.

        Specifically, for each `Probability` node, it calls `_drop_ancestors_by_dsep`
        to remove any conditioning variables (parents) that are d-separated from the
        children given the remaining conditioning set.
        As an example, `P(X | Z, W) -> P(X | W)` when `X d-sep Z | W`.

        Parameters
        ----------
        expr : Expression | Sum | Product | Fraction | Probability
            A y0 expression to be simplified.
        graph : NxMixedGraph
            The causal graph used for d-separation tests.

        Returns
        -------
        Expression
            An expression structurally equivalent to `expr` but with potentially
            smaller conditioning sets inside its probability kernels.

        Notes
        -----
        - This performs conditional-independence (CI) simplification, not general algebraic
          simplification. It is often useful to run ``canonicalize`` before/after this pass.
        - The rewrite rule applied inside kernels is conceptually:
          ``P(X | Z, W) -> P(X | W)`` when ``X ⟂ Z | W`` (as determined by d-separation).
        """
        if isinstance(expr, Probability):
            return self._drop_ancestors_by_dsep(expr, graph)

        if isinstance(expr, Sum):
            return Sum.safe(
                self._drop_unused_conditioning_ancestors(
                    expr.expression, graph
                ),
                expr.ranges,
                simplify=True,
            )

        if isinstance(expr, Product):
            return Product.safe(
                self._drop_unused_conditioning_ancestors(e, graph)
                for e in expr.expressions
            )

        if isinstance(expr, Fraction):
            return Fraction(
                self._drop_unused_conditioning_ancestors(
                    expr.numerator, graph
                ),
                self._drop_unused_conditioning_ancestors(
                    expr.denominator, graph
                ),
            ).simplify()

        return expr

    def _to_hiprof(
        self,
        expr: Expression | Sum | Product | Fraction | Probability,
        bound: Set[str | int] | None = None,
    ) -> str:
        """
        Modify the given expression to produce a string
        compatible with the grammar given by this package.

        Parameters
        ----------
        expr: Expression | Sum | Product | Fraction | Probability
            Expression following the y0 implementation.
        bound: set, optional
            Variables that are bound by summation.
            Free treatment variables are printed as placeholders, e.g. ``X=.``.
            Bound treatment variables are printed as primed dummy variables,
            e.g. ``SUM_{X'} { p(X') }``.

        Returns
        -------
        str
            The modified expression.
        """
        bound = set(bound or set())
        treatments = set(self.treatments)

        def _fmt_name(name: str | int) -> str:
            if name in treatments:
                if name in bound:
                    return f"{name}'"
                return f"{name}=."
            return str(name)

        def _fmt_sum_name(name: str | int) -> str:
            if name in treatments:
                return f"{name}'"
            return str(name)

        def _fmt_var(v: Variable) -> str:
            return _fmt_name(v.name)

        def _fmt_vars(vs: tuple[Variable, ...]) -> str:
            return ", ".join(_fmt_var(v) for v in vs)

        # Probability: p(children | parents)
        if isinstance(expr, Probability):
            dist = expr.distribution
            ch = _fmt_vars(dist.children)
            if dist.parents:
                pa = _fmt_vars(dist.parents)
                return f"p({ch} | {pa})"
            return f"p({ch})"

        # Product
        if isinstance(expr, Product):
            return " * ".join(
                self._to_hiprof(p, bound=bound) for p in expr.expressions
            )

        # Sum: binds its range variables
        if isinstance(expr, Sum):
            rs = ", ".join(_fmt_sum_name(v.name) for v in expr.ranges)
            new_bound = bound | {v.name for v in expr.ranges}
            body = self._to_hiprof(
                expr.expression,
                bound=new_bound,
            )
            return f"SUM_{{{rs}}} {{ {body} }}"

        # Fraction
        if isinstance(expr, Fraction):
            num = self._to_hiprof(expr.numerator, bound=bound)
            den = self._to_hiprof(expr.denominator, bound=bound)
            return f"({num}) / ({den})"

        # Fallback
        return str(expr)

    def run(
        self,
        hiprof_printing: bool = True,
    ) -> str | None:
        """
        Run the ID algorithm.

        Parameters
        ----------
        hiprof_printing : bool
            Whether to modify the final string according the grammar in hiprof.
        Returns
        -------
        str | None
            If the interventional distribution is not identifiable, return None,
            otherwise a string representing the formula.
        """
        treatments = [self._vars[treat] for treat in self.treatments]
        outcomes = [self._vars[out] for out in self.outcomes]

        formula = identify_outcomes(
            self._graph,
            treatments=treatments,
            outcomes=outcomes,
        )

        if formula is None:
            return formula

        if hiprof_printing:
            formula = canonicalize(
                self._drop_unused_conditioning_ancestors(
                    canonicalize(formula), self._graph
                )
            )
            return self._to_hiprof(formula)

        return str(formula)
