"""Adapter between y0 output and hiprof formula.

This module is used to build a y0 graph, run ID,
and render the result back into hiprof's formula language.
Available when the optional ``identification'' dependencies
are installed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping, Sequence

from hiprof.formula.validation import parse_and_validate
from hiprof.graph import Graph, Node, parse_graph
from hiprof.utils import validate_variables


_dsl: Any | None = None
_identify_outcomes: Any | None = None
_mixed_graph: Any | None = None

try:
    from tqdm import TqdmWarning

    # Suppress a tqdm warning that is safe to ignore.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=TqdmWarning)

        from y0 import dsl
        from y0.algorithm.identify import identify_outcomes
        from y0.graph import NxMixedGraph
except ImportError:
    pass
else:
    _dsl = dsl
    _identify_outcomes = identify_outcomes
    _mixed_graph = NxMixedGraph


@dataclass(frozen=True)
class _RenderedExpression:
    text: str
    outputs: frozenset[str]
    inputs: frozenset[str]
    is_base_kernel: bool = False


class IDAlgorithm:
    """Identify an interventional distribution using the ID algorithm.

    This class wraps :mod:`y0` and renders the resulting identifying formulas
    into the hiprof formula language.
    """

    def __init__(
        self,
        graph: str,
        treatments: str | Sequence[str],
        outcomes: str | Sequence[str],
        latents: str | Sequence[str] | None = None,
    ) -> None:
        """Initialise the ID algorithm for a causal query.

        :param graph: Graph specification using ``->`` for directed edges and
            ``<->`` for bidirected edges.
        :param treatments: Treatment variable name, or sequence of treatment
            variable names.
        :param outcomes: Outcome variable name, or sequence of outcome
            variable names.
        :param latents: Variable name, or sequence of variable names, to treat
            as unobserved. Every latent variable must be a node in ``graph``.
        :raises ImportError: If the optional identification dependencies are
            not installed.
        :raises TypeError: If treatments, outcomes, or latents have invalid
            types.
        :raises ValueError: If the graph, treatments, outcomes, or latents are
            invalid.
        """
        if _dsl is None or _identify_outcomes is None or _mixed_graph is None:
            raise ImportError(
                "IDAlgorithm requires the optional `identification` "
                "dependencies. Install them with "
                '`pip install "hiprof[identification]"`.'
            )

        self._dsl = _dsl
        self._identify_outcomes = _identify_outcomes
        self._mixed_graph = _mixed_graph

        self.graph = parse_graph(graph)

        if latents is None:
            self.latents: tuple[str, ...] = ()
        else:
            self.latents = validate_variables(
                latents,
                name="Latents",
                graph=self.graph,
            )

        for latent in self.latents:
            self.graph.nodes[latent].observed = False

        self.treatments = validate_variables(
            treatments,
            name="Treatments",
            graph=self.graph,
        )
        self.outcomes = validate_variables(
            outcomes,
            name="Outcomes",
            graph=self.graph,
        )

        overlap = sorted(set(self.treatments) & set(self.outcomes))
        if overlap:
            raise ValueError(
                "Treatments and outcomes must be disjoint. "
                f"Overlapping variables: {', '.join(overlap)}."
            )

        self._variables = {
            name: self._dsl.Variable(name)
            for name, node in self.graph.nodes.items()
            if node.observed
        }
        self._graph = self._build_y0_graph()

    def _build_y0_graph(self) -> Any:
        directed_edges, bidirected_edges = _latent_projection(self.graph)

        graph = self._mixed_graph.from_edges(
            directed=[
                (self._variables[parent], self._variables[child])
                for parent, child in sorted(directed_edges)
            ],
            undirected=[
                (self._variables[left], self._variables[right])
                for left, right in sorted(bidirected_edges)
            ],
        )

        for variable in self._variables.values():
            graph.add_node(variable)

        return graph

    def run(self) -> str | None:
        """Run identification and render the result as a hiprof formula.

        :returns: A formula string compatible with the hiprof grammar
            if the query is identifiable, otherwise ``None``.
        :raises ValueError: If the identified formula cannot be rendered into
            the hiprof grammar.
        :raises NotImplementedError: If the identified formula uses a
            fraction form that hiprof cannot represent.
        """
        treatments = {self._variables[name] for name in self.treatments}
        outcomes = {self._variables[name] for name in self.outcomes}

        formula = self._identify_outcomes(
            self._graph,
            treatments=treatments,
            outcomes=outcomes,
        )

        if formula is None:
            return None

        rendered = _Y0Renderer(self.treatments, dsl=self._dsl).render(formula)
        parse_and_validate(rendered)
        return rendered


def _latent_projection(
    graph: Graph,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    observed_nodes = tuple(
        node for node in graph.nodes.values() if node.observed
    )
    latent_nodes = tuple(
        node for node in graph.nodes.values() if not node.observed
    )

    directed_edges = {
        (source.name, target)
        for source in observed_nodes
        for target in _first_observed_descendants(source)
    }

    bidirected_edges: set[tuple[str, str]] = set()
    for latent in latent_nodes:
        descendants = sorted(_first_observed_descendants(latent))
        bidirected_edges.update(combinations(descendants, 2))

    return directed_edges, bidirected_edges


def _first_observed_descendants(node: Node) -> set[str]:
    observed_descendants: set[str] = set()
    visited: set[Node] = set()
    pending = list(node.children)

    while pending:
        descendant = pending.pop()
        if descendant in visited:
            continue
        visited.add(descendant)

        if descendant.observed:
            observed_descendants.add(descendant.name)
        else:
            pending.extend(descendant.children)

    return observed_descendants


class _Y0Renderer:
    def __init__(
        self,
        treatments: Sequence[str],
        dsl: Any,
    ) -> None:
        self.treatments = frozenset(treatments)
        self._dsl = dsl

    def render(self, expression: Any) -> str:
        return self._render(expression, bound_copies={}).text

    def _render(
        self,
        expression: Any,
        bound_copies: Mapping[str, int],
    ) -> _RenderedExpression:
        if isinstance(expression, self._dsl.Probability):
            return self._probability(expression, bound_copies)

        if isinstance(expression, self._dsl.Product):
            return self._product(expression, bound_copies)

        if isinstance(expression, self._dsl.Sum):
            return self._sum(expression, bound_copies)

        if isinstance(expression, self._dsl.Fraction):
            return self._fraction(expression, bound_copies)

        raise TypeError(
            "Unsupported y0 expression node: " f"{type(expression).__name__}."
        )

    def _probability(
        self,
        probability: Any,
        bound_copies: Mapping[str, int],
    ) -> _RenderedExpression:
        outputs = frozenset(
            self._format_variable(variable, bound_copies)
            for variable in probability.distribution.children
        )
        inputs = frozenset(
            self._format_variable(variable, bound_copies)
            for variable in probability.distribution.parents
        )

        output_text = _format_names(outputs)
        if inputs:
            text = f"p({output_text} | {_format_names(inputs)})"
        else:
            text = f"p({output_text})"

        return _RenderedExpression(
            text=text,
            outputs=outputs,
            inputs=inputs,
            is_base_kernel=True,
        )

    def _product(
        self,
        product: Any,
        bound_copies: Mapping[str, int],
    ) -> _RenderedExpression:
        factors = tuple(
            self._render(factor, bound_copies)
            for factor in product.expressions
        )

        outputs: set[str] = set()
        for factor in factors:
            overlap = outputs & factor.outputs
            if overlap:
                raise ValueError(
                    "The y0 product repeats kernel outputs: "
                    f"{_format_names(overlap)}."
                )
            outputs.update(factor.outputs)

        all_inputs = set().union(*(factor.inputs for factor in factors))

        return _RenderedExpression(
            text=" * ".join(factor.text for factor in factors),
            outputs=frozenset(outputs),
            inputs=frozenset(all_inputs - outputs),
        )

    def _sum(
        self,
        summation: Any,
        bound_copies: Mapping[str, int],
    ) -> _RenderedExpression:
        ranges = tuple(sorted(summation.ranges, key=lambda v: str(v.name)))
        new_bound_copies = dict(bound_copies)

        for variable in ranges:
            name = str(variable.name)
            if name in self.treatments:
                new_bound_copies[name] = new_bound_copies.get(name, -1) + 1

        rendered_ranges = frozenset(
            self._format_variable(variable, new_bound_copies)
            for variable in ranges
        )
        body = self._render(summation.expression, new_bound_copies)

        missing = rendered_ranges - body.outputs
        if missing:
            raise ValueError(
                "The y0 expression marginalises variables that are not "
                f"kernel outputs: {_format_names(missing)}."
            )

        return _RenderedExpression(
            text=(
                f"sum_{{{_format_names(rendered_ranges)}}}"
                f"{{ {body.text} }}"
            ),
            outputs=body.outputs - rendered_ranges,
            inputs=body.inputs,
        )

    def _fraction(
        self,
        fraction: Any,
        bound_copies: Mapping[str, int],
    ) -> _RenderedExpression:
        numerator = self._render(fraction.numerator, bound_copies)
        denominator = self._render(fraction.denominator, bound_copies)

        outputs, inputs = _icd_signature(numerator, denominator)

        if numerator.is_base_kernel and denominator.is_base_kernel:
            return _RenderedExpression(
                text=f"({numerator.text} / {denominator.text})",
                outputs=outputs,
                inputs=inputs,
            )

        if not (
            isinstance(fraction.denominator, self._dsl.Sum)
            and fraction.denominator.expression == fraction.numerator
        ):
            raise NotImplementedError(
                "The hiprof grammar cannot represent this arbitrary y0 "
                "fraction. Expected either a quotient of two base kernels "
                "or a numerator divided by one of its marginalisations."
            )

        denominator_outputs = _format_names(denominator.outputs)
        denominator_inputs = _format_names(denominator.inputs)

        return _RenderedExpression(
            text=(
                f"icd_{{{denominator_outputs}|{denominator_inputs}}}"
                f"{{ {numerator.text} }}"
            ),
            outputs=outputs,
            inputs=inputs,
        )

    @staticmethod
    def _format_variable(
        variable: Any,
        bound_copies: Mapping[str, int],
    ) -> str:
        name = str(variable.name)
        copy_index = bound_copies.get(name)

        if copy_index is None:
            return name

        suffix = "" if copy_index == 0 else str(copy_index)
        return f"{name}'{suffix}"


def _icd_signature(
    numerator: _RenderedExpression,
    denominator: _RenderedExpression,
) -> tuple[frozenset[str], frozenset[str]]:
    if not denominator.outputs:
        raise ValueError("An ICD denominator must have at least one output.")

    missing_outputs = denominator.outputs - numerator.outputs
    if missing_outputs:
        raise ValueError(
            "Fraction denominator outputs are not numerator outputs: "
            f"{_format_names(missing_outputs)}."
        )

    omitted_inputs = numerator.inputs - denominator.inputs
    if omitted_inputs:
        raise ValueError(
            "Fraction denominator omits numerator inputs: "
            f"{_format_names(omitted_inputs)}."
        )

    remaining_outputs = numerator.outputs - denominator.outputs
    invalid_inputs = denominator.inputs - (
        numerator.inputs | remaining_outputs
    )
    if invalid_inputs:
        raise ValueError(
            "Fraction denominator has invalid inputs: "
            f"{_format_names(invalid_inputs)}."
        )

    return (
        remaining_outputs,
        numerator.inputs | denominator.outputs,
    )


def _format_names(names: Sequence[str] | set[str] | frozenset[str]) -> str:
    return ", ".join(sorted(names))
