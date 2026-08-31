from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import pytest

import hiprof.identification as identification
from hiprof.formula import parse_and_validate


@dataclass(frozen=True)
class FakeVariable:
    name: str


@dataclass(frozen=True)
class FakeDistribution:
    children: tuple[FakeVariable, ...]
    parents: tuple[FakeVariable, ...] = ()


@dataclass(frozen=True)
class FakeProbability:
    distribution: FakeDistribution


@dataclass(frozen=True)
class FakeProduct:
    expressions: tuple[Any, ...]


@dataclass(frozen=True)
class FakeSum:
    expression: Any
    ranges: tuple[FakeVariable, ...]


@dataclass(frozen=True)
class FakeFraction:
    numerator: Any
    denominator: Any


FAKE_DSL = SimpleNamespace(
    Variable=FakeVariable,
    Probability=FakeProbability,
    Product=FakeProduct,
    Sum=FakeSum,
    Fraction=FakeFraction,
)


class FakeMixedGraph:
    def __init__(
        self,
        directed: list[tuple[FakeVariable, FakeVariable]],
        undirected: list[tuple[FakeVariable, FakeVariable]],
    ) -> None:
        self.directed = tuple(directed)
        self.undirected = tuple(undirected)
        self.nodes: set[FakeVariable] = set()

    @classmethod
    def from_edges(
        cls,
        directed: list[tuple[FakeVariable, FakeVariable]],
        undirected: list[tuple[FakeVariable, FakeVariable]],
    ) -> FakeMixedGraph:
        return cls(directed, undirected)

    def add_node(self, variable: FakeVariable) -> None:
        self.nodes.add(variable)


def default_identify(*args: Any, **kwargs: Any) -> None:
    return None


def install_fake_y0(
    monkeypatch: pytest.MonkeyPatch,
    identify: Callable[..., Any] | None = None,
) -> None:
    monkeypatch.setattr(identification, "_dsl", FAKE_DSL)
    monkeypatch.setattr(
        identification, "_identify_outcomes", identify or default_identify
    )
    monkeypatch.setattr(identification, "_mixed_graph", FakeMixedGraph)


def probability(
    outputs: tuple[FakeVariable, ...],
    inputs: tuple[FakeVariable, ...] = (),
) -> FakeProbability:
    return FakeProbability(FakeDistribution(outputs, inputs))


def test_id_algorithm_projects_latent_dag_and_keeps_isolated_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_y0(monkeypatch)

    algorithm = identification.IDAlgorithm(
        graph="X -> U; U -> V; V -> Y; U -> Z; W",
        treatments="X",
        outcomes="Y",
        latents=("U", "V"),
    )

    assert algorithm.latents == ("U", "V")
    assert set(algorithm._variables) == {"W", "X", "Y", "Z"}
    assert set(algorithm._graph.directed) == {
        (FakeVariable("X"), FakeVariable("Y")),
        (FakeVariable("X"), FakeVariable("Z")),
    }
    assert set(algorithm._graph.undirected) == {
        (FakeVariable("Y"), FakeVariable("Z")),
    }
    assert algorithm._graph.nodes == {
        FakeVariable("W"),
        FakeVariable("X"),
        FakeVariable("Y"),
        FakeVariable("Z"),
    }


def test_id_algorithm_run_passes_query_and_renders_formula(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}
    x = FakeVariable("X")
    y = FakeVariable("Y")

    def identify(
        graph: FakeMixedGraph,
        *,
        treatments: set[FakeVariable],
        outcomes: set[FakeVariable],
        conditions: set[FakeVariable] | None = None,
    ) -> FakeProbability:
        calls.update(
            graph=graph,
            treatments=treatments,
            outcomes=outcomes,
            conditions=conditions,
        )
        return probability((y,), (x,))

    install_fake_y0(monkeypatch, identify)
    algorithm = identification.IDAlgorithm(
        "X -> Y",
        treatments="X",
        outcomes="Y",
    )

    assert algorithm.run() == "p(Y | X)"
    assert calls == {
        "graph": algorithm._graph,
        "treatments": {x},
        "outcomes": {y},
        "conditions": None,
    }


def test_id_algorithm_run_passes_conditioning_set_as_conditions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}
    c = FakeVariable("C")
    x = FakeVariable("X")
    y = FakeVariable("Y")

    def identify(
        graph: FakeMixedGraph,
        *,
        treatments: set[FakeVariable],
        outcomes: set[FakeVariable],
        conditions: set[FakeVariable] | None = None,
    ) -> FakeProbability:
        calls["conditions"] = conditions
        return probability((y,), (c, x))

    install_fake_y0(monkeypatch, identify)
    algorithm = identification.IDAlgorithm(
        "C -> X; C -> Y; X -> Y",
        treatments="X",
        outcomes="Y",
        conditioning_set="C",
    )

    assert algorithm.run() == "p(Y | C, X)"
    assert calls == {"conditions": {FakeVariable("C")}}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"treatments": "X", "outcomes": "Y", "conditioning_set": "X"},
        {"treatments": "X", "outcomes": "Y", "conditioning_set": "Y"},
    ],
)
def test_id_algorithm_rejects_conditioning_set_overlapping_query(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, str],
) -> None:
    install_fake_y0(monkeypatch)

    with pytest.raises(ValueError, match="must be disjoint"):
        identification.IDAlgorithm("C -> X; C -> Y; X -> Y", **kwargs)


def test_id_algorithm_run_returns_none_for_nonidentifiable_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_y0(monkeypatch)
    algorithm = identification.IDAlgorithm(
        "X -> Y",
        treatments="X",
        outcomes="Y",
    )

    assert algorithm.run() is None


def test_y0_renderer_formats_products_sums_and_treatment_copies() -> None:
    x = FakeVariable("X")
    y = FakeVariable("Y")
    z = FakeVariable("Z")
    inner = FakeSum(
        expression=FakeProduct(
            (
                probability((y,), (x, z)),
                probability((x,)),
            )
        ),
        ranges=(x,),
    )
    expression = FakeSum(
        expression=FakeProduct((probability((z,), (x,)), inner)),
        ranges=(z,),
    )

    rendered = identification._Y0Renderer(
        treatments=("X",),
        dsl=FAKE_DSL,
    ).render(expression)

    assert rendered == (
        "sum_{Z}{ p(Z | X) * sum_{X'}{ p(Y | X', Z) * p(X') } }"
    )
    assert parse_and_validate(rendered).signature.outputs


def test_y0_renderer_handles_supported_and_arbitrary_fractions() -> None:
    x = FakeVariable("X")
    y = FakeVariable("Y")
    z = FakeVariable("Z")
    renderer = identification._Y0Renderer(treatments=("X",), dsl=FAKE_DSL)

    base_fraction = FakeFraction(
        numerator=probability((y, x), (z,)),
        denominator=probability((x,), (z,)),
    )
    assert renderer.render(base_fraction) == "(p(X, Y | Z) / p(X | Z))"

    numerator = FakeProduct((probability((x,)), probability((y,), (x,))))
    marginal_denominator = FakeSum(expression=numerator, ranges=(y,))
    assert (
        renderer.render(FakeFraction(numerator, marginal_denominator))
        == "icd_{X|}{ p(X) * p(Y | X) }"
    )

    with pytest.raises(NotImplementedError, match="arbitrary y0 fraction"):
        renderer.render(FakeFraction(numerator, probability((x,))))


def test_y0_renderer_collapses_trivial_self_normalization() -> None:
    # IDC always normalises its estimand by marginalising out the
    # outcomes, even when nothing was left to condition on.
    x = FakeVariable("X")
    y = FakeVariable("Y")
    numerator = probability((y,), (x,))
    denominator = FakeSum(expression=numerator, ranges=(y,))

    renderer = identification._Y0Renderer(treatments=("X",), dsl=FAKE_DSL)

    assert renderer.render(FakeFraction(numerator, denominator)) == "p(Y | X)"
