from __future__ import annotations

import builtins
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable

import pytest

from hiprof import HPFalsifier


@dataclass(frozen=True)
class FakeVariable:
    name: str


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


def default_identify_outcomes(*args: Any, **kwargs: Any) -> None:
    return None


def install_fake_y0(
    monkeypatch: pytest.MonkeyPatch,
    identify_outcomes: Callable[..., Any] | None = None,
) -> None:
    identify_module = ModuleType("y0.algorithm.identify")
    identify_module.identify_outcomes = (  # type: ignore[attr-defined]
        identify_outcomes or default_identify_outcomes
    )
    dsl_module = ModuleType("y0.dsl")
    dsl_module.Variable = FakeVariable  # type: ignore[attr-defined]
    graph_module = ModuleType("y0.graph")
    graph_module.NxMixedGraph = FakeMixedGraph  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "y0", ModuleType("y0"))
    monkeypatch.setitem(
        sys.modules, "y0.algorithm", ModuleType("y0.algorithm")
    )
    monkeypatch.setitem(sys.modules, "y0.algorithm.identify", identify_module)
    monkeypatch.setitem(sys.modules, "y0.dsl", dsl_module)
    monkeypatch.setitem(sys.modules, "y0.graph", graph_module)


def block_y0_import(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(
        name: str,
        glob: Any = None,
        loc: Any = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "y0" or name.startswith("y0."):
            raise ModuleNotFoundError(
                "No module named 'y0'",
                name="y0",
            )

        return real_import(name, glob, loc, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_check_none_accepts_fully_observed_claim_without_y0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A fully observed graph is trivially identifiable, so `check(None)`
    # must not even attempt to import the optional y0 dependency.
    block_y0_import(monkeypatch)
    falsifier = HPFalsifier("X -> Y", treatments="X", outcomes="Y")

    result = falsifier.check(None)

    assert not result.accepted


def test_check_none_without_y0_raises_informative_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block_y0_import(monkeypatch)
    falsifier = HPFalsifier(
        "X -> Y; X <-> Y",
        treatments="X",
        outcomes="Y",
    )

    with pytest.raises(ImportError, match=r"hiprof\[identification\]"):
        falsifier.check(None)


@pytest.mark.parametrize(
    ("identified", "expected_accepted"),
    [
        (None, True),
        ("estimand", False),
    ],
)
def test_check_none_result_follows_identify_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    identified: object,
    expected_accepted: bool,
) -> None:
    install_fake_y0(monkeypatch, lambda *args, **kwargs: identified)
    falsifier = HPFalsifier(
        "X -> Y; X <-> Y",
        treatments="X",
        outcomes="Y",
    )

    assert falsifier.check(None).accepted is expected_accepted


def test_check_none_passes_conditioning_set_as_conditions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}

    def identify_outcomes(
        graph: FakeMixedGraph,
        *,
        treatments: set[FakeVariable],
        outcomes: set[FakeVariable],
        conditions: set[FakeVariable] | None = None,
    ) -> None:
        calls["conditions"] = conditions
        return None

    install_fake_y0(monkeypatch, identify_outcomes)
    falsifier = HPFalsifier(
        "X -> Z; Z -> Y; X -> Y; Z <-> Y",
        treatments="X",
        outcomes="Y",
        conditioning_set="Z",
    )

    falsifier.check(None)

    assert calls == {"conditions": {FakeVariable("Z")}}
