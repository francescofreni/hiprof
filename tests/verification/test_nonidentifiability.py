from __future__ import annotations

import builtins
import sys
from types import ModuleType
from typing import Any

import pytest

from hiprof import HPFalsifier


def install_fake_ananke(monkeypatch: pytest.MonkeyPatch) -> None:
    ananke = ModuleType("ananke")
    graphs = ModuleType("ananke.graphs")
    admg = ModuleType("ananke.graphs.admg")
    identification = ModuleType("ananke.identification")
    admg.latent_project_single_vertex = lambda: None  # type: ignore[attr-defined]
    ananke.graphs = graphs  # type: ignore[attr-defined]
    ananke.identification = identification  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "ananke", ananke)
    monkeypatch.setitem(sys.modules, "ananke.graphs", graphs)
    monkeypatch.setitem(sys.modules, "ananke.graphs.admg", admg)
    monkeypatch.setitem(sys.modules, "ananke.identification", identification)


def test_check_none_rejects_fully_observed_nonidentifiability_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_ananke(monkeypatch)
    falsifier = HPFalsifier(
        "X -> Y",
        treatments="X",
        outcomes="Y",
    )

    result = falsifier.check(None)

    assert not result.accepted


def test_check_none_projects_explicit_latent_dag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}

    class FakeDAG:
        def __init__(
            self,
            vertices: list[str],
            di_edges: list[tuple[str, str]],
        ) -> None:
            calls["vertices"] = vertices
            calls["di_edges"] = di_edges

        @staticmethod
        def topological_sort() -> tuple[str, ...]:
            return "U", "X", "Y"

    def latent_project_single_vertex(vertex: str, graph: Any) -> object:
        calls.setdefault("projected", []).append(vertex)
        return "projected", vertex, graph

    class FakeOneLineID:
        def __init__(
            self,
            graph: Any,
            treatments: tuple[str, ...],
            outcomes: tuple[str, ...],
        ) -> None:
            calls["id_graph"] = graph
            calls["treatments"] = treatments
            calls["outcomes"] = outcomes

        @staticmethod
        def id() -> bool:
            return False

    ananke = ModuleType("ananke")
    graphs = ModuleType("ananke.graphs")
    admg = ModuleType("ananke.graphs.admg")
    identification = ModuleType("ananke.identification")
    graphs.DAG = FakeDAG  # type: ignore[attr-defined]
    admg.latent_project_single_vertex = (  # type: ignore[attr-defined]
        latent_project_single_vertex
    )
    identification.OneLineID = FakeOneLineID  # type: ignore[attr-defined]
    ananke.graphs = graphs  # type: ignore[attr-defined]
    ananke.identification = identification  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "ananke", ananke)
    monkeypatch.setitem(sys.modules, "ananke.graphs", graphs)
    monkeypatch.setitem(sys.modules, "ananke.graphs.admg", admg)
    monkeypatch.setitem(sys.modules, "ananke.identification", identification)

    falsifier = HPFalsifier(
        "U -> X; U -> Y; X -> Y",
        treatments="X",
        outcomes="Y",
        latents="U",
    )

    result = falsifier.check(None)

    assert result.accepted
    assert calls["vertices"] == ["U", "X", "Y"]
    assert set(calls["di_edges"]) == {
        ("U", "X"),
        ("U", "Y"),
        ("X", "Y"),
    }
    assert calls["projected"] == ["U"]
    assert calls["treatments"] == ("X",)
    assert calls["outcomes"] == ("Y",)


def test_check_none_without_ananke_raises_informative_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(
        name: str,
        glob: Any = None,
        loc: Any = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "ananke" or name.startswith("ananke."):
            raise ModuleNotFoundError(
                "No module named 'ananke'",
                name="ananke",
            )

        return real_import(name, glob, loc, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    falsifier = HPFalsifier(
        "X -> Y; X <-> Y",
        treatments="X",
        outcomes="Y",
    )

    with pytest.raises(
        ImportError,
        match=r"hiprof\[nonidentifiability\]",
    ):
        falsifier.check(None)
