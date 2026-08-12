from __future__ import annotations

import pytest

from hiprof.graph import parse_graph


def directed_edges(graph_source: str) -> set[tuple[str, str]]:
    graph = parse_graph(graph_source)
    return {
        (parent.name, child.name)
        for parent in graph.nodes.values()
        for child in parent.children
    }


def test_parse_graph_preserves_isolated_nodes_and_string_round_trip() -> None:
    graph = parse_graph("Z, X -> Y;\n")

    assert tuple(graph.nodes) == ("Z", "X", "Y")
    assert graph.nodes["Z"].observed
    assert graph.nodes["Z"].parents == set()
    assert graph.nodes["Z"].children == set()
    assert str(graph) == "Z\nX -> Y"
    assert tuple(parse_graph(str(graph)).nodes) == ("Z", "X", "Y")
    assert directed_edges(str(graph)) == {("X", "Y")}


def test_parse_graph_represents_bidirected_edge_with_latent_parent() -> None:
    graph = parse_graph("Y <-> X")

    latent = graph.nodes["U_X_Y"]
    assert not latent.observed
    assert {child.name for child in latent.children} == {"X", "Y"}
    assert latent in graph.nodes["X"].parents
    assert latent in graph.nodes["Y"].parents
    assert str(graph) == "U_X_Y -> X\nU_X_Y -> Y"


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("X01", "Invalid graph statement"),
        ("X -> Y -> Z", "exactly one edge"),
        ("x -> Y", "left side"),
        ("X -> y", "right side"),
    ],
)
def test_parse_graph_rejects_invalid_statements(
    source: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_graph(source)
