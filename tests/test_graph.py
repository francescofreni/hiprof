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
    graph = parse_graph("Z, X -> Y")

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


def test_parse_graph_ignores_whitespace() -> None:
    graph = parse_graph("X < -\t> Y; Z 1")

    assert tuple(graph.nodes) == ("X", "Y", "U_X_Y", "Z1")


def test_parse_graph_accepts_newline_and_compound_separators() -> None:
    graph = parse_graph("X -> Y\nZ,\nU -> V;\nW")

    assert tuple(graph.nodes) == ("X", "Y", "Z", "U", "V", "W")


@pytest.mark.parametrize(
    "source",
    [
        "",
        " ",
        ",X",
        "X -> Y,",
        "X -> Y;",
        "\nX",
        "X\n",
        "Z;, U",
        "Z,,U",
        "Z\n\nU",
        "Z;\n\nU",
    ],
)
def test_parse_graph_rejects_empty_statements(source: str) -> None:
    with pytest.raises(ValueError, match="Statements must be non-empty"):
        parse_graph(source)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("X01", "Invalid graph statement"),
        ("X -> Y -> Z", "exactly one edge"),
        ("X ->", "right side"),
        ("-> Y", "left side"),
    ],
)
def test_parse_graph_rejects_invalid_statements(
    source: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_graph(source)


@pytest.mark.parametrize("source", ["x -> Y", "X_Y", "X / Y"])
def test_parse_graph_rejects_invalid_characters(source: str) -> None:
    with pytest.raises(ValueError, match="Invalid character"):
        parse_graph(source)
