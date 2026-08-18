from __future__ import annotations

import pytest

from hiprof.graph import adjacency_to_graph, parse_graph


def directed_edges(graph_source: str) -> set[tuple[str, str]]:
    graph = parse_graph(graph_source)
    return {
        (parent.name, child.name)
        for parent in graph.nodes.values()
        for child in parent.children
    }


@pytest.mark.parametrize(
    ("edge_direction", "expected"),
    [
        ("from row to column", "T -> Y"),
        ("from column to row", "Y -> T"),
    ],
)
def test_adjacency_to_graph_respects_edge_direction(
    edge_direction: str,
    expected: str,
) -> None:
    graph = adjacency_to_graph(
        [[0, 1], [0, 0]],
        nodes=["T", "Y"],
        edge_direction=edge_direction,  # type: ignore[arg-type]
    )

    assert graph == expected


@pytest.mark.parametrize(
    "edge_direction",
    ["from row to column", "from column to row"],
)
def test_adjacency_to_graph_combines_reciprocal_arrows(
    edge_direction: str,
) -> None:
    graph = adjacency_to_graph(
        [[0, 1], [1, 0]],
        nodes=["T", "Y"],
        edge_direction=edge_direction,  # type: ignore[arg-type]
    )

    assert graph == "T <-> Y"


def test_adjacency_to_graph_defaults_names_and_retains_isolated_nodes() -> (
    None
):
    graph = adjacency_to_graph(
        [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
    )

    assert graph == "X2\nX0 -> X1"
    assert tuple(parse_graph(graph).nodes) == ("X2", "X0", "X1")


def test_adjacency_to_graph_reads_dataframe_like_input_positionally() -> None:
    values = [[0, 1], [0, 0]]

    class PositionalIndexer:
        def __getitem__(self, position: tuple[int, int]) -> int:
            row, column = position
            return values[row][column]

    class DataFrameLike:
        shape = (2, 2)
        iloc = PositionalIndexer()

        def __len__(self) -> int:
            return 2

    graph = adjacency_to_graph(
        DataFrameLike(),  # type: ignore[arg-type]
        nodes=["T", "Y"],
    )

    assert graph == "T -> Y"


@pytest.mark.parametrize(
    ("adjacency", "message"),
    [
        ([], "at least one node"),
        ([[0, 1]], "must be square"),
        ([[0, 2], [0, 0]], "must be 0 or 1"),
        ([[1]], "Diagonal"),
        ([["0"]], "must be 0 or 1"),
    ],
)
def test_adjacency_to_graph_rejects_invalid_matrices(
    adjacency: list[list[object]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        adjacency_to_graph(adjacency)


def test_adjacency_to_graph_accepts_float_zero_and_one() -> None:
    graph = adjacency_to_graph([[0.0, 1.0], [0.0, 0.0]])

    assert graph == "X0 -> X1"


@pytest.mark.parametrize(
    ("nodes", "message"),
    [
        (["X"], "Expected 2"),
        (["X", "X"], "unique"),
        (["x", "Y"], "Invalid node name"),
    ],
)
def test_adjacency_to_graph_rejects_invalid_node_names(
    nodes: list[str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        adjacency_to_graph([[0, 0], [0, 0]], nodes=nodes)


def test_adjacency_to_graph_rejects_invalid_direction() -> None:
    with pytest.raises(ValueError, match="edge_direction"):
        adjacency_to_graph(
            [[0]],
            edge_direction="sideways",  # type: ignore[arg-type]
        )


def test_adjacency_to_graph_rejects_directed_cycles() -> None:
    with pytest.raises(ValueError, match="directed cycle"):
        adjacency_to_graph(
            [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
        )


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
