from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal


VARIABLE = r"[A-Z]+(?:0|[1-9][0-9]*)?"
VARIABLE_PATTERN = re.compile(rf"^{VARIABLE}$")
EDGE_PATTERN = re.compile(r"<->|->")
INVALID_CHARACTER_PATTERN = re.compile(r"[^A-Z0-9<>;,\s-]")
STATEMENT_SEPARATOR_PATTERN = re.compile(r"[,;]\n?|\n")
EdgeDirection = Literal["from row to column", "from column to row"]


@dataclass(eq=False, slots=True)
class Node:
    name: str
    observed: bool = True
    parents: set[Node] = field(default_factory=set)
    children: set[Node] = field(default_factory=set)


@dataclass
class Graph:
    nodes: dict[str, Node] = field(default_factory=dict)

    def __str__(self) -> str:
        isolated_nodes = sorted(
            node.name
            for node in self.nodes.values()
            if node.observed and not node.parents and not node.children
        )
        edges = sorted(
            (parent.name, child.name)
            for parent in self.nodes.values()
            for child in parent.children
        )
        statements = isolated_nodes + [
            f"{parent} -> {child}" for parent, child in edges
        ]
        return "\n".join(statements)

    def add_node(
        self,
        name: str,
        observed: bool = True,
    ) -> Node:
        if name not in self.nodes:
            self.nodes[name] = Node(name=name, observed=observed)

        return self.nodes[name]

    def add_directed_edge(
        self,
        parent_name: str,
        child_name: str,
    ) -> None:
        parent = self.add_node(parent_name)
        child = self.add_node(child_name)

        parent.children.add(child)
        child.parents.add(parent)

    def add_bidirected_edge(
        self,
        left_name: str,
        right_name: str,
    ) -> None:
        left = self.add_node(left_name)
        right = self.add_node(right_name)

        first, second = sorted((left_name, right_name))
        latent = self.add_node(
            f"U_{first}_{second}",
            observed=False,
        )

        latent.children.update((left, right))
        left.parents.add(latent)
        right.parents.add(latent)

    def check_acyclic(self) -> None:
        visiting: set[Node] = set()
        visited: set[Node] = set()
        path: list[Node] = []

        def visit(node: Node) -> None:
            if node in visited:
                return

            visiting.add(node)
            path.append(node)

            for child in node.children:
                if child in visiting:
                    cycle_start = path.index(child)
                    cycle = path[cycle_start:] + [child]

                    cycle_string = " -> ".join(node.name for node in cycle)
                    raise ValueError(
                        "The graph contains a directed cycle: "
                        f"{cycle_string}."
                    )

                visit(child)

            path.pop()
            visiting.remove(node)
            visited.add(node)

        for node in self.nodes.values():
            if node not in visited:
                visit(node)


def adjacency_to_graph(
    adjacency: Sequence[Sequence[object]],
    nodes: Sequence[str] | None = None,
    edge_direction: EdgeDirection = "from row to column",
) -> str:
    """Translate an adjacency matrix into a hiprof graph specification.

    Matrix entries must be either 0 or 1. A 1 denotes one arrow, with its
    orientation controlled by ``edge_direction``:

    - ``"from row to column"`` makes entry ``[r, c]`` encode ``r -> c``;
    - ``"from column to row"`` makes entry ``[r, c]`` encode ``c -> r``.

    If both ``[r, c]`` and ``[c, r]`` are 1, the two arrows are written as
    one bidirected edge, ``r <-> c``, regardless of ``edge_direction``.
    Diagonal entries must be 0. Isolated nodes are retained in the returned
    graph specification.

    The input may be any square, indexable matrix, including nested Python
    sequences, NumPy arrays, and Pandas DataFrames.

    Node names default to ``X0``, ``X1``, and so on. Explicit names must be
    unique and follow hiprof's variable syntax.

    Example:
        >>> adjacency_to_graph(
        ...     [[0, 1, 1], [0, 0, 1], [1, 0, 0]],
        ...     nodes=["T", "M", "Y"],
        ...     edge_direction="from row to column",
        ... )
        'T -> M\nT <-> Y\nM -> Y'

    :param adjacency: Square adjacency matrix containing only 0 and 1.
    :param nodes: Optional node names in matrix order.
    :param edge_direction: Orientation represented by a matrix entry of 1.
    :returns: Graph specification accepted by :func:`parse_graph`.
    :raises ValueError: If the matrix, node names, direction, or resulting
        graph is invalid.
    """
    if edge_direction not in (
        "from row to column",
        "from column to row",
    ):
        raise ValueError(
            "edge_direction must be 'from row to column' or "
            "'from column to row'."
        )

    matrix = _normalise_adjacency(adjacency)
    node_names = _normalise_node_names(nodes, len(matrix))
    row_to_column = edge_direction == "from row to column"
    connected: set[int] = set()
    edges: list[str] = []

    for row in range(len(matrix)):
        for column in range(row + 1, len(matrix)):
            row_column = matrix[row][column]
            column_row = matrix[column][row]

            if row_column and column_row:
                edges.append(f"{node_names[row]} <-> {node_names[column]}")
            elif row_column:
                source, target = (
                    (row, column) if row_to_column else (column, row)
                )
                edges.append(f"{node_names[source]} -> {node_names[target]}")
            elif column_row:
                source, target = (
                    (column, row) if row_to_column else (row, column)
                )
                edges.append(f"{node_names[source]} -> {node_names[target]}")
            else:
                continue

            connected.update((row, column))

    isolated = [
        name
        for position, name in enumerate(node_names)
        if position not in connected
    ]
    graph_text = "\n".join(isolated + edges)
    parse_graph(graph_text)
    return graph_text


def parse_graph(text: str) -> Graph:
    """Parse a graph specification.

    Statements may specify an isolated node using its variable name, or use
    ``->`` for directed edges and ``<->`` for bidirected edges. Statements
    are separated by a comma, semicolon, or newline. A comma or semicolon may
    be followed by one newline as part of the same separator. Separators may
    not be leading, trailing, or consecutive. Other whitespace is ignored,
    and all remaining characters must be uppercase letters, digits, ``<``,
    ``>``, or ``-``. Bidirected edges are represented internally by unobserved
    latent parents.

    :param text: Graph specification to parse.
    :returns: Parsed graph.
    :raises ValueError: If the graph syntax is invalid, contains a self-edge,
        or contains a directed cycle.
    """
    graph = Graph()
    text = _normalise_graph_text(text)

    for statement in _split_graph_statements(text):
        edge_matches = list(EDGE_PATTERN.finditer(statement))

        if not edge_matches:
            if VARIABLE_PATTERN.fullmatch(statement) is not None:
                graph.add_node(statement)
                continue

            raise ValueError(
                f"Invalid graph statement {statement!r}. "
                "Each statement must be either a variable name or contain "
                "exactly one edge of type '->' or '<->'."
            )

        if len(edge_matches) > 1:
            raise ValueError(
                f"Invalid graph statement {statement!r}. "
                "Each statement must contain exactly one edge."
            )

        edge_match = edge_matches[0]
        left = statement[: edge_match.start()]
        edge = edge_match.group()
        right = statement[edge_match.end() :]

        if VARIABLE_PATTERN.fullmatch(left) is None:
            raise ValueError(
                f"Invalid variable name {left!r} on the left side "
                f"of statement {statement!r}. Variable names must "
                "contain one or more uppercase letters, optionally "
                "followed by 0 or a positive integer without leading "
                "zeros, for example 'X', 'AB', 'X0', or 'Y12'."
            )

        if VARIABLE_PATTERN.fullmatch(right) is None:
            raise ValueError(
                f"Invalid variable name {right!r} on the right side "
                f"of statement {statement!r}. Variable names must "
                "contain one or more uppercase letters, optionally "
                "followed by 0 or a positive integer without leading "
                "zeros, for example 'X', 'AB', 'X0', or 'Y12'."
            )

        if left == right:
            raise ValueError(f"Self-edge {statement!r} is not allowed.")

        if edge == "->":
            graph.add_directed_edge(left, right)
        else:
            graph.add_bidirected_edge(left, right)

    graph.check_acyclic()
    return graph


def _normalise_graph_text(text: str) -> str:
    invalid_character = INVALID_CHARACTER_PATTERN.search(text)
    if invalid_character is not None:
        character = invalid_character.group()
        raise ValueError(
            f"Invalid character {character!r} in graph specification. "
            "Graph specifications may contain only whitespace, uppercase "
            "letters, digits, '<', '>', '-', ',', and ';'."
        )

    return re.sub(r"[^\S\n]+", "", text)


def _split_graph_statements(text: str) -> list[str]:
    statements = STATEMENT_SEPARATOR_PATTERN.split(text)
    if any(not statement for statement in statements):
        raise ValueError(
            "Invalid graph specification. Statements must be non-empty, "
            "and separators may not be leading, trailing, or consecutive."
        )

    return statements


def _normalise_adjacency(
    adjacency: Sequence[Sequence[object]],
) -> list[list[int]]:
    size = len(adjacency)
    if size == 0:
        raise ValueError(
            "The adjacency matrix must contain at least one node."
        )

    shape = getattr(adjacency, "shape", None)
    if shape is not None and tuple(shape) != (size, size):
        raise ValueError(
            "The adjacency matrix must be square; "
            f"received shape {tuple(shape)}."
        )

    matrix: list[list[int]] = []
    positional = getattr(adjacency, "iloc", None)

    for row in range(size):
        if positional is None and len(adjacency[row]) != size:
            raise ValueError(
                "The adjacency matrix must be square; "
                f"row {row} has length {len(adjacency[row])}, expected {size}."
            )

        matrix_row: list[int] = []
        for column in range(size):
            value = (
                positional[row, column]
                if positional is not None
                else adjacency[row][column]
            )
            if value == 0:
                entry = 0
            elif value == 1:
                entry = 1
            else:
                raise ValueError(
                    "Adjacency entries must be 0 or 1; "
                    f"entry [{row}, {column}] is {value!r}."
                )
            if row == column and entry:
                raise ValueError(
                    "Diagonal adjacency entries must be 0; "
                    f"entry [{row}, {column}] is 1."
                )

            matrix_row.append(entry)
        matrix.append(matrix_row)

    return matrix


def _normalise_node_names(
    nodes: Sequence[str] | None,
    size: int,
) -> list[str]:
    node_names = (
        list(nodes)
        if nodes is not None
        else [f"X{i}" for i in range(size)]
    )
    if len(node_names) != size:
        raise ValueError(
            f"Expected {size} node names, received {len(node_names)}."
        )
    for name in node_names:
        if (
            not isinstance(name, str)
            or VARIABLE_PATTERN.fullmatch(name) is None
        ):
            raise ValueError(
                f"Invalid node name {name!r}. Node names must contain one or "
                "more uppercase letters, optionally followed by 0 or a "
                "positive integer without leading zeros."
            )

    if len(set(node_names)) != size:
        raise ValueError("Node names must be unique.")

    return node_names
