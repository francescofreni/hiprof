from __future__ import annotations

import re
from dataclasses import dataclass, field


VARIABLE = r"[A-Z]+(?:0|[1-9][0-9]*)?"
VARIABLE_PATTERN = re.compile(rf"^{VARIABLE}$")
EDGE_PATTERN = re.compile(r"<->|->")


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


def parse_graph(text: str) -> Graph:
    """Parse a graph specification.

    Statements may specify an isolated node using its variable name, or use
    ``->`` for directed edges and ``<->`` for bidirected edges. Statements
    may be separated by commas, semicolons, or newlines. Bidirected edges are
    represented internally by unobserved latent parents.

    :param text: Graph specification to parse.
    :returns: Parsed graph.
    :raises ValueError: If the graph syntax is invalid, contains a self-edge,
        or contains a directed cycle.
    """
    graph = Graph()

    for statement in re.split(r"[,;\n]+", text):
        statement = statement.strip()

        if not statement:
            continue

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
        left = statement[: edge_match.start()].strip()
        edge = edge_match.group()
        right = statement[edge_match.end() :].strip()

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
