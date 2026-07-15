import pandas as pd
from typing import Tuple
from collections import deque
import warnings

# suppress some ananke-related warnings that are safe to ignore
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="google.api_core"
)
warnings.filterwarnings("ignore", message=".*IProgress not found.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"pgmpy\..*")
from ananke.graphs import ADMG


class Graph:
    """
    Graph class. The graph can be a directed acyclic graph (DAG)
    or an acyclic directed mixed graph (ADMG).
    """

    def __init__(
        self,
        directed: pd.DataFrame,
        bidirected: pd.DataFrame | None = None,
        row_to_col: bool = True,
    ) -> None:
        """
        Initialize the Graph object.

        Parameters
        ----------
        directed : pd.DataFrame
            Adjacency matrix -- with named rows and columns -- specifying directed edges.
            The column names must be the same as the row names.
        bidirected : pd.DataFrame | None, default = None
            Adjacency matrix -- with named rows and columns -- specifying bidirected edges.
            The column names must be the same as the row names.
            Must be symmetric.
        row_to_col : bool, optional
            If True, edges go from rows to columns. If False, edges go from columns to rows.
            By default, True.
        """
        self.is_admg = bidirected is not None

        if not row_to_col:
            directed = directed.T
            if self.is_admg:
                bidirected = bidirected.T

        if self.is_admg:
            if not directed.index.equals(
                bidirected.index
            ) or not directed.columns.equals(bidirected.columns):
                raise ValueError(
                    "'directed' and 'bidirected' must "
                    "have the same row and column names."
                )

            if not bidirected.equals(bidirected.T):
                raise ValueError("'bidirected' must be symmetric.")

        if not directed.index.equals(directed.columns):
            raise ValueError(
                "Rows and columns must represent the same set of nodes."
            )

        isacyclic, cycle = self._is_acyclic(directed)
        if not isacyclic:
            raise ValueError(
                "'directed' is not acyclic. "
                f"It contains the following cycle: {cycle}."
            )

        self.directed = directed
        self.bidirected = bidirected
        self.observed_nodes = directed.index.to_list()
        self.nodes = self.observed_nodes.copy()

        self.adjmat = self.directed.copy()
        if self.is_admg:
            # canonical DAG
            self._admg_to_dag_with_latents()

    @staticmethod
    def _is_acyclic(adj_mat: pd.DataFrame) -> Tuple[bool, list | None]:
        """
        Check if the adjacency matrix is a DAG.

        Reference: https://cp-algorithms.com/graph/finding-cycle.html

        Parameters
        ----------
        adj_mat : pd.DataFrame
            Adjacency matrix with named rows and columns.

        Returns
        -------
        bool
            True if the adjacency matrix is a DAG, False otherwise.
        list | None
            If adj_mat is a DAG, None, otherwise a list with the cycle.
        """

        def dfs(v):
            nonlocal cycle_start, cycle_end
            color[v] = 1

            for u in nodes:
                if not adj_mat.loc[v, u]:
                    continue
                if color[u] == 0:
                    parent[u] = v
                    if dfs(u):
                        return True
                elif color[u] == 1:
                    cycle_start = u
                    cycle_end = v
                    return True

            color[v] = 2
            return False

        nodes = adj_mat.index.to_list()

        color = {
            node: 0 for node in nodes
        }  # 0 = unvisited, 1 = visiting, 2 = done
        parent = {node: None for node in nodes}
        cycle_start = None
        cycle_end = None

        for v in nodes:
            if color[v] == 0 and dfs(v):
                break

        if cycle_start is None:
            return True, None

        cycle = [cycle_start]
        v = cycle_end
        while v != cycle_start:
            cycle.append(v)
            v = parent[v]
        cycle.append(cycle_start)
        cycle.reverse()

        return False, cycle

    def _admg_to_dag_with_latents(self) -> None:
        """
        Introduce a latent variable for each bidirected edge.

         For each bidirected edge 'X <-> Y', the inserted variable will take names of the form
        'U_X_Y' and new directed edges 'U_X_Y -> Y' and 'U_X_Y -> X' are inserted.
        """
        all_nodes = self.nodes.copy()
        dag = self.directed.copy().astype(int)

        for i, X in enumerate(self.nodes):
            for Y in self.nodes[i + 1 :]:
                if self.bidirected.loc[X, Y]:
                    latent_name = f"U_{X}_{Y}"
                    all_nodes.append(latent_name)

        dag = dag.reindex(index=all_nodes, columns=all_nodes, fill_value=0)

        for i, X in enumerate(self.nodes):
            for Y in self.nodes[i + 1 :]:
                if self.bidirected.loc[X, Y]:
                    latent_name = f"U_{X}_{Y}"
                    dag.loc[latent_name, X] = 1
                    dag.loc[latent_name, Y] = 1

        self.nodes = all_nodes
        self.adjmat = dag

    @staticmethod
    def get_topological_order(adjmat: pd.DataFrame) -> list[str]:
        """
        Given an adjacency matrix, return the topological order of the nodes.
        This function uses Kahn's algorithm (see https://en.wikipedia.org/wiki/Topological_sorting).

        Parameters
        ----------
        adjmat : pd.DataFrame
            Adjacency matrix with named rows and columns.

        Returns
        -------
        order: list[str]
            Topological order of the nodes.
        """
        nodes = adjmat.index.to_list()
        indeg = {v: 0 for v in nodes}
        children = {v: [] for v in nodes}

        for parent in nodes:
            for child in nodes:
                if adjmat.loc[parent, child] != 0:
                    children[parent].append(child)
                    indeg[child] += 1

        q = deque([v for v in nodes if indeg[v] == 0])
        order = []

        while q:
            v = q.popleft()
            order.append(v)
            for child in children[v]:
                indeg[child] -= 1
                if indeg[child] == 0:
                    q.append(child)

        if len(order) != len(nodes):
            raise ValueError("The graph is not acyclic.")

        return order

    def maximal_arid_projection(self) -> "Graph":
        """
        Return the maximal arid projection of this graph.

        If the graph is a DAG, this function returns a copy of the graph.
        If the graph is an ADMG, it uses ananke's implementation to build
        the maximal arid projection.

        Returns
        -------
        Graph
            A new Graph object representing the maximal arid projection.
        """
        if not self.is_admg:
            return Graph(
                directed=self.directed.copy(), bidirected=self.bidirected
            )
        vertices = self.observed_nodes

        di_edges = [
            (src, dst)
            for src in vertices
            for dst in vertices
            if self.directed.loc[src, dst] != 0
        ]

        bi_edges = []
        for i, src in enumerate(vertices):
            for dst in vertices[i + 1 :]:
                if self.bidirected.loc[src, dst] != 0:
                    bi_edges.append((src, dst))

        g_ananke = ADMG(
            vertices=vertices, di_edges=di_edges, bi_edges=bi_edges
        )
        g_arid = g_ananke.maximal_arid_projection()

        # Convert back to adjacency matrices
        directed_new = pd.DataFrame(
            0, index=vertices, columns=vertices, dtype=int
        )
        bidirected_new = pd.DataFrame(
            0, index=vertices, columns=vertices, dtype=int
        )

        for src, dst in g_arid.di_edges:
            directed_new.loc[src, dst] = 1

        for a, b in g_arid.bi_edges:
            bidirected_new.loc[a, b] = 1
            bidirected_new.loc[b, a] = 1

        return Graph(directed=directed_new, bidirected=bidirected_new)

    def make_maximal_arid_projection(self) -> None:
        """
        Replace this graph by its maximal arid projection in place.
        """
        graph = self.maximal_arid_projection()
        self.is_admg = graph.is_admg
        self.directed = graph.directed
        self.bidirected = graph.bidirected
        self.observed_nodes = graph.observed_nodes
        self.nodes = graph.nodes
        self.adjmat = graph.adjmat
