from typing import Tuple, List
from hiprof.base.graph import Graph


class BaseCausalTask:
    """
    Base class for validating and initializing treatments, outcomes, and the graph.
    """

    def __init__(
        self,
        treatments: Tuple[str | int, ...] | List[str | int] | str | int,
        outcomes: Tuple[str | int, ...] | List[str | int] | str | int,
        graph: Graph,
    ) -> None:
        """
        Initialize BaseCausalTask.

        Parameters
        ----------
        treatments : list, tuple, str or int
            Tuple or list of treatments. It could also be a single treatment.
            Each element must correspond to a node in the graph.
        outcomes : list, tuple, str or int
            Tuple or list of outcomes. It could also be a single outcome.
            Each element must correspond to a node in the graph.
        graph : Graph
            The DAG or ADMG.
        """
        self.graph = graph

        # Treatments checks
        if isinstance(treatments, (str, int)) and not isinstance(
            treatments, bool
        ):
            treatments = [treatments]
        elif isinstance(treatments, tuple):
            treatments = list(treatments)
        elif not isinstance(treatments, list):
            raise TypeError(
                "'treatments' must be a tuple, a list, a string or an int."
            )

        if not all(
            isinstance(t, (str, int)) and not isinstance(t, bool)
            for t in treatments
        ):
            raise TypeError("Each treatment must be a string or int.")
        if len(treatments) == 0:
            raise ValueError("'treatments' cannot be empty.")
        if len(set(treatments)) != len(treatments):
            raise ValueError("'treatments' cannot contain duplicates.")

        # Outcomes checks
        if isinstance(outcomes, (str, int)):
            outcomes = [outcomes]
        elif isinstance(outcomes, tuple):
            outcomes = list(outcomes)
        elif not isinstance(outcomes, list):
            raise TypeError(
                "'outcomes' must be a tuple, a list, a string or an int."
            )

        if not all(isinstance(o, (str, int)) for o in outcomes):
            raise TypeError("Each outcome must be a string or int.")
        if len(outcomes) == 0:
            raise ValueError("'outcomes' cannot be empty.")
        if len(set(outcomes)) != len(outcomes):
            raise ValueError("'outcomes' cannot contain duplicates.")
        if len(set(outcomes).intersection(set(treatments))) > 0:
            raise ValueError(
                "'outcomes' cannot contain elements in 'treatments'."
            )

        # Final check against graph
        for node in treatments + outcomes:
            if node not in self.graph.directed.index:
                raise ValueError(
                    "All elements of 'treatments' and 'outcomes' must be among the observed nodes."
                )

        self.treatments = treatments
        self.outcomes = outcomes
        self.query_nodes = set(self.treatments) | set(self.outcomes)
