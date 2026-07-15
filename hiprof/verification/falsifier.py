import os
import re
import random
import numpy as np
import pandas as pd
from typing import Any, Tuple, List

from flint import fmpq
from tqdm import tqdm
from lark import Lark

from hiprof.base.basetask import BaseCausalTask
from hiprof.base.parameter import Normal, Gamma
from hiprof.base.canonicalform import (
    CanonicalForm,
    ExactCanonicalForm,
    marginal,
    conditional,
    exact_marginal,
    exact_conditional,
)
from hiprof.base.graph import Graph
from hiprof.base.utils import solve_spd, logdet_spd, to_fmpq
from hiprof.formula.visitors import CanonicalFormVisitor
from hiprof.formula.formula import Formula
from hiprof.formula.transformer import TreeToFormula, normalize_var_name

import warnings

# suppress some ananke-related warnings that are safe to ignore
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="google.api_core"
)
warnings.filterwarnings("ignore", message=".*IProgress not found.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"pgmpy\..*")
from ananke import graphs, identification

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_VAR = r"[A-Za-z][A-Za-z0-9_]*(?:'+[0-9]*)?"
_PLACEHOLDER_ITEM = re.compile(rf"^\s*({_VAR})\s*=\s*\.\s*$")
_PLACEHOLDER = re.compile(rf"\b({_VAR})\s*=\s*\.")


class HPFalsifier(BaseCausalTask):
    """
    High-Probability Falsifier.

    Given a DAG or an ADMG, a query (treatment-outcome), and a candidate formula,
    verifies whether the candidate formula identifies the target interventional distribution.

    The formula should be an observational formula for p(outcomes | do(treatments)).
    """

    def __init__(
        self,
        treatments: Tuple[str | int, ...] | List[str | int] | str | int,
        outcomes: Tuple[str | int, ...] | List[str | int] | str | int,
        graph: Graph,
    ) -> None:
        """
        Initialize the HPV falsifier.

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
        super().__init__(treatments, outcomes, graph)
        self._treatment_names = tuple(str(t) for t in self.treatments)

    # ==============
    # Floating point
    # ==============

    def _sample_SCM(
        self,
        seed: int = 42,
        use_GNMM: bool = False,
        marg: Graph | None = None,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series | pd.DataFrame]:
        """
        Given the DAG, sample the weighted adjacency matrix, offsets,
        and variances that define the linear Gaussian SCM.

        Parameters
        ----------
        seed : int, optional
            Seed for the random number generator, by default 42.
        use_GNMM : bool, optional
            Whether to build a linear Gaussian SCM with correlated errors.
            In this case, returns the parameter matrix, offsets, and
            error covariance matrix.
        marg: Graph | None, optional
            Maximal arid graph used if use_GNMM is True.

        Returns
        -------
        if not use_GNMM:
            weighted_adj_matrix : pd.DataFrame
                Weighted adjacency matrix specifying the weights in the linear Gaussian SCM.
            offsets : pd.Series
                Offsets for each variable in the linear Gaussian SCM.
            variances : pd.Series
                Noise variances in the linear Gaussian SCM.
        else:
            B: pd.DataFrame
                Parameter matrix with non-zero values where directed edges are present.
            offsets : pd.Series
                Offsets for each variable in the linear Gaussian SCM.
            Omega: pd.DataFrame
                Covariance matrix of the error term.
        """
        if not use_GNMM:
            weighted_adj_mat_dist = Normal(seed=seed)
            offsets_dist = Normal(seed=seed)
            variances_dist = Gamma(seed=seed)

            weighted_adj_mat = pd.DataFrame(
                0.0, index=self.graph.nodes, columns=self.graph.nodes
            )
            for parent in self.graph.nodes:
                for child in self.graph.nodes:
                    if self.graph.adjmat.loc[parent, child] != 0:
                        weighted_adj_mat.loc[parent, child] = (
                            weighted_adj_mat_dist.sample()
                        )

            offsets = pd.Series(
                {node: offsets_dist.sample() for node in self.graph.nodes},
                index=self.graph.nodes,
            )

            variances = pd.Series(
                {node: variances_dist.sample() for node in self.graph.nodes},
                index=self.graph.nodes,
            )

            return weighted_adj_mat, offsets, variances

        np.random.seed(seed)
        nodes = marg.observed_nodes

        B = pd.DataFrame(0.0, index=nodes, columns=nodes)
        for parent in nodes:
            for child in nodes:
                if marg.directed.loc[parent, child] != 0:
                    B.loc[parent, child] = np.random.normal(0, 1)

        offsets = pd.Series(
            np.random.normal(0, 1, size=len(nodes)), index=nodes
        )

        Omega = pd.DataFrame(0.0, index=nodes, columns=nodes)
        if marg.is_admg:
            for i, u in enumerate(nodes):
                for v in nodes[i + 1 :]:
                    if marg.bidirected.loc[u, v] != 0:
                        value = np.random.normal(0, 1)
                        Omega.loc[u, v] = value
                        Omega.loc[v, u] = value

        # diagonally dominant to ensure it is positive semi-definite
        for v in nodes:
            Omega.loc[v, v] = np.sum(
                np.abs(Omega.loc[v, :])
            ) + np.random.gamma(2, 2)

        return B, offsets, Omega

    def _build_joint(
        self,
        weighted_adj_mat: pd.DataFrame,
        offsets: pd.Series,
        variances: pd.Series,
        int_values: List[float] | Tuple[float, ...] | None = None,
    ) -> CanonicalForm | Tuple[CanonicalForm, CanonicalForm]:
        """
        Build the joint distribution over all variables in the linear Gaussian SCM.
        If int_value is not None, computes the true interventional distribution
        p(self.outcomes | do(self.treatments=int_values)).

        Parameters
        ----------
        weighted_adj_mat : pd.DataFrame
            Weighted adjacency matrix specifying the weights in the linear Gaussian SCM.
        offsets : pd.Series
            Offsets for each variable in the linear Gaussian SCM.
        variances : pd.Series
            Noise variances in the linear Gaussian SCM.
        int_values : list or tuple, optional
            Values assigned to the treatments.

        Returns
        -------
        CanonicalForm | Tuple[CanonicalForm, CanonicalForm]
            joint distribution in canonical form.
            If int_values is not None, also the true interventional distribution
        """
        factors = {}
        nodes = weighted_adj_mat.index.to_list()
        for child in nodes:
            parent_mask = weighted_adj_mat[child] != 0
            parents = weighted_adj_mat.index[parent_mask]

            if len(parents) == 0:
                cf = marginal(
                    name=child,
                    mean=offsets[child],
                    variance=variances[child],
                )
            else:
                cf = conditional(
                    name=child,
                    parents=parents,
                    w=tuple(weighted_adj_mat.loc[parents, child]),
                    b=offsets[child],
                    variance=variances[child],
                )

            factors[child] = cf

        joint = None
        for child in nodes:
            if joint is None:
                joint = factors[child]
            else:
                joint *= factors[child]

        if int_values is not None:
            # remove incoming edges on the treatment
            factors_do = factors.copy()
            for treatment in self.treatments:
                factors_do[treatment] = marginal(
                    name=treatment,
                    mean=offsets[treatment],
                    variance=variances[treatment],
                )

            joint_do = None
            for child in nodes:
                if joint_do is None:
                    joint_do = factors_do[child]
                else:
                    joint_do *= factors_do[child]

            query = self.treatments + self.outcomes
            nuisance = tuple([node for node in nodes if node not in query])
            p_out_treat_do = joint_do.marginalization(nuisance)

            p_treat_do = p_out_treat_do.marginalization(tuple(self.outcomes))
            p_out_do_treat = p_out_treat_do / p_treat_do
            evidence = {
                treatment: int_values[idx]
                for idx, treatment in enumerate(self.treatments)
            }
            p_out_do_treat = p_out_do_treat.reduction(evidence)

            # or just:
            # p_out_do_treat = p_out_treat_do.reduction(evidence)

            return joint, p_out_do_treat

        return joint

    @staticmethod
    def _gaussian_to_canonical(
        mu: pd.Series,
        Sigma: pd.DataFrame,
    ) -> CanonicalForm:
        """
        Convert a Gaussian distribution into its canonical form.

        Parameters
        ----------
        mu: pd.Series
            Mean of the Gaussian distribution.
        Sigma: pd.DataFrame
            Covariance matrix of the Gaussian distribution.

        Returns
        -------
        CanonicalForm
            Gaussian distribution in canonical form.
        """
        nodes = Sigma.index.to_list()
        n = len(nodes)

        Sigma_np = Sigma.to_numpy(dtype=float)
        mu_np = mu.to_numpy(dtype=float)

        I = np.eye(n)

        J_np = solve_spd(Sigma_np, I)
        J = pd.DataFrame(J_np, index=nodes, columns=nodes)

        h_np = solve_spd(Sigma_np, mu_np)
        h = pd.Series(h_np, index=nodes)

        logdet = logdet_spd(Sigma_np)
        g = -0.5 * (mu_np.T @ h_np + n * np.log(2 * np.pi) + logdet)

        return CanonicalForm(J, h, float(g))

    def _build_joint_GNMM(
        self,
        B: pd.DataFrame,
        offsets: pd.Series,
        Omega: pd.DataFrame,
        int_values: List[float] | Tuple[float, ...] | None = None,
    ) -> CanonicalForm | Tuple[CanonicalForm, CanonicalForm]:
        """
        Build the joint distribution over all variables in the linear Gaussian SCM.
        This function does not use the canonical DAG, but a multivariate Gaussian
        over the observed variables with correlated errors where bidirected
        edges are present.
        If int_value is not None, computes the true interventional distribution
        p(self.outcomes | do(self.treatments=int_values)).

        Parameters
        ----------
        B : pd.DataFrame
            Parameter matrix with non-zero values where directed edges are present.
        offsets : pd.Series
            Offsets for each variable in the linear Gaussian SCM.
        Omega : pd.DataFrame
            Covariance matrix of the error term.
        int_values : list or tuple, optional
            Values assigned to the treatments

        Returns
        -------
        CanonicalForm | Tuple[CanonicalForm, CanonicalForm]
            joint distribution in canonical form.
            If int_values is not None, also the true interventional distribution
        """
        nodes = B.index.to_list()
        n = len(nodes)
        I = np.eye(n)

        B_np, offsets_np, Omega_np = (
            B.to_numpy(),
            offsets.to_numpy(),
            Omega.to_numpy(),
        )

        A = I - B_np.T
        mu = np.linalg.solve(A, offsets_np)
        tmp = np.linalg.solve(A, Omega_np)
        Sigma = np.linalg.solve(A, tmp.T).T
        # avoiding potential floating-point asymmetries
        Sigma = 0.5 * (Sigma + Sigma.T)

        joint_cf = self._gaussian_to_canonical(
            pd.Series(mu, index=nodes),
            pd.DataFrame(Sigma, index=nodes, columns=nodes),
        )

        if int_values is not None:
            B_do = B.copy()
            Omega_do = Omega.copy()

            for treatment in self.treatments:
                B_do.loc[:, treatment] = 0.0

                Omega_do.loc[treatment, :] = 0.0
                Omega_do.loc[:, treatment] = 0.0
                # Keep original variance to maintain positive definiteness for J matrix
                Omega_do.loc[treatment, treatment] = Omega.loc[
                    treatment, treatment
                ]

            B_do_np, Omega_do_np = B_do.to_numpy(), Omega_do.to_numpy()

            A_do = I - B_do_np.T
            mu_do = np.linalg.solve(A_do, offsets_np)
            tmp_do = np.linalg.solve(A_do, Omega_do_np)
            Sigma_do = np.linalg.solve(A_do, tmp_do.T).T
            Sigma_do = 0.5 * (Sigma_do + Sigma_do.T)

            joint_do_cf = self._gaussian_to_canonical(
                pd.Series(mu_do, index=nodes),
                pd.DataFrame(Sigma_do, index=nodes, columns=nodes),
            )

            query = self.treatments + self.outcomes
            nuisance = tuple([node for node in nodes if node not in query])
            p_out_treat_do = joint_do_cf.marginalization(nuisance)

            evidence = {
                treatment: int_values[idx]
                for idx, treatment in enumerate(self.treatments)
            }
            p_out_do_treat = p_out_treat_do.reduction(evidence)

            return joint_cf, p_out_do_treat

        return joint_cf

    # ================
    # Exact arithmetic
    # ================

    def _sample_SCM_exact(
        self,
        seed: int = 42,
        M: int = 2**63 - 1,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Sample an integer-valued linear Gaussian SCM for exact testing.

        Weights and offsets are sampled from {-M, ..., M}; edge weights are
        resampled until nonzero. Variances are sampled from {1, ..., M}.
        """
        if M < 1:
            raise ValueError("M must be at least 1.")

        rng = random.Random(seed)
        nodes = list(self.graph.nodes)

        weighted_adj_mat = pd.DataFrame(
            [[fmpq(0) for _ in nodes] for _ in nodes],
            index=nodes,
            columns=nodes,
            dtype=object,
        )

        for parent in nodes:
            for child in nodes:
                if self.graph.adjmat.loc[parent, child] != 0:
                    value = 0
                    while value == 0:
                        value = rng.randrange(-M, M + 1)
                    weighted_adj_mat.loc[parent, child] = fmpq(value)

        offsets = pd.Series(
            [fmpq(rng.randrange(-M, M + 1)) for _ in nodes],
            index=nodes,
            dtype=object,
        )

        variances = pd.Series(
            [fmpq(rng.randrange(1, 2 * M + 1)) for _ in nodes],
            index=nodes,
            dtype=object,
        )

        return weighted_adj_mat, offsets, variances

    def _build_joint_exact(
        self,
        weighted_adj_mat: pd.DataFrame,
        offsets: pd.Series,
        variances: pd.Series,
        int_values: List[Any] | Tuple[Any, ...] | None = None,
    ) -> ExactCanonicalForm | Tuple[ExactCanonicalForm, ExactCanonicalForm]:
        """
        Exact rational version of `_build_joint`.
        """
        factors = {}
        nodes = weighted_adj_mat.index.to_list()

        for child in nodes:
            parent_mask = weighted_adj_mat[child] != fmpq(0)
            parents = list(weighted_adj_mat.index[parent_mask])

            if len(parents) == 0:
                cf = exact_marginal(
                    name=child,
                    mean=offsets.loc[child],
                    variance=variances.loc[child],
                )
            else:
                weights = tuple(weighted_adj_mat.loc[parents, child].tolist())

                cf = exact_conditional(
                    name=child,
                    parents=tuple(parents),
                    w=weights,
                    b=offsets.loc[child],
                    variance=variances.loc[child],
                )

            factors[child] = cf

        joint = None
        for child in nodes:
            joint = factors[child] if joint is None else joint * factors[child]

        if int_values is not None:
            factors_do = factors.copy()

            for treatment in self.treatments:
                factors_do[treatment] = exact_marginal(
                    name=treatment,
                    mean=offsets.loc[treatment],
                    variance=variances.loc[treatment],
                )

            joint_do = None
            for child in nodes:
                joint_do = (
                    factors_do[child]
                    if joint_do is None
                    else joint_do * factors_do[child]
                )

            query = self.treatments + self.outcomes
            nuisance = tuple(node for node in nodes if node not in query)
            p_out_treat_do = joint_do.marginalization(nuisance)

            p_treat_do = p_out_treat_do.marginalization(tuple(self.outcomes))
            p_out_do_treat = p_out_treat_do / p_treat_do

            evidence = {
                treatment: to_fmpq(int_values[idx])
                for idx, treatment in enumerate(self.treatments)
            }
            p_out_do_treat = p_out_do_treat.reduction(evidence)

            return joint, p_out_do_treat

        return joint

    @staticmethod
    def _series_equal_exact(a: pd.Series, b: pd.Series) -> bool:
        """Entrywise exact equality for rational Series."""
        if list(a.index) != list(b.index):
            return False
        return all(to_fmpq(a.loc[i]) == to_fmpq(b.loc[i]) for i in a.index)

    @staticmethod
    def _df_equal_exact(a: pd.DataFrame, b: pd.DataFrame) -> bool:
        """Entrywise exact equality for rational DataFrames."""
        if list(a.index) != list(b.index):
            return False
        if list(a.columns) != list(b.columns):
            return False
        return all(
            to_fmpq(a.loc[i, j]) == to_fmpq(b.loc[i, j])
            for i in a.index
            for j in a.columns
        )

    # ===========================================
    # Identifiability and formula parsing helpers
    # ===========================================

    def _is_identifiable(self):
        """
        Determine whether the interventional distribution is identifiable or not.

        Returns
        -------
        bool
            True if the interventional distribution is identifiable, False otherwise.
        """
        di_edges = [
            (src, dst)
            for src in self.graph.directed.index
            for dst in self.graph.directed.columns
            if self.graph.directed.loc[src, dst] != 0
        ]
        bi_edges = []
        if self.graph.is_admg:
            for i, src in enumerate(self.graph.bidirected.index):
                for dst in self.graph.bidirected.columns[i + 1 :]:
                    if self.graph.bidirected.loc[src, dst] != 0:
                        bi_edges.append((src, dst))

        g = graphs.ADMG(
            self.graph.observed_nodes, di_edges=di_edges, bi_edges=bi_edges
        )
        id_obj = identification.OneLineID(
            g, treatments=self.treatments, outcomes=self.outcomes
        )
        return id_obj.id()

    @staticmethod
    def _split_distribution_items(text: str) -> list[str]:
        return [item.strip() for item in text.split(",") if item.strip()]

    def _iter_distribution_bodies(self, formula: str):
        i = 0
        while i < len(formula):
            if (
                formula[i] != "p"
                or i + 1 >= len(formula)
                or formula[i + 1] != "("
            ):
                i += 1
                continue

            start = i + 2
            depth = 1
            j = start
            while j < len(formula) and depth:
                if formula[j] == "(":
                    depth += 1
                elif formula[j] == ")":
                    depth -= 1
                j += 1

            if depth:
                raise ValueError(
                    "Unbalanced parentheses in distribution term."
                )

            yield formula[start : j - 1]
            i = j

    def _validate_placeholders(self, formula: str) -> None:
        seen = set()

        for body in self._iter_distribution_bodies(formula):
            if "|" in body:
                target_text, cond_text = body.split("|", 1)
            else:
                target_text, cond_text = body, ""

            for item in self._split_distribution_items(target_text):
                if _PLACEHOLDER_ITEM.match(item):
                    raise ValueError(
                        "The placeholder `=.` is only allowed in conditioning "
                        "variables, e.g. `p(Z | X=.)`."
                    )

            for item in self._split_distribution_items(cond_text):
                match = _PLACEHOLDER_ITEM.match(item)
                if not match:
                    continue
                name = normalize_var_name(match.group(1))
                if name not in self._treatment_names:
                    raise ValueError(
                        "The placeholder `=.` is only allowed for treatment "
                        f"variables {self._treatment_names}; got `{name}=.`."
                    )
                seen.add(name)

        if "=." in formula and not seen:
            raise ValueError(
                "The placeholder `=.` must appear as a full conditioning item, "
                "e.g. `p(Z | X=.)`."
            )

    @staticmethod
    def _format_value(value: float | fmpq) -> str:
        return str(value)

    def _fill_placeholders(
        self,
        formula: str,
        int_values: List[float | fmpq],
    ) -> str:
        values = {
            treatment: self._format_value(int_values[idx])
            for idx, treatment in enumerate(self._treatment_names)
        }

        def replace(match):
            raw_name = match.group(1)
            name = normalize_var_name(raw_name)
            if name not in values:
                raise ValueError(
                    f"No sampled intervention value available for `{raw_name}=.`."
                )
            return f"{raw_name}={values[name]}"

        return _PLACEHOLDER.sub(replace, formula)

    @staticmethod
    def _parse_formula(
        formula: Formula | str,
        *,
        exact: bool = False,
    ) -> Formula:
        if not isinstance(formula, str):
            return formula

        parser = Lark.open(
            os.path.join(SCRIPT_DIR, "../formula/grammar.lark"),
            parser="lalr",
            transformer=TreeToFormula(exact=exact),
        )
        return parser.parse(formula)

    def _parse_for_assignment(
        self,
        formula: Formula | str,
        int_values: List[float | fmpq],
        *,
        exact: bool,
    ) -> Formula:
        if not isinstance(formula, str):
            return formula
        return self._parse_formula(
            self._fill_placeholders(formula, int_values),
            exact=exact,
        )

    def _get_intervention_assignments(
        self,
        *,
        exact: bool,
    ) -> List[List[float | fmpq]]:
        """Return the origin and coordinate basis intervention assignments."""
        n_treatments = len(self.treatments)

        if exact:
            zero = fmpq(0)
            one = fmpq(1)
        else:
            zero = 0.0
            one = 1.0

        assignments = [[zero for _ in range(n_treatments)]]

        for i in range(n_treatments):
            assignment = [zero for _ in range(n_treatments)]
            assignment[i] = one
            assignments.append(assignment)

        return assignments

    def _reduce_surviving_treatments(
        self,
        candidate_target,
        int_values: List[float | fmpq],
    ):
        """
        Fix treatment variables that survive candidate-formula evaluation.

        Before reduction, a candidate distribution may only contain outcomes and
        treatment variables. Treatment variables are internally set to the sampled
        intervention values. After reduction, only outcomes may remain.
        """
        candidate_scope = set(candidate_target.scope())
        allowed_scope = set(self.outcomes) | set(self.treatments)
        extra_scope = candidate_scope - allowed_scope

        if extra_scope:
            raise ValueError(
                "The formula for the interventional distribution must yield "
                f"a distribution over outcomes {self.outcomes}, optionally with "
                f"surviving treatments {self.treatments}, but yielded extra scope "
                f"{sorted(extra_scope)}."
            )

        evidence = {
            treatment: int_values[idx]
            for idx, treatment in enumerate(self.treatments)
            if treatment in candidate_scope
        }

        if evidence:
            candidate_target = candidate_target.reduction(evidence)

        self._validate_candidate_scope(candidate_target)
        return candidate_target

    def _validate_candidate_scope(self, candidate_target) -> None:
        """Checks that the final candidate scope is the same as the outcome."""
        candidate_scope = set(candidate_target.scope())
        outcome_scope = set(self.outcomes)

        if candidate_scope != outcome_scope:
            raise ValueError(
                "The formula for the interventional distribution must yield "
                f"a distribution over outcomes {self.outcomes}, but yielded scope "
                f"{candidate_target.scope()}."
            )

    # ==============================
    # API
    # ==============================

    def check(
        self,
        formula: Formula | str | None,
        nsim: int = 1,
        tol: float = 1e-6,
        disable_progress_bar: bool = True,
        use_GNMM: bool = False,
        backend: str = "auto",
        M: int = 2**63 - 1,
    ) -> Tuple[bool, float]:
        """
        Check if the candidate formula identifies the target
        interventional distribution, depending on mode.

        Parameters
        ----------
        formula : Formula | str | None
            The candidate formula.
        nsim : int, optional
            Number of sampled linear Gaussian SCMs.
        tol : float, optional
            The tolerance parameter for the difference between the target
            and the distribution obtained using the candidate formula.
            Ignored by backend="exact".
        disable_progress_bar : bool, default False
            Whether to disable the progress bar during iteration.
        use_GNMM : bool, default False
            Whether to use a linear Gaussian SCM corresponding to the maximal arid
            projection of the ADMG. If the graph is a DAG, nothing changes.
            The linear SCM corresponding to the MArG is exactly the Gaussian
            nested Markov model.
            Only used when `mode`=='distribution'.
            For more details, see:
            I. Shpitser, R. Evans, T. Richardson (2018). Acyclic Linear SEMs
                Obey the Nested Markov Property.
        backend : {'float', 'exact', 'auto'}, default='auto'
            Computation backend.
        M : int, default=2**63-1
            Exact mode samples edge coefficients from {-M, ..., M}\{0},
            and conditional variances from {1, ..., 2M}.

        Returns
        -------
        bool
            True if the formula identifies the target, False otherwise.
        float
            Fraction of times the candidate formula yielded the true target.
        """
        if backend not in {"float", "exact", "auto"}:
            raise ValueError(
                "backend must be either 'float', 'exact', or 'auto'."
            )

        if backend == "auto":
            backend = "exact" if not use_GNMM else "float"

        if backend == "exact" and use_GNMM:
            raise NotImplementedError(
                "backend='exact' currently supports use_GNMM=False only."
            )

        if formula is None:
            if (
                not self.graph.is_admg
            ):  # in a DAG, all effects are identifiable
                return False, 0.0

            if (
                not self._is_identifiable()
            ):  # candidate formula is None and effect is not identifiable
                return True, 1.0

            return (
                False,
                0.0,
            )  # candidate formula is None and effect is identifiable

        if (
            not self._is_identifiable()
        ):  # candidate formula is not None and effect is not identifiable
            return False, 0.0

        if isinstance(formula, str):
            self._validate_placeholders(formula)

        if backend == "exact":
            return self._check_exact(
                formula=formula,
                nsim=nsim,
                disable_progress_bar=disable_progress_bar,
                M=M,
            )

        return self._check_float(
            formula=formula,
            nsim=nsim,
            tol=tol,
            disable_progress_bar=disable_progress_bar,
            use_GNMM=use_GNMM,
            M=M,
        )

    def _check_float(
        self,
        formula: Formula | str,
        nsim: int,
        tol: float,
        disable_progress_bar: bool,
        use_GNMM: bool,
        M: int,
    ) -> Tuple[bool, float]:
        """Floating-point checking loop."""
        if use_GNMM and self.graph.is_admg:
            marg = self.graph.maximal_arid_projection()
        else:
            marg = self.graph

        num_correct = 0
        for i in tqdm(range(nsim), disable=disable_progress_bar):

            if not use_GNMM:
                weighted_adj_mat, offsets, variances = self._sample_SCM(seed=i)
                joint = self._build_joint(weighted_adj_mat, offsets, variances)
            else:
                B, offsets, Omega = self._sample_SCM(
                    seed=i, use_GNMM=True, marg=marg
                )
                joint = self._build_joint_GNMM(B, offsets, Omega)

            assignments = self._get_intervention_assignments(exact=False)
            assignment_matches = 0

            for int_values in assignments:
                parsed_formula = self._parse_for_assignment(
                    formula,
                    int_values,
                    exact=False,
                )
                visitor = CanonicalFormVisitor(joint)
                candidate_int_dist = parsed_formula.accept(visitor)

                if not isinstance(candidate_int_dist, CanonicalForm):
                    raise ValueError(
                        "The formula for the interventional distribution "
                        "did not yield a distribution."
                    )

                candidate_int_dist = self._reduce_surviving_treatments(
                    candidate_int_dist,
                    int_values,
                )

                if not use_GNMM:
                    _, true_int_dist = self._build_joint(
                        weighted_adj_mat,
                        offsets,
                        variances,
                        int_values,
                    )
                else:
                    _, true_int_dist = self._build_joint_GNMM(
                        B,
                        offsets,
                        Omega,
                        int_values,
                    )

                true_int_mean = true_int_dist.mean()[self.outcomes]
                true_int_cov = true_int_dist.covariance().loc[
                    self.outcomes, self.outcomes
                ]

                candidate_int_mean = candidate_int_dist.mean()[self.outcomes]
                candidate_int_cov = candidate_int_dist.covariance().loc[
                    self.outcomes, self.outcomes
                ]

                means_match = np.allclose(
                    true_int_mean.to_numpy(dtype=float),
                    candidate_int_mean.to_numpy(dtype=float),
                    atol=tol,
                    rtol=0,
                )
                covariances_match = np.allclose(
                    true_int_cov.to_numpy(dtype=float),
                    candidate_int_cov.to_numpy(dtype=float),
                    atol=tol,
                    rtol=0,
                )

                if means_match and covariances_match:
                    assignment_matches += 1

            if assignment_matches == len(assignments):
                num_correct += 1

        correct = num_correct == nsim
        return correct, num_correct / nsim

    def _check_exact(
        self,
        formula: Formula | str,
        nsim: int,
        disable_progress_bar: bool,
        M: int,
    ) -> Tuple[bool, float]:
        """Exact rational checking loop for interventional distributions."""
        num_correct = 0

        for i in tqdm(range(nsim), disable=disable_progress_bar):
            weighted_adj_mat, offsets, variances = self._sample_SCM_exact(
                seed=i,
                M=M,
            )

            joint = self._build_joint_exact(
                weighted_adj_mat,
                offsets,
                variances,
            )

            assignments = self._get_intervention_assignments(exact=True)
            assignment_matches = 0

            for int_values in assignments:
                parsed_formula = self._parse_for_assignment(
                    formula,
                    int_values,
                    exact=True,
                )
                visitor = CanonicalFormVisitor(joint)
                candidate_int_dist = parsed_formula.accept(visitor)

                if not isinstance(candidate_int_dist, ExactCanonicalForm):
                    raise ValueError(
                        "The formula for the interventional distribution "
                        "did not yield an ExactCanonicalForm."
                    )

                candidate_int_dist = self._reduce_surviving_treatments(
                    candidate_int_dist,
                    int_values,
                )

                _, true_int_dist = self._build_joint_exact(
                    weighted_adj_mat,
                    offsets,
                    variances,
                    int_values,
                )

                true_int_mean = true_int_dist.mean().loc[self.outcomes]
                true_int_cov = true_int_dist.covariance().loc[
                    self.outcomes, self.outcomes
                ]

                candidate_int_mean = candidate_int_dist.mean().loc[
                    self.outcomes
                ]
                candidate_int_cov = candidate_int_dist.covariance().loc[
                    self.outcomes, self.outcomes
                ]

                means_match = self._series_equal_exact(
                    true_int_mean,
                    candidate_int_mean,
                )
                covariances_match = self._df_equal_exact(
                    true_int_cov,
                    candidate_int_cov,
                )

                if means_match and covariances_match:
                    assignment_matches += 1

            if assignment_matches == len(assignments):
                num_correct += 1

        correct = num_correct == nsim
        return correct, num_correct / nsim
