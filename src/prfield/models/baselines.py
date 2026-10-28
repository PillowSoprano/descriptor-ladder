"""Models of absolute properties and of substitution responses.

  NodeModel            absolute property from (phi_A, phi_B, phi_X); the descriptor basis
  DifferenceFromNode   the same model, differenced on a substitution pair
  DirectResponseModel  the response fitted directly on substitution pairs, with a bilinear
                       context structure

The sign convention is that of `prfield.graph.typed_graph`: a pair is directed by the
lexicographic order of the two species labels and Delta = f(dst) - f(src).

`NodeModel.fit` and `DifferenceFromNode.fit` take a frame of compositions, `DirectResponseModel.fit`
a frame of substitution pairs; `DirectResponseModel.fit_from_nodes` accepts compositions.

Ridge is solved in closed form on the regularised normal equations, with the penalty applied to
the coefficient blocks and not to the intercept, so that each fitted block can be read by name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..descriptors.elemental import ElementalDescriptors, Standardizer, standardize
from ..graph.typed_graph import ComparabilityPolicy, accepted, build_edges
from ..schema import SITES


# ---------------------------------------------------------------------------
# Shared linear algebra
# ---------------------------------------------------------------------------


def ridge_solve(X: np.ndarray, y: np.ndarray, alpha: float, penalize: Optional[np.ndarray] = None) -> np.ndarray:
    """Closed-form ridge: (X^T X + alpha diag(p)) beta = X^T y, solved by lstsq.

    `penalize` is a 0/1 mask; the intercept column carries 0 so that shrinking the model
    never shifts the predicted mean, which for a difference target would introduce a
    systematic bias in Delta that looks like a real asymmetry between substitution
    directions.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    p = np.ones(X.shape[1]) if penalize is None else np.asarray(penalize, dtype=float)
    gram = X.T @ X + alpha * np.diag(p)
    return np.linalg.lstsq(gram, X.T @ y, rcond=None)[0]


def edge_endpoint_frames(edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct the (src, dst) sample coordinates of every edge.

    `build_edges` stores the varied coordinate as NaN in `site_{site}` and puts the two
    species in `src_species`/`dst_species`; this puts them back so a node-level model can
    be evaluated at both endpoints.
    """
    src = pd.DataFrame(index=edges.index)
    dst = pd.DataFrame(index=edges.index)
    site = edges["site"].to_numpy()
    for s in SITES:
        shared = edges[f"site_{s}"].to_numpy(dtype=object)
        src[f"site_{s}"] = np.where(site == s, edges["src_species"].to_numpy(dtype=object), shared)
        dst[f"site_{s}"] = np.where(site == s, edges["dst_species"].to_numpy(dtype=object), shared)
    for col in ("prototype_id", "edge_id"):
        if col in edges.columns:
            src[col] = edges[col].to_numpy()
            dst[col] = edges[col].to_numpy()
    return src, dst


def _fit_scalers(
    descriptors: Mapping[str, ElementalDescriptors], labels: Mapping[str, Sequence[str]]
) -> dict[str, Standardizer]:
    scalers: dict[str, Standardizer] = {}
    for s, desc in descriptors.items():
        present = sorted(set(labels[s]))
        _, scaler = standardize(desc.matrix(present), desc.feature_names)
        scalers[s] = scaler
    return scalers


# ---------------------------------------------------------------------------
# Node-level baseline
# ---------------------------------------------------------------------------


@dataclass
class NodeModel:
    """Absolute property from (phi_A, phi_B, phi_X), optionally with pair interactions.

    Pair interactions are the outer products phi_s (x) phi_t flattened, i.e. exactly the
    M_st blocks of the low-order model written at node level.  With them switched off the
    model is purely additive, and its differenced version is context-free by construction
    — that ablation is the cleanest way to show what the context terms buy.
    """

    descriptors: Mapping[str, ElementalDescriptors]
    prop: str
    alpha: float = 1.0
    interactions: bool = True
    sites: tuple[str, ...] = SITES

    scalers_: dict[str, Standardizer] = field(default_factory=dict, init=False)
    coef_: Optional[np.ndarray] = field(default=None, init=False)
    blocks_: dict[str, slice] = field(default_factory=dict, init=False)
    n_train_: int = field(default=0, init=False)

    # -- features ----------------------------------------------------------
    def _phi(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        out = {}
        for s in self.sites:
            labels = df[f"site_{s}"].tolist()
            out[s] = self.scalers_[s].transform(self.descriptors[s].matrix(labels))
        return out

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        phi = self._phi(df)
        n = len(df)
        blocks = [np.ones((n, 1))]
        names: list[tuple[str, int]] = [("intercept", 1)]
        for s in self.sites:
            blocks.append(phi[s])
            names.append((f"phi_{s}", phi[s].shape[1]))
        if self.interactions:
            for s, t in combinations(self.sites, 2):
                inter = np.einsum("ni,nj->nij", phi[s], phi[t]).reshape(n, -1)
                blocks.append(inter)
                names.append((f"M_{s}{t}", inter.shape[1]))
        if not self.blocks_:
            start = 0
            for name, width in names:
                self.blocks_[name] = slice(start, start + width)
                start += width
        return np.hstack(blocks)

    # -- api ---------------------------------------------------------------
    def fit(self, df_train: pd.DataFrame) -> "NodeModel":
        work = df_train.dropna(subset=[self.prop])
        labels = {s: work[f"site_{s}"].dropna().tolist() for s in self.sites}
        self.scalers_ = _fit_scalers({s: self.descriptors[s] for s in self.sites}, labels)
        self.blocks_ = {}
        X = self._design(work)
        y = work[self.prop].to_numpy(dtype=float)
        penalize = np.ones(X.shape[1])
        penalize[self.blocks_["intercept"]] = 0.0
        self.coef_ = ridge_solve(X, y, self.alpha, penalize)
        self.n_train_ = len(work)
        return self

    def predict_node(self, df: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("NodeModel.fit must be called before predict_node")
        return self._design(df) @ self.coef_

    def predict_delta(self, edges: pd.DataFrame) -> np.ndarray:
        src, dst = edge_endpoint_frames(edges)
        return self.predict_node(dst) - self.predict_node(src)

    def block(self, name: str) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("NodeModel.fit must be called before block")
        return self.coef_[self.blocks_[name]]

    def interaction_matrix(self, site1: str, site2: str) -> np.ndarray:
        """M_{site1,site2} of shape (d_site1, d_site2) as fitted at node level."""
        if not self.interactions:
            raise RuntimeError("this NodeModel was fitted without interaction terms")
        key = f"M_{site1}{site2}"
        transpose = False
        if key not in self.blocks_:
            key = f"M_{site2}{site1}"
            transpose = True
        if key not in self.blocks_:
            raise KeyError(f"no interaction block for ({site1}, {site2})")
        d1 = self.descriptors[site1].n_features
        d2 = self.descriptors[site2].n_features
        flat = self.block(key)
        M = flat.reshape((d2, d1)).T if transpose else flat.reshape((d1, d2))
        return M


@dataclass
class DifferenceFromNode:
    """A fitted NodeModel evaluated at both endpoints and subtracted.

    Has no parameters of its own. If it matches `DirectResponseModel`, fitting on
    differences adds nothing on this dataset.
    """

    node_model: NodeModel

    def fit(self, df_train: pd.DataFrame) -> "DifferenceFromNode":
        self.node_model.fit(df_train)
        return self

    def predict_node(self, df: pd.DataFrame) -> np.ndarray:
        return self.node_model.predict_node(df)

    def predict_delta(self, edges: pd.DataFrame) -> np.ndarray:
        return self.node_model.predict_delta(edges)

    @property
    def prop(self) -> str:
        return self.node_model.prop


# ---------------------------------------------------------------------------
# Direct response model
# ---------------------------------------------------------------------------


@dataclass
class ResponseBlocks:
    """The recovered parameters for substitutions at one site."""

    site: str
    n_edges: int
    intercept: float
    theta: np.ndarray                                   # (d_site,)
    context: dict[str, np.ndarray] = field(default_factory=dict)  # context site -> (d_ctx, d_site)

    def render(self) -> str:
        lines = [
            f"### response blocks — site {self.site}",
            f"- edges: {self.n_edges}, intercept: {self.intercept:.4g}",
            f"- ||theta||: {float(np.linalg.norm(self.theta)):.4g}",
        ]
        for c, M in sorted(self.context.items()):
            lines.append(f"- ||M_{c}{self.site}||_F: {float(np.linalg.norm(M)):.4g} (shape {M.shape})")
        return "\n".join(lines)


def fit_node_model_response_loss(
    descriptors, prop: str, train_nodes: pd.DataFrame, alpha: float = 1.0,
    degree_normalised: bool = False,
) -> "NodeModel":
    """The descriptor potential fitted under the graph-Laplacian response loss.

    This is the matched counterpart of `NodeModel`, not of `DirectResponseModel`. It keeps the
    single second-order descriptor potential, the shared pair matrices, the standardisation and
    the ridge, and changes only the loss from ||y - X g||^2 to ||D(y - X g)||^2, with D the
    incidence operator of the training substitution graph. That is what makes the 2 x 2 a
    factorial: the site-wise edge model differs from the node model in the loss *and* in
    parameter sharing, in integrability, and in carrying a per-site constant, so it cannot
    occupy a cell.

    A constant response term is worth avoiding on its own account: responses are antisymmetric,
    Delta_{u->v} f = -Delta_{v->u} f, and differencing a node potential can never produce one.

    Differences annihilate the intercept, so it is set afterwards to the training residual mean;
    every response the model predicts is unchanged by that choice.
    """
    from .hybrid import substitution_edges

    nm = NodeModel(descriptors, prop, interactions=True).fit(train_nodes)
    work = train_nodes.dropna(subset=[prop]).reset_index(drop=True)
    X = nm._design(work)
    y = work[prop].to_numpy(dtype=float)
    E = substitution_edges(work)
    if len(E) == 0:
        return nm

    n = len(work)
    if degree_normalised:
        deg = np.bincount(E.ravel(), minlength=n).astype(float)
        w = 1.0 / np.sqrt(np.maximum(deg[E[:, 0]] * deg[E[:, 1]], 1.0))
    else:
        w = np.ones(len(E))
    DX = (X[E[:, 1]] - X[E[:, 0]]) * w[:, None]
    Dy = (y[E[:, 1]] - y[E[:, 0]]) * w

    penalize = np.ones(X.shape[1])
    penalize[nm.blocks_["intercept"]] = 0.0
    g = ridge_solve(DX, Dy, alpha, penalize)
    g[nm.blocks_["intercept"]] = 0.0
    g[nm.blocks_["intercept"]] = float(np.mean(y - X @ g))
    nm.coef_ = g
    return nm


@dataclass
class DirectResponseModel:
    """Delta fitted directly on substitution pairs, with the bilinear context structure.

    Under the low-order multilinear model a substitution at site B, from
    species b to b', in the context (a, x) has response

        Delta = theta_B^T (phi_B' - phi_B)
              + phi_A^T M_AB (phi_B' - phi_B)
              + phi_X^T M_BX (phi_B' - phi_B)

    so the response is *bilinear*: linear in the descriptor difference, and linear in the
    context descriptors, with the two entering only through their product.  The design
    matrix therefore has one block for the difference itself and one Kronecker block
    kron(phi_context, phi_difference) per context site, and the fitted coefficients of
    those blocks reshape directly into theta and the M matrices.  This is the model whose
    identifiability the theory section is about, so the blocks are kept individually
    addressable (`theta`, `interaction_matrix`) rather than being left as one flat vector.

    Two consequences worth stating because they are what the theory has to handle:

    * Each site is fitted separately.  Only the *difference* of the descriptor enters,
      so any component of theta that is constant across a substitution is unidentifiable
      from edges at that site alone — no amount of edge data recovers the absolute level.
    * M_AB is estimated twice, once from B-site edges with A as context and once from
      A-site edges with B as context, and consistency demands the two be transposes of
      one another.  `interaction_symmetry_gap` reports how far apart they land; a large
      gap means the bilinear form is not what the data contains, which is a finding about
      the model class, not a tuning problem.

    An intercept per site is fitted by default.  It is *not* part of the model above
    (which gives Delta = 0 for a null substitution) — it absorbs any mean offset in the
    canonical edge direction and can be switched off with `fit_intercept=False` to test
    the theory as written.
    """

    descriptors: Mapping[str, ElementalDescriptors]
    prop: str
    alpha: float = 1.0
    sites: tuple[str, ...] = SITES
    use_context: bool = True
    fit_intercept: bool = True

    scalers_: dict[str, Standardizer] = field(default_factory=dict, init=False)
    coef_: dict[str, np.ndarray] = field(default_factory=dict, init=False)
    slices_: dict[str, dict[str, slice]] = field(default_factory=dict, init=False)
    blocks_: dict[str, ResponseBlocks] = field(default_factory=dict, init=False)
    n_train_: int = field(default=0, init=False)

    # -- design ------------------------------------------------------------
    def _context_sites(self, site: str) -> tuple[str, ...]:
        if not self.use_context:
            return ()
        return tuple(s for s in SITES if s != site and s in self.descriptors)

    def _register_slices(self, site: str, widths: list[tuple[str, int]]) -> None:
        if site in self.slices_:
            return
        out: dict[str, slice] = {}
        start = 0
        for name, w in widths:
            out[name] = slice(start, start + w)
            start += w
        self.slices_[site] = out

    def _design_for_site(self, site: str, block: pd.DataFrame) -> np.ndarray:
        desc = self.descriptors[site]
        scaler = self.scalers_[site]
        n = len(block)
        phi_src = scaler.transform(desc.matrix(block["src_species"].tolist()))
        phi_dst = scaler.transform(desc.matrix(block["dst_species"].tolist()))
        diff = phi_dst - phi_src

        widths: list[tuple[str, int]] = []
        parts: list[np.ndarray] = []
        if self.fit_intercept:
            parts.append(np.ones((n, 1)))
            widths.append(("intercept", 1))
        parts.append(diff)
        widths.append(("theta", diff.shape[1]))
        for c in self._context_sites(site):
            ctx = self.scalers_[c].transform(
                self.descriptors[c].matrix(block[f"site_{c}"].tolist())
            )
            # row-wise Kronecker product: column (i, j) is ctx_i * diff_j, so the fitted
            # coefficients reshape to M_{c,site} with shape (d_c, d_site) in C order.
            kron = np.einsum("ni,nj->nij", ctx, diff).reshape(n, -1)
            parts.append(kron)
            widths.append((f"M_{c}", kron.shape[1]))
        self._register_slices(site, widths)
        return np.hstack(parts)

    # -- api ---------------------------------------------------------------
    def fit(self, edges: pd.DataFrame) -> "DirectResponseModel":
        acc = accepted(edges)
        acc = acc[acc["delta"].notna()]
        if acc.empty:
            raise ValueError("no accepted edges with a finite delta to fit on")
        if "property" in acc.columns:
            wrong = set(acc["property"].dropna().unique()) - {self.prop}
            if wrong:
                raise ValueError(
                    f"edge frame mixes properties {sorted(wrong)} with {self.prop!r}; "
                    "deltas of different properties are not comparable"
                )

        # Scalers are fitted on the species that actually appear in the training edges,
        # endpoints and contexts alike, and are then frozen: a held-out species must be
        # transformed with training statistics or the span diagnostics that select it
        # stop describing the model being tested.
        labels: dict[str, list[str]] = {s: [] for s in self.descriptors}
        for s in self.descriptors:
            at_site = acc[acc["site"] == s]
            labels[s].extend(at_site["src_species"].dropna().tolist())
            labels[s].extend(at_site["dst_species"].dropna().tolist())
            labels[s].extend(acc[f"site_{s}"].dropna().tolist())
            if not labels[s]:
                labels[s] = list(self.descriptors[s].labels)
        self.scalers_ = _fit_scalers(dict(self.descriptors), labels)

        self.coef_, self.slices_, self.blocks_ = {}, {}, {}
        for site in self.sites:
            block = acc[acc["site"] == site]
            if block.empty:
                continue
            X = self._design_for_site(site, block)
            y = block["delta"].to_numpy(dtype=float)
            penalize = np.ones(X.shape[1])
            if self.fit_intercept:
                penalize[self.slices_[site]["intercept"]] = 0.0
            beta = ridge_solve(X, y, self.alpha, penalize)
            self.coef_[site] = beta
            self.blocks_[site] = self._unpack(site, beta, len(block))
        self.n_train_ = len(acc)
        return self

    def fit_from_nodes(
        self,
        df_train: pd.DataFrame,
        policy: ComparabilityPolicy = ComparabilityPolicy(),
    ) -> "DirectResponseModel":
        """Build substitution pairs from a canonical sample frame, then fit them."""
        return self.fit(build_edges(df_train, self.prop, policy, sites=self.sites))

    def _unpack(self, site: str, beta: np.ndarray, n_edges: int) -> ResponseBlocks:
        sl = self.slices_[site]
        d_site = self.descriptors[site].n_features
        intercept = float(beta[sl["intercept"]][0]) if self.fit_intercept else 0.0
        context = {}
        for c in self._context_sites(site):
            d_c = self.descriptors[c].n_features
            context[c] = beta[sl[f"M_{c}"]].reshape((d_c, d_site))
        return ResponseBlocks(
            site=site,
            n_edges=n_edges,
            intercept=intercept,
            theta=beta[sl["theta"]].copy(),
            context=context,
        )

    def predict_delta(self, edges: pd.DataFrame) -> np.ndarray:
        if not self.coef_:
            raise RuntimeError("DirectResponseModel.fit must be called before predict_delta")
        out = np.full(len(edges), np.nan)
        positions = {idx: i for i, idx in enumerate(edges.index)}
        for site in edges["site"].dropna().unique():
            if site not in self.coef_:
                continue  # no training edges at this site: refuse to invent a response
            block = edges[edges["site"] == site]
            X = self._design_for_site(site, block)
            pred = X @ self.coef_[site]
            for idx, v in zip(block.index, pred):
                out[positions[idx]] = v
        return out

    # -- parameter access --------------------------------------------------
    def theta(self, site: str) -> np.ndarray:
        return self.blocks_[site].theta

    def interaction_matrix(self, site1: str, site2: str) -> np.ndarray:
        """M_{site1,site2}: (d_site1, d_site2), from site2 edges with site1 as context."""
        if site2 not in self.blocks_:
            raise KeyError(f"no fitted response at site {site2!r}")
        blocks = self.blocks_[site2]
        if site1 not in blocks.context:
            raise KeyError(f"site {site1!r} is not a context of site {site2!r}")
        return blocks.context[site1]

    def interaction_symmetry_gap(self, site1: str, site2: str) -> float:
        """Relative Frobenius gap between M_{s1,s2} and M_{s2,s1}^T.

        Zero iff the two independently fitted estimates of the same pair interaction
        agree.  Reported rather than enforced: forcing symmetry during the fit would hide
        precisely the inconsistency that tells us whether the bilinear form holds.
        """
        M12 = self.interaction_matrix(site1, site2)
        M21 = self.interaction_matrix(site2, site1)
        denom = float(np.linalg.norm(M12) + np.linalg.norm(M21))
        if denom == 0:
            return 0.0
        return float(np.linalg.norm(M12 - M21.T) * 2.0 / denom)

    def render(self) -> str:
        lines = [
            "## direct response model",
            f"- property: {self.prop}, ridge alpha: {self.alpha}, "
            f"context terms: {'on' if self.use_context else 'off'}",
            f"- training edges: {self.n_train_}",
        ]
        for site in self.sites:
            if site in self.blocks_:
                lines.append(self.blocks_[site].render())
        for s1, s2 in combinations([s for s in self.sites if s in self.blocks_], 2):
            try:
                gap = self.interaction_symmetry_gap(s1, s2)
            except KeyError:
                continue
            lines.append(f"- symmetry gap M_{s1}{s2} vs M_{s2}{s1}^T: {gap:.4g}")
        return "\n".join(lines)


__all__ = [
    "DifferenceFromNode",
    "DirectResponseModel",
    "NodeModel",
    "ResponseBlocks",
    "edge_endpoint_frames",
    "ridge_solve",
    "fit_node_model_response_loss",
]
