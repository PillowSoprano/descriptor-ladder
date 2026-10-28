"""The shared-trend model with element corrections.

Free element effects are identified for every element that appears in training, and descriptors
add no capacity there, because a descriptor column is a linear combination of the element
indicators. For an element that never appears, no free effect exists and the descriptors must carry
the prediction. Putting both into one regression leaves the descriptor coefficients unidentified,
so the model is fitted in two steps:

    element effects   free main effects and pair interactions on the elements seen in training,
                      optionally under the similarity prior (`fit_stage1`)
    descriptor map    a regression of the fitted effects on the element descriptors, across
                      elements (`fit_stage2`)
    prediction        the free effect for an element seen in training, the descriptor map for an
                      element that was not

The descriptor map has one observation per element, so its sample size is the number of elements,
not the number of compositions. `transfer_cv` scores it by leaving each element out in turn and
refitting the element effects without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..descriptors.elemental import ElementalDescriptors
from ..schema import SITES

PAIRS = tuple(a + b for a, b in combinations(SITES, 2))


def parameter_dimensions(levels: Mapping[str, Sequence]) -> tuple[int, int, int]:
    """Raw, identifying-constraint and identifiable dimensions of the element-effect fit.

    The raw count includes the intercept.  Main-effect sum constraints contribute one
    independent constraint per site.  An ``n_s x n_t`` interaction block has
    ``n_s + n_t - 1`` independent zero-margin constraints because the row and column sums
    share one redundancy.
    """
    sizes = {s: len(levels[s]) for s in SITES}
    raw = 1 + sum(sizes.values())
    raw += sum(sizes[p[0]] * sizes[p[1]] for p in PAIRS)
    constraints = len(SITES)
    constraints += sum(sizes[p[0]] + sizes[p[1]] - 1 for p in PAIRS)
    return raw, constraints, raw - constraints


def _ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    G = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.lstsq(G, X.T @ y, rcond=None)[0]


@dataclass
class Stage1Effects:
    """Sum-to-zero categorical effects estimated on the observed cells."""

    grand: float
    levels: dict[str, list]
    main: dict[str, np.ndarray]            # site -> (n_levels,)
    inter: dict[str, np.ndarray]           # "AB" -> (n_A, n_B)
    n_obs: int
    rank_deficient: bool
    # Filled only when fit_stage1(with_variance=True). These are the diagonal of the ridge
    # posterior covariance sigma^2 G^-1, so they say how well each effect is pinned by the
    # data, and they are what the descriptor map needs in order to stop treating a level estimated from
    # four cells as being as trustworthy as one estimated from ninety.
    main_var: Optional[dict[str, np.ndarray]] = None
    inter_var: Optional[dict[str, np.ndarray]] = None

    def index(self, site: str, level) -> Optional[int]:
        try:
            return self.levels[site].index(level)
        except ValueError:
            return None


@dataclass
class Stage2Maps:
    """Descriptor meta-regressions, one per main effect and one per interaction block."""

    main_coef: dict[str, np.ndarray] = field(default_factory=dict)
    main_r2: dict[str, float] = field(default_factory=dict)
    inter_coef: dict[str, np.ndarray] = field(default_factory=dict)
    inter_r2: dict[str, float] = field(default_factory=dict)
    # the kernel path stores dual coefficients and the training features instead of coefs
    gamma: Optional[float] = None
    combined: bool = False
    main_dual: dict[str, np.ndarray] = field(default_factory=dict)
    inter_dual: dict[str, np.ndarray] = field(default_factory=dict)
    train_feat: dict[str, np.ndarray] = field(default_factory=dict)

    def render(self) -> str:
        lines = ["| block | meta-regression R^2 |", "|---|---|"]
        for k in sorted(self.main_r2):
            lines.append(f"| {k} main | {self.main_r2[k]:.3f} |")
        for k in sorted(self.inter_r2):
            lines.append(f"| {k} interaction | {self.inter_r2[k]:.3f} |")
        return "\n".join(lines)


def _design_and_slices(df: pd.DataFrame, levels: dict[str, list]):
    """Sparse one-hot design for mains and pairs.

    Each row has exactly 7 non-zeros (intercept, three mains, three pairs), so the dense
    form would be 18,928 x 3,544 = 537 MB of almost entirely zeros on the full Castelli
    grid.  The normal equations are formed from the sparse design instead.
    """
    n = len(df)
    idx = {s: df[f"site_{s}"].map({l: i for i, l in enumerate(levels[s])}).to_numpy() for s in SITES}
    sizes = {s: len(levels[s]) for s in SITES}
    slices, off = {}, 1
    cols = [np.zeros(n, dtype=np.int64)]
    for s in SITES:
        cols.append(off + idx[s])
        slices[s] = slice(off, off + sizes[s])
        off += sizes[s]
    for p in PAIRS:
        s1, s2 = p[0], p[1]
        cols.append(off + idx[s1] * sizes[s2] + idx[s2])
        slices[p] = slice(off, off + sizes[s1] * sizes[s2])
        off += sizes[s1] * sizes[s2]
    col = np.concatenate(cols)
    row = np.tile(np.arange(n), len(cols))
    X = sp.csr_matrix((np.ones(len(col)), (row, col)), shape=(n, off))
    return X, slices


def _constraints(levels: dict[str, list], slices: dict, n_cols: int):
    """Sum-to-zero on every effect block, and on both margins of every interaction."""
    sizes = {s: len(levels[s]) for s in SITES}
    rows, cols = [], []
    r = 0
    for s in SITES:
        sl = slices[s]
        cols.extend(range(sl.start, sl.stop)); rows.extend([r] * sizes[s]); r += 1
    for p in PAIRS:
        s1, s2 = p[0], p[1]
        base = slices[p].start
        for i in range(sizes[s1]):
            cols.extend(base + i * sizes[s2] + j for j in range(sizes[s2]))
            rows.extend([r] * sizes[s2]); r += 1
        for j in range(sizes[s2]):
            cols.extend(base + i * sizes[s2] + j for i in range(sizes[s1]))
            rows.extend([r] * sizes[s1]); r += 1
    return sp.csr_matrix((np.ones(len(cols)), (rows, cols)), shape=(r, n_cols))


def level_laplacian(desc: ElementalDescriptors, labels: Sequence, k: int = 3) -> np.ndarray:
    """Graph Laplacian of a k-nearest-neighbour graph on levels in descriptor space.

    This is the object a categorical parameterisation lacks. A free effect per level has no
    notion of which levels resemble which, so an interaction cell that was never observed can
    only be filled in by the penalty and the margin constraints. A tree on descriptors fills it
    from cells whose levels are nearby. Penalising beta^T L beta puts that metric back without
    giving up any capacity: the model can still fit every observed cell exactly.
    """
    M = np.nan_to_num(np.asarray(desc.matrix(list(labels), allow_missing=True), float))
    Z = (M - M.mean(0)) / np.where(M.std(0) > 0, M.std(0), 1.0)
    D = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    n = len(labels)
    W = np.zeros((n, n))
    kk = min(k, n - 1)
    for i in range(n):
        for j in np.argsort(D[i])[1:kk + 1]:
            W[i, j] = W[j, i] = 1.0
    return np.diag(W.sum(1)) - W


def fit_stage1(
    df: pd.DataFrame, prop: str, lam: float = 1e-3,
    levels: Optional[dict[str, list]] = None,
    smooth: float = 0.0, desc: Optional[Mapping[str, ElementalDescriptors]] = None,
    knn: int = 3, adaptive: bool = False, with_variance: bool = False,
) -> Stage1Effects:
    work = df.dropna(subset=[prop])
    levels = levels or {s: sorted(work[f"site_{s}"].dropna().unique().tolist()) for s in SITES}
    X, sl = _design_and_slices(work, levels)
    y = work[prop].to_numpy(dtype=float)
    C = _constraints(levels, sl, X.shape[1])

    # Constraints, not priors: weighted far above the data so they are satisfied rather
    # than traded off against fit.
    w = np.sqrt(max(len(work), 1)) * 10.0
    G = (X.T @ X).toarray() + (w ** 2) * (C.T @ C).toarray() + lam * np.eye(X.shape[1])
    if smooth > 0 and desc is not None:
        L = {s_: level_laplacian(desc[s_], levels[s_], knn) for s_ in SITES}
        if adaptive:
            # The graded gate. The prior is worth more where a level is thinly observed and
            # less where its effect is already pinned by data, so each Laplacian edge (i,j)
            # is weighted by nbar / sqrt(n_i n_j) with n the observation count of the level.
            # At uniform coverage this is the identity and the constant matches the fixed
            # version, so the two agree on a complete grid by construction.
            for s_ in SITES:
                cnt = work[f"site_{s_}"].value_counts()
                n = np.array([max(int(cnt.get(l, 0)), 1) for l in levels[s_]], dtype=float)
                w = float(n.mean()) / np.sqrt(np.outer(n, n))
                A_ = -L[s_] * (1 - np.eye(len(n)))          # adjacency
                A_ = A_ * w
                L[s_] = np.diag(A_.sum(1)) - A_
        for s_ in SITES:
            b = sl[s_]
            G[b, b] += smooth * L[s_]
        for p_ in PAIRS:
            s1, s2 = p_[0], p_[1]
            n1, n2 = len(levels[s1]), len(levels[s2])
            # index is i * n2 + j, so the row-major Kronecker order is (s1 outer, s2 inner)
            b = sl[p_]
            G[b, b] += smooth * (np.kron(L[s1], np.eye(n2)) + np.kron(np.eye(n1), L[s2]))
    beta = np.linalg.lstsq(G, X.T @ y, rcond=None)[0]

    mvar = ivar = None
    if with_variance:
        # sigma^2 G^-1, the ridge posterior covariance. The effective degrees of freedom are
        # tr(X G^-1 X^T), not the column count, or a design with more columns than rows would
        # give a negative residual variance.
        Ginv = np.linalg.pinv(G)
        resid = np.asarray(y - X @ beta).ravel()
        # tr(X G^-1 X^T) = tr(G^-1 X^T X), which avoids forming the n x n hat matrix
        edf = float(np.sum(Ginv * (X.T @ X).toarray()))
        dof = max(len(y) - edf, 1.0)
        sigma2 = float(resid @ resid) / dof
        d = sigma2 * np.clip(np.diag(Ginv), 0.0, None)
        mvar = {s: d[sl[s]].copy() for s in SITES}
        ivar = {p: d[sl[p]].reshape(len(levels[p[0]]), len(levels[p[1]])).copy() for p in PAIRS}

    return Stage1Effects(
        grand=float(beta[0]),
        levels=levels,
        main={s: beta[sl[s]].copy() for s in SITES},
        inter={p: beta[sl[p]].reshape(len(levels[p[0]]), len(levels[p[1]])).copy() for p in PAIRS},
        n_obs=len(work),
        rank_deficient=bool(np.linalg.matrix_rank(G, tol=1e-8) < G.shape[0]),
        main_var=mvar, inter_var=ivar,
    )


def substitution_edges(df: pd.DataFrame) -> np.ndarray:
    """Every single-site substitution among the given nodes, each undirected pair once.

    Grouping by the two unchanged sites and taking all pairs inside a group enumerates the
    A-, B- and X-substitutions exactly once each. On the 14 x 14 x 7 subgrid that is about
    8.9k A-edges, 8.9k B-edges and 4.1k X-edges, so the three types are *not* equally
    represented; `fit_stage1_response` weights every edge alike unless told otherwise, and
    the imbalance is a property of the grid rather than a modelling choice.
    """
    pos = {}
    for k, r in enumerate(df.itertuples(index=False)):
        pos[(r.site_A, r.site_B, r.site_X)] = k
    groups: dict[tuple, list[int]] = {}
    for (a, b, x), k in pos.items():
        groups.setdefault(("A", b, x), []).append(k)
        groups.setdefault(("B", a, x), []).append(k)
        groups.setdefault(("X", a, b), []).append(k)
    rows = []
    for members in groups.values():
        for ii in range(len(members)):
            for jj in range(ii + 1, len(members)):
                rows.append((members[ii], members[jj]))
    return np.asarray(rows, dtype=np.int64) if rows else np.zeros((0, 2), dtype=np.int64)


def fit_stage1_response(
    df: pd.DataFrame, prop: str, lam: float = 1e-3,
    levels: Optional[dict[str, list]] = None, degree_normalised: bool = False,
) -> Stage1Effects:
    """The element-effect fit under the response loss: minimise ||D(y - X b)||^2 rather than ||y - X b||^2.

    D is the incidence operator of the training substitution graph, so D^T D is its
    Laplacian and the normal equations use X^T L X and X^T L y. This is not a change of
    prediction target: the Laplacian reweights residuals by how connected a node is, and on
    an incomplete grid that is not equivalent to the node sum of squares. Every other
    ingredient, the categorical design, the sum-to-zero constraints, the constraint weight
    and the ridge, is identical to `fit_stage1`, so the two differ in the loss and in
    nothing else.

    `degree_normalised` divides each edge by sqrt(deg_u deg_v), which is the sensitivity
    check against the result being driven by high-degree nodes being counted more often.

    Differences annihilate the intercept, so the grand mean is not identified by this loss;
    it is set afterwards to the training residual mean, which leaves every response the
    model predicts unchanged.
    """
    work = df.dropna(subset=[prop]).reset_index(drop=True)
    levels = levels or {s: sorted(work[f"site_{s}"].dropna().unique().tolist()) for s in SITES}
    X, sl = _design_and_slices(work, levels)
    y = work[prop].to_numpy(dtype=float)
    C = _constraints(levels, sl, X.shape[1])
    E = substitution_edges(work)
    if len(E) == 0:
        return fit_stage1(df, prop, lam, levels)

    n = len(work)
    if degree_normalised:
        deg = np.bincount(E.ravel(), minlength=n).astype(float)
        wt = 1.0 / np.sqrt(np.maximum(deg[E[:, 0]] * deg[E[:, 1]], 1.0))
    else:
        wt = np.ones(len(E))
    D = sp.csr_matrix(
        (np.concatenate([-wt, wt]),
         (np.concatenate([np.arange(len(E))] * 2),
          np.concatenate([E[:, 0], E[:, 1]]))), shape=(len(E), n))
    DX, Dy = D @ X, D @ y

    w = np.sqrt(max(n, 1)) * 10.0
    G = (DX.T @ DX).toarray() + (w ** 2) * (C.T @ C).toarray() + lam * np.eye(X.shape[1])
    beta = np.linalg.lstsq(G, DX.T @ Dy, rcond=None)[0]
    beta[0] = 0.0
    beta[0] = float(np.mean(y - X @ beta))

    return Stage1Effects(
        grand=float(beta[0]),
        levels=levels,
        main={s: beta[sl[s]].copy() for s in SITES},
        inter={p: beta[sl[p]].reshape(len(levels[p[0]]), len(levels[p[1]])).copy() for p in PAIRS},
        n_obs=n,
        rank_deficient=bool(np.linalg.matrix_rank(G, tol=1e-8) < G.shape[0]),
    )


def _phi(desc: ElementalDescriptors, labels: Sequence, mean=None, std=None):
    M = desc.matrix(list(labels), allow_missing=True)
    M = np.nan_to_num(np.asarray(M, dtype=float), nan=0.0)
    if mean is None:
        mean, std = M.mean(0), np.where(M.std(0) > 0, M.std(0), 1.0)
    Z = (M - mean) / std
    return np.hstack([np.ones((len(labels), 1)), Z]), mean, std


def _wridge(X: np.ndarray, Y: np.ndarray, lam: float, w: Optional[np.ndarray]) -> np.ndarray:
    """Ridge, optionally weighted by w on the rows. w=None is exactly `_ridge`."""
    if w is None:
        return _ridge(X, Y, lam)
    W = np.asarray(w, float).ravel()
    Xw = X * W[:, None]
    return np.linalg.lstsq(X.T @ Xw + lam * np.eye(X.shape[1]), Xw.T @ Y, rcond=None)[0]


def _rbf(A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
    d = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return np.exp(-gamma * d)


def _kmat(A: np.ndarray, B: np.ndarray, gamma: float, combined: bool) -> np.ndarray:
    """Either the radial kernel alone, or it plus the degenerate one a basis amounts to.

    A radial kernel is a local average and cannot leave the range of the values it averages,
    which is why it fails on a level whose effect is extreme. The degenerate part 1 + x'y is
    exactly the descriptor basis, and it extrapolates. Neither alone matches the sum on the
    transfer measurement, so the combination is not a hedge, it is two different jobs.
    """
    return _rbf(A, B, gamma) + (1.0 + A @ B.T if combined else 0.0)


def fit_stage2(
    eff: Stage1Effects, desc: Mapping[str, ElementalDescriptors], lam: float = 1e-2,
    source: Optional[Stage1Effects] = None,
    gamma: Optional[float] = None,
    precision_weighted: bool = False,
    combined: bool = False,
) -> tuple[Stage2Maps, dict]:
    """Regress estimated effects on level descriptors, across levels.

    `source` lets the descriptor map be learned from a different, usually larger, set of
    levels than the element effects were fitted on. The map from element properties to site effects is a
    chemistry-level object and does not have to be trained on whichever levels happen to sit
    in the evaluation subgrid; the descriptor map on thirteen levels is eleven parameters fitted to
    thirteen points. Nothing about the held-out level may enter `source`; `fit` enforces that.

    `gamma` replaces the linear map by kernel ridge with an RBF kernel of that bandwidth, on
    the same standardised descriptor features and with the same penalty. The default stays
    linear. With a linear kernel the dual solution is the primal one, so the two differ in the
    kernel and in nothing else, which is what makes the comparison a test of non-linearity
    rather than a comparison of two unrelated estimators.

    `precision_weighted` weights each level by the reciprocal of its element-effect posterior
    variance, so an effect the design pinned down counts for more than one it barely
    identified. It needs `fit_stage1(with_variance=True)` and is ignored otherwise.
    """
    if source is not None:
        eff = source
    maps, norm = Stage2Maps(), {}
    P = {}
    for s in SITES:
        P[s], mu, sd = _phi(desc[s], eff.levels[s])
        norm[s] = (mu, sd)

    wt: dict[str, Optional[np.ndarray]] = {s: None for s in SITES}
    if precision_weighted and eff.main_var is not None:
        for s in SITES:
            # normalised to mean one so the penalty keeps its scale, and clipped so a level
            # the design happened to pin almost exactly cannot dominate the fit
            iv = 1.0 / np.maximum(np.asarray(eff.main_var[s], float), 1e-12)
            wt[s] = np.clip(iv / iv.mean(), 1e-3, 1e3)

    if gamma is not None:
        maps.gamma = float(gamma)
        maps.combined = bool(combined)
        for s in SITES:
            Z = P[s][:, 1:]                     # drop the intercept; the kernel carries it
            maps.train_feat[s] = Z
            K = _kmat(Z, Z, gamma, combined)
            y = eff.main[s]
            a_ = np.linalg.lstsq(K + lam * np.eye(len(K)), y, rcond=None)[0]
            maps.main_dual[s] = a_
            resid = y - K @ a_
            maps.main_r2[s] = float(1 - resid.var() / y.var()) if y.var() > 0 else float("nan")
        for p in PAIRS:
            s1, s2 = p[0], p[1]
            M = eff.inter[p]
            K1 = _kmat(maps.train_feat[s1], maps.train_feat[s1], gamma, combined)
            K2 = _kmat(maps.train_feat[s2], maps.train_feat[s2], gamma, combined)
            A = np.linalg.lstsq(K1 + lam * np.eye(len(K1)), M, rcond=None)[0]
            Ad = np.linalg.lstsq(K2 + lam * np.eye(len(K2)), A.T, rcond=None)[0].T
            maps.inter_dual[p] = Ad
            resid = M - K1 @ Ad @ K2
            maps.inter_r2[p] = float(1 - (resid ** 2).sum() / (M ** 2).sum()) \
                if (M ** 2).sum() > 0 else float("nan")
        return maps, norm

    for s in SITES:
        y = eff.main[s]
        c = _wridge(P[s], y, lam, wt[s])
        maps.main_coef[s] = c
        resid = y - P[s] @ c
        maps.main_r2[s] = float(1 - resid.var() / y.var()) if y.var() > 0 else float("nan")
    for p in PAIRS:
        s1, s2 = p[0], p[1]
        M = eff.inter[p]
        # vec(M) ~ (P1 kron P2) vec(W): solved as the two-sided projection, which is the
        # same estimator and avoids materialising a (n1*n2) x (d1*d2) design.
        A = _wridge(P[s1], M, lam, wt[s1])
        W = _wridge(P[s2], A.T, lam, wt[s2]).T
        maps.inter_coef[p] = W
        resid = M - P[s1] @ W @ P[s2].T
        maps.inter_r2[p] = float(1 - (resid ** 2).sum() / (M ** 2).sum()) if (M ** 2).sum() > 0 else float("nan")
    return maps, norm


@dataclass
class HybridModel:
    stage1: Stage1Effects
    stage2: Stage2Maps
    norm: dict
    desc: Mapping[str, ElementalDescriptors]

    def _feat(self, site: str, level) -> np.ndarray:
        P, _, _ = _phi(self.desc[site], [level], *self.norm[site])
        return P

    def _main_for(self, site: str, level) -> float:
        i = self.stage1.index(site, level)
        if i is not None:
            return float(self.stage1.main[site][i])
        P = self._feat(site, level)
        if self.stage2.gamma is not None:
            k = _kmat(P[:, 1:], self.stage2.train_feat[site], self.stage2.gamma,
                      self.stage2.combined)
            return float(np.ravel(k @ self.stage2.main_dual[site])[0])
        return float(np.ravel(P @ self.stage2.main_coef[site])[0])

    def _inter_for(self, pair: str, l1, l2) -> float:
        s1, s2 = pair[0], pair[1]
        i, j = self.stage1.index(s1, l1), self.stage1.index(s2, l2)
        if i is not None and j is not None:
            return float(self.stage1.inter[pair][i, j])
        P1, P2 = self._feat(s1, l1), self._feat(s2, l2)
        if self.stage2.gamma is not None:
            g = self.stage2.gamma
            k1 = _kmat(P1[:, 1:], self.stage2.train_feat[s1], g, self.stage2.combined)
            k2 = _kmat(P2[:, 1:], self.stage2.train_feat[s2], g, self.stage2.combined)
            return float(np.ravel(k1 @ self.stage2.inter_dual[pair] @ k2.T)[0])
        return float(np.ravel(P1 @ self.stage2.inter_coef[pair] @ P2.T)[0])

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        out = np.full(len(df), self.stage1.grand)
        for k, row in enumerate(df.itertuples(index=False)):
            v = self.stage1.grand
            for s in SITES:
                v += self._main_for(s, getattr(row, f"site_{s}"))
            for p in PAIRS:
                v += self._inter_for(p, getattr(row, f"site_{p[0]}"), getattr(row, f"site_{p[1]}"))
            out[k] = v
        return out


def fit(df: pd.DataFrame, prop: str, desc: Mapping[str, ElementalDescriptors],
        lam1: float = 1e-3, lam2: float = 1e-2,
        donor: Optional[pd.DataFrame] = None,
        forbid: Sequence[tuple[str, object]] = (),
        gamma: Optional[float] = None, combined: bool = False) -> HybridModel:
    """Fit both steps, optionally learning the descriptor map from a wider set of elements.

    `donor` is an extra frame of compositions used only for the descriptor map. `forbid` lists the
    (site, level) pairs being held out; every one of them is checked to be absent from the
    donor frame, because a donor that still contains the held-out level would make the
    transfer number meaningless and the mistake would be invisible in the output.
    """
    eff = fit_stage1(df, prop, lam1)
    source = None
    if donor is not None:
        for site, level in forbid:
            leaked = int((donor[f"site_{site}"] == level).sum())
            if leaked:
                raise ValueError(
                    f"donor frame contains {leaked} rows with the held-out level "
                    f"{site}={level!r}; the descriptor map would see what it is meant to predict")
        source = fit_stage1(donor, prop, lam1)
    maps, norm = fit_stage2(eff, desc, lam2, source=source, gamma=gamma, combined=combined)
    return HybridModel(eff, maps, norm, desc)


def fit_response(df: pd.DataFrame, prop: str, desc: Mapping[str, ElementalDescriptors],
                 lam1: float = 1e-3, lam2: float = 1e-2,
                 degree_normalised: bool = False,
                 donor: Optional[pd.DataFrame] = None,
                 forbid: Sequence[tuple[str, object]] = ()) -> HybridModel:
    """The response-loss counterpart of `fit`: same parameterisation, same fallback.

    The descriptor map is fitted to the response-loss effects, so the whole estimator is the matched
    comparison. Without the fallback a purely categorical response model could not predict an
    unseen level at all, and the leave-one-element-out designs would drop out of the comparison.
    """
    eff = fit_stage1_response(df, prop, lam1, degree_normalised=degree_normalised)
    source = None
    if donor is not None:
        for site, level in forbid:
            leaked = int((donor[f"site_{site}"] == level).sum())
            if leaked:
                raise ValueError(f"donor frame contains {leaked} rows with {site}={level!r}")
        source = fit_stage1_response(donor, prop, lam1, degree_normalised=degree_normalised)
    maps, norm = fit_stage2(eff, desc, lam2, source=source)
    return HybridModel(eff, maps, norm, desc)


def transfer_cv(
    df: pd.DataFrame, prop: str, site: str, desc: Mapping[str, ElementalDescriptors],
    levels: Optional[Sequence] = None, lam1: float = 1e-3, lam2: float = 1e-2,
) -> pd.DataFrame:
    """Leave-one-level-out cross-fitting of the transfer step.

    The element-effect fit is redone without the held-out level, so the effects the meta-regression
    trains on carry no information about it.  Refitting only the descriptor map would leak: the
    held-out level's own estimated effect would have been in the element-effect solution.
    """
    levels = list(levels) if levels is not None else sorted(df[f"site_{site}"].dropna().unique())
    rows = []
    for lv in levels:
        train = df[df[f"site_{site}"] != lv]
        test = df[df[f"site_{site}"] == lv]
        if train.empty or test.empty:
            continue
        model = fit(train, prop, desc, lam1, lam2)
        pred = model.predict(test)
        true = test[prop].to_numpy(dtype=float)
        # what the true free effect would have been, for a direct look at the transfer step
        full = fit_stage1(df, prop, lam1)
        i = full.index(site, lv)
        rows.append({
            "site": site, "level": lv, "n_test": len(test),
            "rmse": float(np.sqrt(np.mean((true - pred) ** 2))),
            "mae": float(np.mean(np.abs(true - pred))),
            "true_main_effect": float(full.main[site][i]) if i is not None else np.nan,
            "pred_main_effect": model._main_for(site, lv),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["main_effect_error"] = out["pred_main_effect"] - out["true_main_effect"]
    return out
