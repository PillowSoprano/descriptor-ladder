"""Element effects as a Gaussian process over descriptor space.

Each site's effect vector carries a Gaussian-process prior with covariance K_s = k(phi_i, phi_j) over
the elements of site s, and each interaction block the product kernel K_s (x) K_t. The penalised
problem is

    min_beta  ||y - X beta||^2  +  tau * sum_s eta_s^T K_s^{-1} eta_s
                               +  tau * sum_pairs vec(M)^T (K_s^{-1} (x) K_t^{-1}) vec(M)

and both regimes of the ladder are evaluations of the same posterior:

  element seen      the fitted coefficient is the posterior mean at that point, the descriptor-
                    similarity shrinkage of `hybrid.fit_stage1(smooth=...)` with K^{-1} in place of
                    the Laplacian
  element not seen  the posterior mean is k(*, obs) K^{-1} eta, a kernel regression from the fitted
                    effects, which is what the descriptor map of `hybrid` computes

The descriptor basis is the same estimator with a degenerate kernel: d descriptors as a basis are
the rank-d linear kernel k(x, y) = x^T y on those features, and `linear_kernel=True` selects it.
A radial kernel on the same features is full rank.

Usage
-----
    m = fit(train_df, "heat_of_formation", desc)          # bandwidths by inner CV
    m.predict(nodes_df)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..descriptors.elemental import ElementalDescriptors
from .hybrid import PAIRS, SITES, _constraints, _design_and_slices, _phi

GAMMAS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
TAUS = tuple(10.0 ** k for k in range(-5, 3))
NUGGET = 1e-4
CV_LEVELS = 4


def _rbf(A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
    d = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return np.exp(-gamma * d)


def _kernel(A: np.ndarray, B: np.ndarray, gamma: float, linear: bool) -> np.ndarray:
    """The kernel, in the two forms the paper's two roles correspond to.

    The degenerate part, a constant plus the linear kernel on the standardised descriptors, has
    rank d+1 and its reproducing space is exactly the span a descriptor *basis* offers, so
    `linear=True` is the basis. Adding a radial term makes the kernel full rank, which is the
    *metric*. The basis is therefore not a different model from the metric; it is this kernel
    with its radial part removed, which is what makes "the basis is a low-rank projection of
    the metric" a statement about code rather than a figure of speech.
    """
    deg = 1.0 + A @ B.T
    return deg if linear else deg + _rbf(A, B, gamma)


@dataclass
class GPEffects:
    """Fitted effects, plus everything needed to evaluate the posterior off the observed set."""

    grand: float
    levels: dict[str, list]
    main: dict[str, np.ndarray]
    inter: dict[str, np.ndarray]
    feat: dict[str, np.ndarray] = field(default_factory=dict)     # standardised descriptors
    norm: dict[str, tuple] = field(default_factory=dict)
    kinv: dict[str, np.ndarray] = field(default_factory=dict)     # K^{-1} on observed levels
    gamma: dict[str, float] = field(default_factory=dict)
    linear: bool = False
    tau: float = 1.0
    n_obs: int = 0

    def index(self, site: str, level) -> Optional[int]:
        try:
            return self.levels[site].index(level)
        except ValueError:
            return None

    def _cross(self, site: str, level) -> np.ndarray:
        """k(level, observed levels), the row that carries a new point into the posterior."""
        P, _, _ = _phi(self._desc[site], [level], *self.norm[site])
        return _kernel(P[:, 1:], self.feat[site], self.gamma[site], self.linear)


def _standardised(desc: ElementalDescriptors, levels: Sequence) -> tuple[np.ndarray, tuple]:
    P, mu, sd = _phi(desc, list(levels))
    return P[:, 1:], (mu, sd)


def _inner_cv_gamma(Z: np.ndarray, y: np.ndarray, linear: bool) -> float:
    """Leave-one-level-out over the observed levels only. Never sees a held-out level."""
    if linear:
        return 0.0
    best, best_e = GAMMAS[0], np.inf
    idx = np.arange(len(y))
    for g in GAMMAS:
        K = _kernel(Z, Z, g, False) + NUGGET * np.eye(len(Z))
        errs = []
        for i in idx:
            k = [j for j in idx if j != i]
            a = np.linalg.solve(K[np.ix_(k, k)], y[k])
            errs.append(y[i] - float(K[i, k] @ a))
        e = float(np.mean(np.square(errs)))
        if e < best_e:
            best, best_e = g, e
    return best


def fit(
    df: pd.DataFrame, prop: str, desc: Mapping[str, ElementalDescriptors],
    lam: float = 1e-3, tau: Optional[float] = None, linear_kernel: bool = False,
    gammas: Optional[dict[str, float]] = None,
) -> "GPModel":
    """Fit the whole estimator as one penalised problem. No routing anywhere."""
    work = df.dropna(subset=[prop])
    levels = {s: sorted(work[f"site_{s}"].dropna().unique().tolist()) for s in SITES}
    X, sl = _design_and_slices(work, levels)
    y = work[prop].to_numpy(dtype=float)
    C = _constraints(levels, sl, X.shape[1])

    feat, norm, Kinv, gam = {}, {}, {}, {}
    for s in SITES:
        Z, nz = _standardised(desc[s], levels[s])
        feat[s], norm[s] = Z, nz

    # a first pass with an uninformative penalty gives effects to tune the bandwidth on;
    # it is fitted on training data only, so nothing held out is involved
    w = np.sqrt(max(len(work), 1)) * 10.0
    G0 = (X.T @ X).toarray() + (w ** 2) * (C.T @ C).toarray() + 1e-2 * np.eye(X.shape[1])
    b0 = np.linalg.lstsq(G0, X.T @ y, rcond=None)[0]
    for s in SITES:
        g = (gammas or {}).get(s)
        if g is None:
            g = _inner_cv_gamma(feat[s], b0[sl[s]], linear_kernel)
        gam[s] = g
        K = _kernel(feat[s], feat[s], g, linear_kernel)
        Kinv[s] = np.linalg.pinv(K + NUGGET * np.eye(len(K)))

    def _solve(t: float) -> np.ndarray:
        G = (X.T @ X).toarray() + (w ** 2) * (C.T @ C).toarray() + lam * np.eye(X.shape[1])
        for s_ in SITES:
            G[sl[s_], sl[s_]] += t * Kinv[s_]
        for p_ in PAIRS:
            G[sl[p_], sl[p_]] += t * np.kron(Kinv[p_[0]], Kinv[p_[1]])
        return np.linalg.lstsq(G, X.T @ y, rcond=None)[0]

    if tau is not None:
        best, best_beta = float(tau), _solve(float(tau))
    else:
        # Inner cross-validation that holds out a training *level*, not a cell. The regime
        # this estimator is judged in is an unseen level, and tuning against held-out cells
        # would be tuning for interpolation and reporting on extrapolation.
        rng = np.random.default_rng(0)
        folds = []
        for s_ in SITES:
            if len(levels[s_]) < 4:
                continue
            for lv in rng.choice(levels[s_], min(CV_LEVELS, len(levels[s_])), replace=False):
                folds.append((s_, lv))
        best, best_score = TAUS[0], np.inf
        for t in TAUS:
            errs = []
            for s_, lv in folds:
                inner = work[work[f"site_{s_}"] != lv]
                held = work[work[f"site_{s_}"] == lv]
                if inner.empty or held.empty:
                    continue
                m = fit(inner, prop, desc, lam=lam, tau=t, linear_kernel=linear_kernel,
                        gammas=gam)
                errs.append(float(np.mean(np.abs(
                    m.predict(held) - held[prop].to_numpy(float)))))
            sc = float(np.mean(errs)) if errs else np.inf
            if sc < best_score:
                best, best_score = t, sc
        best_beta = _solve(best)

    eff = GPEffects(
        grand=float(best_beta[0]), levels=levels,
        main={s: best_beta[sl[s]].copy() for s in SITES},
        inter={p: best_beta[sl[p]].reshape(len(levels[p[0]]), len(levels[p[1]])).copy()
               for p in PAIRS},
        feat=feat, norm=norm, kinv=Kinv, gamma=gam, linear=linear_kernel,
        tau=float(best), n_obs=len(work))
    eff._desc = desc                                              # for the posterior evaluation
    return GPModel(eff, desc)


@dataclass
class GPModel:
    eff: GPEffects
    desc: Mapping[str, ElementalDescriptors]

    # -- the posterior, evaluated wherever it is asked for -------------------------------
    def _w(self, site: str, level) -> tuple[Optional[int], Optional[np.ndarray]]:
        """Either the index of an observed level, or the weights that carry a new one in."""
        i = self.eff.index(site, level)
        if i is not None:
            return i, None
        P, _, _ = _phi(self.desc[site], [level], *self.eff.norm[site])
        k = _kernel(P[:, 1:], self.eff.feat[site], self.eff.gamma[site], self.eff.linear)
        return None, np.ravel(k @ self.eff.kinv[site])

    def _main(self, site: str, level) -> float:
        i, w = self._w(site, level)
        v = self.eff.main[site]
        return float(v[i]) if i is not None else float(w @ v)

    def _inter(self, pair: str, l1, l2) -> float:
        s1, s2 = pair[0], pair[1]
        M = self.eff.inter[pair]
        i, w1 = self._w(s1, l1)
        j, w2 = self._w(s2, l2)
        # the conditional only extrapolates the margin that is missing; an observed partner
        # keeps its own estimated row rather than being pushed through descriptors as well
        if i is not None and j is not None:
            return float(M[i, j])
        if i is None and j is not None:
            return float(w1 @ M[:, j])
        if i is not None and j is None:
            return float(M[i, :] @ w2)
        return float(w1 @ M @ w2)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        out = np.empty(len(df))
        for k, row in enumerate(df.itertuples(index=False)):
            v = self.eff.grand
            for s in SITES:
                v += self._main(s, getattr(row, f"site_{s}"))
            for p in PAIRS:
                v += self._inter(p, getattr(row, f"site_{p[0]}"), getattr(row, f"site_{p[1]}"))
            out[k] = v
        return out
