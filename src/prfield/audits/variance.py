"""How much of a property is main effects, pair interactions and the three-site term.

On a complete product grid the three-factor ANOVA is exact, with no estimation, regularisation or
model selection, so the variance shares are properties of the data. `incomplete_anova` handles
grids with missing cells.

Sum-to-zero (effect) coding throughout, so the terms are orthogonal on a complete grid and the
shares add to one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd

FACTORS = ("A", "B", "X")


MEASURES = {
    # Each cell of the grid counts once, duplicates (polymorphs) averaged within a cell:
    # a ~ Unif(A), b ~ Unif(B), x ~ Unif(X), independent.  This is the only weighting
    # under which the decomposition is orthogonal and the shares are exact.
    "uniform_product": "a ~ Unif(A), b ~ Unif(B), x ~ Unif(X), independent; duplicates averaged per cell",
    # Every observed row counts once, so a composition with 17 polymorphs outweighs one
    # with 1.  Not orthogonal, and the shares mix chemical with structural variation.
    "empirical_rows": "each observed row weighted equally; compositions with more polymorphs dominate",
}


@dataclass
class VarianceDecomposition:
    property: str
    n_obs: int
    grid_shape: tuple
    complete: bool
    total_variance: float
    shares: dict[str, float] = field(default_factory=dict)
    censored_fraction: float = 0.0
    measure: str = "uniform_product"
    notes: list[str] = field(default_factory=list)

    @property
    def main(self) -> float:
        return sum(v for k, v in self.shares.items() if len(k) == 1)

    @property
    def pair(self) -> float:
        return sum(v for k, v in self.shares.items() if len(k) == 2)

    @property
    def triple(self) -> float:
        return self.shares.get("ABX", 0.0)

    def render(self) -> str:
        lines = [
            f"### {self.property}",
            f"- observations: {self.n_obs}, grid {self.grid_shape}, "
            f"{'complete (decomposition is exact)' if self.complete else 'incomplete (least squares)'}",
            f"- measure: **{self.measure}** — {MEASURES.get(self.measure, 'custom')}",
            f"- total variance: {self.total_variance:.4g}",
        ]
        if self.censored_fraction > 0.01:
            lines.append(
                f"- **censored: {self.censored_fraction:.1%} of values sit exactly at the "
                "boundary** — variance shares below describe a mostly-degenerate field"
            )
        for k in sorted(self.shares, key=lambda s: (len(s), s)):
            lines.append(f"  - {k:>4}: {self.shares[k]:6.1%}")
        lines.append(
            f"- main {self.main:.1%} | pairs {self.pair:.1%} | triple {self.triple:.1%}"
        )
        for n in self.notes:
            lines.append(f"- note: {n}")
        return "\n".join(lines)


def tensor_from_frame(
    df: pd.DataFrame, prop: str, factors: tuple = FACTORS
) -> tuple[np.ndarray, np.ndarray, dict[str, list]]:
    """(values, mask, levels).  Duplicate cells are averaged and reported by the caller."""
    cols = [f"site_{f}" for f in factors]
    levels = {f: sorted(df[c].dropna().unique().tolist()) for f, c in zip(factors, cols)}
    index = {f: {lab: i for i, lab in enumerate(levels[f])} for f in factors}
    shape = tuple(len(levels[f]) for f in factors)

    acc = np.zeros(shape)
    cnt = np.zeros(shape)
    for row in df[cols + [prop]].dropna(subset=[prop]).itertuples(index=False):
        idx = tuple(index[f][getattr(row, f"site_{f}")] for f in factors)
        acc[idx] += float(getattr(row, prop))
        cnt[idx] += 1
    with np.errstate(invalid="ignore"):
        vals = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    return vals, cnt > 0, levels


def complete_grid_anova(vals: np.ndarray, factors: tuple = FACTORS) -> dict[str, float]:
    """Exact orthogonal decomposition of a fully observed d-way array."""
    d = vals.ndim
    axes = list(range(d))
    grand = vals.mean()
    centred = vals - grand

    effects: dict[str, np.ndarray] = {}
    # main effects: mean over all other axes
    for i, f in enumerate(factors):
        other = tuple(a for a in axes if a != i)
        effects[f] = centred.mean(axis=other, keepdims=True)
    # pair interactions: two-way mean minus the two mains
    for (i, fi), (j, fj) in combinations(list(enumerate(factors)), 2):
        other = tuple(a for a in axes if a not in (i, j))
        m2 = centred.mean(axis=other, keepdims=True) if other else centred
        effects[fi + fj] = m2 - effects[fi] - effects[fj]
    # triple: whatever is left
    resid = centred.copy()
    for term in effects.values():
        resid = resid - term
    effects["".join(factors)] = resid

    ss_total = float((centred ** 2).sum())
    shares = {}
    for name, term in effects.items():
        ss = float((np.broadcast_to(term, vals.shape) ** 2).sum())
        shares[name] = ss / ss_total if ss_total > 0 else float("nan")
    return shares


def _design(df: pd.DataFrame, levels: dict, terms: list[str], factors: tuple = FACTORS) -> np.ndarray:
    """Sum-to-zero design matrix for the requested terms."""
    n = len(df)
    blocks = [np.ones((n, 1))]
    codes: dict[str, np.ndarray] = {}
    for f in factors:
        labs = levels[f]
        pos = {lab: i for i, lab in enumerate(labs)}
        idx = df[f"site_{f}"].map(pos).to_numpy()
        Z = np.zeros((n, len(labs) - 1))
        for r, i in enumerate(idx):
            if i < len(labs) - 1:
                Z[r, i] = 1.0
            else:
                Z[r, :] = -1.0
        codes[f] = Z
    for term in terms:
        if len(term) == 1:
            blocks.append(codes[term])
        else:
            M = codes[term[0]]
            for f in term[1:]:
                M = np.einsum("ni,nj->nij", M, codes[f]).reshape(n, -1)
            blocks.append(M)
    return np.hstack(blocks)


MAX_DESIGN_COLUMNS = 6000


def incomplete_anova(
    df: pd.DataFrame, prop: str, factors: tuple = FACTORS, ridge: float = 1e-8
) -> dict[str, float]:
    """Hierarchical R^2 increments when the grid has holes.

    Terms enter in order (mains, then pairs) and each share is the *incremental* variance
    explained.  With missing cells the terms are no longer orthogonal, so these shares
    depend on that order; the ordering is fixed and stated rather than optimised.

    The saturated triple term is never fitted.  On an incomplete grid it has more free
    parameters than there are observations (for Castelli, 51*51*6 = 15,606 columns), so a
    fitted value would be pure overfitting dressed as a variance share.  What is reported
    instead is the remainder after mains and pairs, which contains the triple
    interaction *and* the noise together and is labelled as such.
    """
    work = df.dropna(subset=[prop]).copy()
    levels = {f: sorted(work[f"site_{f}"].dropna().unique().tolist()) for f in factors}
    y = work[prop].to_numpy(dtype=float)
    y = y - y.mean()
    ss_total = float((y ** 2).sum())

    order = list(factors) + [a + b for a, b in combinations(factors, 2)]
    shares: dict[str, float] = {}
    used: list[str] = []
    prev_r2 = 0.0
    for term in order:
        trial = used + [term]
        D = _design(work, levels, trial, factors)
        if D.shape[1] > MAX_DESIGN_COLUMNS or D.shape[1] >= len(work):
            shares[term] = float("nan")
            continue
        gram = D.T @ D + ridge * np.eye(D.shape[1])
        beta = np.linalg.lstsq(gram, D.T @ y, rcond=None)[0]
        resid = y - D @ beta
        r2 = 1.0 - float((resid ** 2).sum()) / ss_total if ss_total > 0 else float("nan")
        shares[term] = max(r2 - prev_r2, 0.0)
        prev_r2 = r2
        used = trial
    shares["triple+noise"] = max(1.0 - prev_r2, 0.0)
    return shares


def decompose(
    df: pd.DataFrame,
    prop: str,
    factors: tuple = FACTORS,
    censor_value: Optional[float] = None,
    measure: str = "uniform_product",
) -> VarianceDecomposition:
    """Variance shares of `prop` under an explicitly named probability measure.

    The shares are not measure-free.  "13.8% pair interaction" is a statement about the
    uniform product measure on the level sets, not a chemical constant, and it changes
    when compositions are weighted by how many polymorphs they happen to have.  The
    measure is therefore a required part of the result and is carried on the returned
    object rather than left to the reader.
    """
    if measure not in MEASURES:
        raise ValueError(f"measure must be one of {sorted(MEASURES)}, got {measure!r}")
    work = df.dropna(subset=[prop])
    vals, mask, levels = tensor_from_frame(work, prop, factors)
    complete = bool(mask.all())
    shape = tuple(len(levels[f]) for f in factors)

    censored = 0.0
    if censor_value is not None:
        censored = float((work[prop] == censor_value).mean())

    if measure == "empirical_rows":
        shares = incomplete_anova(work, prop, factors)
        d = VarianceDecomposition(
            property=prop, n_obs=int(len(work)), grid_shape=shape, complete=complete,
            total_variance=float(np.nanvar(work[prop].to_numpy(dtype=float))),
            shares=shares, censored_fraction=censored, measure=measure,
        )
        d.notes.append(
            "row-weighted: terms are not orthogonal, shares are hierarchical R^2 "
            "increments in the fixed order mains -> pairs, and the remainder mixes the "
            "triple interaction with within-composition (polymorph) variation"
        )
        return d

    if complete:
        shares = complete_grid_anova(vals, factors)
        note = ("exact orthogonal decomposition on a complete grid; "
                "no fitting is involved and the shares are properties of the data")
    else:
        shares = incomplete_anova(work, prop, factors)
        filled = int(mask.sum())
        note = (f"grid {filled}/{mask.size} cells filled ({filled / mask.size:.1%}); "
                "shares are hierarchical R^2 increments in the fixed order mains -> pairs -> triple")

    d = VarianceDecomposition(
        property=prop,
        n_obs=int(len(work)),
        grid_shape=shape,
        complete=complete,
        total_variance=float(np.nanvar(work[prop].to_numpy(dtype=float))),
        shares=shares,
        censored_fraction=censored,
        measure=measure,
    )
    d.notes.append(note)
    dupes = int(len(work) - mask.sum()) if complete else 0
    if dupes > 0:
        d.notes.append(f"{dupes} duplicate cells were averaged before decomposition")
    return d
