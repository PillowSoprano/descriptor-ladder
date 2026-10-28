"""Scoring for response fields, interactions, reconstruction and selective prediction.

Absolute-property metrics are not enough for this project and can actively mislead.  A
model can have an excellent R^2 on the property and still get every substitution
direction wrong, because the property is dominated by a between-composition spread that
the substitution never moves through.  Everything here therefore scores *differences*:
how large they are (MAE/RMSE), whether they point the right way (sign accuracy, Spearman
on the substitution direction), and whether the predicted difference field is internally
consistent enough to be integrated back to node values.

Spearman is computed here rather than imported from scipy so that the metrics module has
the same dependency surface as the rest of the package (numpy/pandas only) and so the
tie-handling convention — average ranks — is visible instead of assumed.

Every result is a dataclass with `.render()`, matching `prfield.audits.variance`, because
these numbers end up pasted into the report and a bare dict loses the units, the
conventions and the caveats that make them readable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .graph.typed_graph import accepted, connected_components, incidence_matrix


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _clean_pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray, int]:
    a = np.asarray(y_true, dtype=float).ravel()
    b = np.asarray(y_pred, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError(f"length mismatch: {a.size} true vs {b.size} predicted")
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok], b[ok], int((~ok).sum())


def _rank(x: np.ndarray) -> np.ndarray:
    """Average ranks, so ties do not create an arbitrary ordering."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    ranks[order] = np.arange(1, x.size + 1, dtype=float)
    sx = x[order]
    i = 0
    while i < sx.size:
        j = i
        while j + 1 < sx.size and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b, _ = _clean_pair(a, b)
    if a.size < 2:
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra @ rb) / denom) if denom > 0 else float("nan")


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    ss_res = float(((y - p) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _sign_accuracy(y: np.ndarray, p: np.ndarray, zero_tol: float = 0.0) -> tuple[float, int]:
    """Fraction of substitutions whose direction is predicted correctly.

    Edges whose true difference is within `zero_tol` of zero are excluded from the
    denominator: their direction is not defined, and counting them rewards a model that
    predicts noise around zero.
    """
    resolvable = np.abs(y) > zero_tol
    if not resolvable.any():
        return float("nan"), 0
    return float((np.sign(y[resolvable]) == np.sign(p[resolvable])).mean()), int(resolvable.sum())


# ---------------------------------------------------------------------------
# response metrics
# ---------------------------------------------------------------------------


@dataclass
class ResponseMetrics:
    n: int
    n_dropped: int
    mae: float
    rmse: float
    r2: float
    sign_accuracy: float
    n_signed: int
    spearman: float
    label: str = "response"
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"### {self.label}",
            f"- edges scored: {self.n}" + (f" ({self.n_dropped} non-finite dropped)" if self.n_dropped else ""),
            f"- MAE: {self.mae:.4g} | RMSE: {self.rmse:.4g} | R2: {self.r2:.4f}",
            f"- sign accuracy: {self.sign_accuracy:.1%} over {self.n_signed} directed edges",
            f"- Spearman(direction): {self.spearman:.4f}",
        ]
        for n in self.notes:
            lines.append(f"- note: {n}")
        return "\n".join(lines)


def response_metrics(
    y_true_delta, y_pred_delta, label: str = "response", zero_tol: float = 0.0
) -> ResponseMetrics:
    y, p, dropped = _clean_pair(y_true_delta, y_pred_delta)
    if y.size == 0:
        return ResponseMetrics(0, dropped, *[float("nan")] * 3, float("nan"), 0, float("nan"), label,
                               ["no finite (true, predicted) pairs"])
    sign_acc, n_signed = _sign_accuracy(y, p, zero_tol)
    return ResponseMetrics(
        n=int(y.size),
        n_dropped=dropped,
        mae=float(np.abs(y - p).mean()),
        rmse=float(np.sqrt(((y - p) ** 2).mean())),
        r2=_r2(y, p),
        sign_accuracy=sign_acc,
        n_signed=n_signed,
        spearman=spearman(y, p),
        label=label,
    )


# ---------------------------------------------------------------------------
# interaction metrics
# ---------------------------------------------------------------------------


@dataclass
class InteractionMetrics:
    overall: ResponseMetrics
    sign_agreement: float
    threshold: float
    resolvable_fraction: float
    n_resolvable: int
    resolvable: Optional[ResponseMetrics] = None
    label: str = "mixed second difference"

    def render(self) -> str:
        lines = [f"### {self.label}", self.overall.render().split("\n", 1)[1]]
        lines.append(f"- sign agreement: {self.sign_agreement:.1%}")
        lines.append(
            f"- resolvable above |mixed delta| > {self.threshold:g}: "
            f"{self.resolvable_fraction:.1%} of squares ({self.n_resolvable})"
        )
        if self.resolvable is not None and self.n_resolvable > 0:
            lines.append("- restricted to resolvable squares:")
            lines.append(f"  - MAE: {self.resolvable.mae:.4g} | RMSE: {self.resolvable.rmse:.4g} "
                         f"| R2: {self.resolvable.r2:.4f}")
            lines.append(f"  - sign accuracy: {self.resolvable.sign_accuracy:.1%}")
        if self.resolvable_fraction < 0.05:
            lines.append(
                "- **almost no square exceeds the threshold**: whatever the metrics above "
                "say, the interaction term is not resolvable above noise in this data"
            )
        return "\n".join(lines)


def interaction_metrics(
    true_mixed, pred_mixed, threshold: float = 1e-3, label: str = "mixed second difference"
) -> InteractionMetrics:
    """Score predicted mixed second differences against measured ones.

    `threshold` is a magnitude in the units of the property and should be set from the
    noise scale the data actually has (replicate spread, or the functional-to-functional
    difference from audit A2) — not from what looks good.  Squares below it carry an
    interaction that cannot be distinguished from zero, so a high sign accuracy over them
    is a statement about noise, not about the model.
    """
    y, p, dropped = _clean_pair(true_mixed, pred_mixed)
    overall = response_metrics(y, p, label=label)
    if y.size == 0:
        return InteractionMetrics(overall, float("nan"), threshold, float("nan"), 0, None, label)
    agree = float((np.sign(y) == np.sign(p)).mean())
    mask = np.abs(y) > threshold
    resolvable = (
        response_metrics(y[mask], p[mask], label=f"{label} (resolvable)") if mask.any() else None
    )
    return InteractionMetrics(
        overall=overall,
        sign_agreement=agree,
        threshold=float(threshold),
        resolvable_fraction=float(mask.mean()),
        n_resolvable=int(mask.sum()),
        resolvable=resolvable,
        label=label,
    )


# ---------------------------------------------------------------------------
# node reconstruction
# ---------------------------------------------------------------------------


@dataclass
class NodeReconstruction:
    n_nodes: int
    n_edges: int
    n_components: int
    mae: float
    rmse: float
    largest_component_size: int
    largest_component_mae: float
    nonconservative_rms: float
    per_component: pd.DataFrame = field(default_factory=pd.DataFrame)
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            "### node reconstruction from predicted edges",
            f"- nodes: {self.n_nodes}, accepted edges: {self.n_edges}, "
            f"connected components: {self.n_components}",
            f"- per-component centred MAE: {self.mae:.4g} | RMSE: {self.rmse:.4g}",
            f"- largest component ({self.largest_component_size} nodes) MAE: "
            f"{self.largest_component_mae:.4g}",
            f"- non-conservative residual RMS: {self.nonconservative_rms:.4g}",
        ]
        if self.n_components > 1:
            lines.append(
                f"- note: the gauge is fixed separately in each of the {self.n_components} "
                "components, so these numbers say nothing about the relative offset between them"
            )
        for n in self.notes:
            lines.append(f"- note: {n}")
        return "\n".join(lines)


def node_reconstruction_error(
    edges_pred: pd.DataFrame,
    node_index: Mapping[str, int],
    y_true_nodes,
    delta_col: str = "delta_pred",
) -> NodeReconstruction:
    """Integrate predicted edge differences back to node values and score them.

    Solve D f = delta_pred in least squares, where D is the edge-node incidence matrix of
    the accepted edges.  D has a null space: adding a constant to f on any connected
    component leaves every difference unchanged, so f is only determined up to one
    constant per component.  The gauge is fixed by shifting each component's f to the mean
    of the true values on that component, and the MAE is computed after that shift.

    Doing this globally instead would be wrong, and quietly so: a second component's
    arbitrary offset would be added to every one of its nodes and would show up as a large
    node error that has nothing to do with how well the model predicts substitutions.  The
    per-component treatment is not a convenience, it is the only version of this metric
    that measures the model.

    `nonconservative_rms` is the part of the predicted edge field the least-squares fit
    could not explain: ||D f - delta_pred||, i.e. the curl of the predicted response.  A
    conservative field has zero here by construction, so a large value means the model's
    predicted differences are mutually inconsistent around closed loops.
    """
    acc = accepted(edges_pred)
    if delta_col not in acc.columns:
        raise KeyError(f"edge frame has no {delta_col!r} column to integrate")
    node_index = dict(node_index)
    nodes = list(node_index)

    # `nodes` is in insertion order; every array below is indexed by node_index instead,
    # so that a caller passing a plain array does not have to know that order.
    if isinstance(y_true_nodes, pd.Series):
        y_true = np.full(len(nodes), np.nan)
        for n, i in node_index.items():
            if n in y_true_nodes.index:
                y_true[i] = float(y_true_nodes.loc[n])
    else:
        y_true = np.asarray(y_true_nodes, dtype=float).ravel()
        if y_true.size != len(nodes):
            raise ValueError(
                "y_true_nodes must be a Series indexed by node id, or an array already "
                "ordered by node_index"
            )
    nodes = [None] * len(node_index)
    for n, i in node_index.items():
        nodes[i] = n

    D = incidence_matrix(acc, node_index)
    delta = acc[delta_col].to_numpy(dtype=float)
    finite = np.isfinite(delta)
    D, delta = D[finite], delta[finite]

    notes: list[str] = []
    if D.shape[0] == 0:
        return NodeReconstruction(len(nodes), 0, 0, float("nan"), float("nan"), 0,
                                  float("nan"), float("nan"),
                                  notes=["no accepted edges with a finite prediction"])

    f, *_ = np.linalg.lstsq(D, delta, rcond=None)
    nonconservative = float(np.sqrt(np.mean((D @ f - delta) ** 2)))

    comp = connected_components(acc, nodes)
    comp_labels = np.array([comp.get(n, n) for n in nodes], dtype=object)

    errors = np.full(len(nodes), np.nan)
    records = []
    for label in pd.unique(comp_labels):
        mask = comp_labels == label
        if not np.isfinite(y_true[mask]).any():
            continue
        shift = float(np.nanmean(y_true[mask]) - f[mask].mean())
        err = np.abs((f[mask] + shift) - y_true[mask])
        errors[mask] = err
        records.append({
            "component": label,
            "n_nodes": int(mask.sum()),
            "mae": float(np.nanmean(err)),
        })
    per_component = pd.DataFrame.from_records(records).sort_values("n_nodes", ascending=False)

    valid = np.isfinite(errors)
    largest = per_component.iloc[0] if len(per_component) else None
    isolated = int((per_component["n_nodes"] == 1).sum()) if len(per_component) else 0
    if isolated:
        notes.append(
            f"{isolated} components are single nodes; their centred error is exactly zero "
            "by construction and inflates the aggregate downwards"
        )

    return NodeReconstruction(
        n_nodes=len(nodes),
        n_edges=int(D.shape[0]),
        n_components=int(len(per_component)),
        mae=float(np.nanmean(errors[valid])) if valid.any() else float("nan"),
        rmse=float(np.sqrt(np.nanmean(errors[valid] ** 2))) if valid.any() else float("nan"),
        largest_component_size=int(largest["n_nodes"]) if largest is not None else 0,
        largest_component_mae=float(largest["mae"]) if largest is not None else float("nan"),
        nonconservative_rms=nonconservative,
        per_component=per_component,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# selective prediction
# ---------------------------------------------------------------------------


@dataclass
class RiskCoverage:
    coverages: np.ndarray
    risks: np.ndarray
    counts: np.ndarray
    full_risk: float
    label: str = "risk-coverage"

    def risk_at_coverage(self, coverage: float) -> float:
        if self.coverages.size == 0:
            return float("nan")
        i = int(np.argmin(np.abs(self.coverages - coverage)))
        return float(self.risks[i])

    def render(self) -> str:
        lines = [f"### {self.label}", f"- risk at full coverage: {self.full_risk:.4g}"]
        for c, r, n in zip(self.coverages, self.risks, self.counts):
            lines.append(f"  - coverage {c:5.0%} (n={int(n):5d}): risk {r:.4g}")
        if self.risks.size and self.risks[0] >= self.full_risk:
            lines.append(
                "- note: risk does not decrease as coverage shrinks, so the abstention "
                "score carries no information about where the model fails"
            )
        return "\n".join(lines)


def selective_risk(
    errors,
    scores,
    coverages: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    label: str = "risk-coverage",
) -> RiskCoverage:
    """Risk-coverage curve for an abstention rule.

    `scores` are *risk* scores: higher means the model expects to be less reliable there
    (predictive variance, distance from the training span, disagreement between blocks).
    At coverage c the lowest-scoring c fraction is kept and the mean error over it is the
    selective risk.  The convention matters — feeding a confidence score in unchanged
    produces a monotonically increasing curve that looks like a broken model.
    """
    e = np.asarray(errors, dtype=float).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    if e.size != s.size:
        raise ValueError(f"length mismatch: {e.size} errors vs {s.size} scores")
    ok = np.isfinite(e) & np.isfinite(s)
    e, s = e[ok], s[ok]
    order = np.argsort(s, kind="mergesort")
    e_sorted = e[order]

    cov, risk, cnt = [], [], []
    for c in coverages:
        k = max(1, int(round(c * e_sorted.size))) if e_sorted.size else 0
        if k == 0:
            continue
        cov.append(k / e_sorted.size)
        risk.append(float(e_sorted[:k].mean()))
        cnt.append(k)
    return RiskCoverage(
        coverages=np.array(cov),
        risks=np.array(risk),
        counts=np.array(cnt),
        full_risk=float(e.mean()) if e.size else float("nan"),
        label=label,
    )


@dataclass
class CoverageAtRisk:
    target_risk: float
    coverage: float
    achieved_risk: float
    n_kept: int
    n_total: int

    def render(self) -> str:
        if not np.isfinite(self.coverage) or self.coverage == 0.0:
            return (f"### coverage at risk <= {self.target_risk:.4g}\n"
                    "- no non-empty selection reaches the target risk")
        return "\n".join([
            f"### coverage at risk <= {self.target_risk:.4g}",
            f"- coverage: {self.coverage:.1%} ({self.n_kept}/{self.n_total} predictions kept)",
            f"- achieved risk: {self.achieved_risk:.4g}",
        ])


def coverage_at_risk(errors, scores, target_risk: float) -> CoverageAtRisk:
    """Largest coverage whose selective risk stays at or below `target_risk`.

    Computed over every prefix of the score-ordered predictions rather than over the
    coarse grid of `selective_risk`, because the answer is the operating point at which
    the predictions would be used.
    """
    e = np.asarray(errors, dtype=float).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    ok = np.isfinite(e) & np.isfinite(s)
    e, s = e[ok], s[ok]
    n = e.size
    if n == 0:
        return CoverageAtRisk(float(target_risk), float("nan"), float("nan"), 0, 0)
    e_sorted = e[np.argsort(s, kind="mergesort")]
    running = np.cumsum(e_sorted) / np.arange(1, n + 1)
    feasible = np.flatnonzero(running <= target_risk)
    if feasible.size == 0:
        return CoverageAtRisk(float(target_risk), 0.0, float("nan"), 0, n)
    k = int(feasible[-1]) + 1
    return CoverageAtRisk(float(target_risk), k / n, float(running[k - 1]), k, n)


# ---------------------------------------------------------------------------
# interval calibration
# ---------------------------------------------------------------------------


@dataclass
class IntervalCoverage:
    n: int
    nominal: float
    empirical: float
    mean_width: float
    median_width: float
    n_inverted: int

    @property
    def calibration_gap(self) -> float:
        return self.empirical - self.nominal

    def render(self) -> str:
        lines = [
            "### prediction interval calibration",
            f"- intervals: {self.n}, nominal level: {self.nominal:.0%}",
            f"- empirical coverage: {self.empirical:.1%} (gap {self.calibration_gap:+.1%})",
            f"- width: mean {self.mean_width:.4g}, median {self.median_width:.4g}",
        ]
        if self.n_inverted:
            lines.append(f"- **{self.n_inverted} intervals have lo > hi** and were counted as misses")
        if self.calibration_gap < -0.05:
            lines.append("- note: intervals are too narrow; selective-risk numbers built on them are optimistic")
        return "\n".join(lines)


def prediction_interval_coverage(y, lo, hi, nominal: float = 0.9) -> IntervalCoverage:
    """Empirical coverage and width of prediction intervals.

    Width is reported alongside coverage because coverage alone is trivially satisfiable:
    an interval spanning the full range of the property covers everything and says nothing.
    """
    y = np.asarray(y, dtype=float).ravel()
    lo = np.asarray(lo, dtype=float).ravel()
    hi = np.asarray(hi, dtype=float).ravel()
    if not (y.size == lo.size == hi.size):
        raise ValueError("y, lo and hi must have the same length")
    ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    y, lo, hi = y[ok], lo[ok], hi[ok]
    inverted = int((hi < lo).sum())
    inside = (y >= lo) & (y <= hi)
    width = np.abs(hi - lo)
    return IntervalCoverage(
        n=int(y.size),
        nominal=float(nominal),
        empirical=float(inside.mean()) if y.size else float("nan"),
        mean_width=float(width.mean()) if y.size else float("nan"),
        median_width=float(np.median(width)) if y.size else float("nan"),
        n_inverted=inverted,
    )


__all__ = [
    "CoverageAtRisk",
    "InteractionMetrics",
    "IntervalCoverage",
    "NodeReconstruction",
    "ResponseMetrics",
    "RiskCoverage",
    "coverage_at_risk",
    "interaction_metrics",
    "node_reconstruction_error",
    "prediction_interval_coverage",
    "response_metrics",
    "selective_risk",
    "spearman",
]
