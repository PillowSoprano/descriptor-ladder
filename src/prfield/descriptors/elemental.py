"""Element- and composition-level descriptor matrices phi_A, phi_B, phi_X.

Every model and every descriptor-span calculation reads its descriptors through
`ElementalDescriptors`, so all of them use identical features in identical order.

Missing values are never imputed. A zero-filled electronegativity would look like a genuine value
near the training mean and place an element inside the descriptor span when it lies outside it.
Missing entries stay NaN, and the label and feature pairs concerned are reported on the returned
bundle (`.missing`, `.partial`), so a caller decides explicitly what to do.

Site levels are not all elements. The Castelli X site carries composite anion sets ("O2F", "N3");
pymatgen has no `Element` entry for them. `composition_descriptors` parses the formula and takes
stoichiometry-weighted statistics of the same elemental features, so both kinds of site level give
vectors built from the same quantities.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from pymatgen.core import Element

from ..ingest.canonical import ORGANIC_CATIONS

# label -> formula for every non-elemental A-site level the canonical vocabulary knows.
# Imported rather than restated: if a cation is added to the vocabulary and forgotten
# here, its descriptor row would silently fall back to parsing the *label* as a formula
# ("MA" -> Mo + ...), which is worse than an error.
MOLECULAR_FORMULAS: dict[str, str] = {
    label: formula for label, formula in ORGANIC_CATIONS.values()
}

# Features that scale with the number of atoms in a composite level.  For those a sum is
# meaningful and a mean is a per-atom average; for the rest (radii, electronegativity,
# row, group) only the mean has a physical reading, so no sum column is produced.
EXTENSIVE_FEATURES = ("atomic_number", "atomic_mass", "valence_electrons")


class MissingDescriptor(ValueError):
    """Raised when a caller asks for a matrix that still contains unresolved NaNs."""


def _valence_electrons(el: Element) -> float:
    """Electrons in the outermost shell plus any unfilled (n-1)d / (n-2)f subshell.

    `Element.valence` raises for elements with an ambiguous valence orbital (Cr, Mo, Nb,
    Pt, Rh, Ru in the current pymatgen tables), which would knock six transition metals
    out of the Castelli B site.  Counting from the full electronic structure is defined
    for every element, at the price of a convention: filled d/f shells are treated as
    core (Cu -> 1, not 11).
    """
    try:
        structure = el.full_electronic_structure
    except Exception:  # pragma: no cover - defensive, pymatgen table gap
        return float("nan")
    if not structure:
        return float("nan")
    max_n = max(n for n, _, _ in structure)
    total = 0.0
    for n, orbital, occ in structure:
        if n == max_n:
            total += occ
        elif orbital == "d" and n == max_n - 1 and occ < 10:
            total += occ
        elif orbital == "f" and n == max_n - 2 and occ < 14:
            total += occ
    return float(total)


def _safe(value) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _ionic_radius(el: Element) -> float:
    # pymatgen returns exactly 0.0 when it has no ionic-radius table for the element
    # (H is the case that matters here).  A zero radius is not a physical measurement,
    # so it is reported as missing rather than fed to the model as a very small ion.
    r = _safe(el.average_ionic_radius)
    return float("nan") if r == 0.0 else r


ELEMENT_FEATURES: dict[str, callable] = {
    "atomic_number": lambda el: _safe(el.Z),
    "atomic_mass": lambda el: _safe(el.atomic_mass),
    "electronegativity": lambda el: _safe(el.X),
    "ionic_radius": _ionic_radius,
    "atomic_radius": lambda el: _safe(el.atomic_radius),
    "row": lambda el: _safe(el.row),
    "group": lambda el: _safe(el.group),
    "valence_electrons": _valence_electrons,
    "ionization_energy": lambda el: _safe(el.ionization_energy),
    "electron_affinity": lambda el: _safe(el.electron_affinity),
}

FEATURE_NAMES = tuple(ELEMENT_FEATURES)


# ---------------------------------------------------------------------------
# Descriptor bundle
# ---------------------------------------------------------------------------


@dataclass
class ElementalDescriptors:
    """One descriptor table for one site.

    `frame` is indexed by canonical species label (the strings in `site_A`/`site_B`/
    `site_X`) with numeric columns.  `missing` and `partial` are the companion report:
    `missing[label]` lists features with no value at all, `partial[label]` lists features
    of a composite level that were averaged over only part of its atoms because some
    constituent element lacked the property.  Both are empty for a clean bundle.
    """

    site: str
    frame: pd.DataFrame
    kind: str = "element"
    missing: dict[str, list[str]] = field(default_factory=dict)
    partial: dict[str, list[str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.frame.columns)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self.frame.index)

    @property
    def n_features(self) -> int:
        return int(self.frame.shape[1])

    def matrix(self, labels: Sequence[str], allow_missing: bool = False) -> np.ndarray:
        """(n_labels, d) array in the order given; labels may repeat."""
        labels = list(labels)
        unknown = sorted({lab for lab in labels if lab not in self.frame.index})
        if unknown:
            raise KeyError(
                f"site {self.site}: no descriptor row for {unknown!r}; "
                "add a formula to MOLECULAR_FORMULAS or build the bundle over more labels"
            )
        out = self.frame.reindex(labels).to_numpy(dtype=float)
        if not allow_missing and not np.isfinite(out).all():
            bad = sorted({labels[i] for i in np.unique(np.argwhere(~np.isfinite(out))[:, 0])})
            raise MissingDescriptor(
                f"site {self.site}: descriptor rows contain NaN for {bad!r}; "
                "drop the feature, drop the species, or pass allow_missing=True knowingly"
            )
        return out

    def subset(self, labels: Iterable[str]) -> "ElementalDescriptors":
        keep = [lab for lab in self.frame.index if lab in set(labels)]
        return ElementalDescriptors(
            site=self.site,
            frame=self.frame.loc[keep],
            kind=self.kind,
            missing={k: v for k, v in self.missing.items() if k in set(keep)},
            partial={k: v for k, v in self.partial.items() if k in set(keep)},
            notes=list(self.notes),
        )

    def drop_missing_features(self) -> "ElementalDescriptors":
        """Drop whole columns that are NaN for any label — an explicit, recorded choice."""
        bad = [c for c in self.frame.columns if not np.isfinite(self.frame[c].to_numpy(float)).all()]
        if not bad:
            return self
        out = ElementalDescriptors(
            site=self.site,
            frame=self.frame.drop(columns=bad),
            kind=self.kind,
            missing={},
            partial=dict(self.partial),
            notes=list(self.notes),
        )
        out.notes.append(f"dropped features with missing values: {', '.join(bad)}")
        return out

    def render(self) -> str:
        lines = [
            f"### descriptors — site {self.site} ({self.kind})",
            f"- levels: {len(self.frame)}, features: {self.n_features}",
            f"- features: {', '.join(self.feature_names)}",
        ]
        if self.missing:
            for lab in sorted(self.missing):
                lines.append(f"- **missing** {lab}: {', '.join(self.missing[lab])}")
        else:
            lines.append("- no missing values")
        for lab in sorted(self.partial):
            lines.append(f"- partial (averaged over a subset of atoms) {lab}: {', '.join(self.partial[lab])}")
        for n in self.notes:
            lines.append(f"- note: {n}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Formula parsing
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"([A-Z][a-z]?)|(\d+)|([()\[\]])")


def parse_formula(formula: str) -> dict[str, float]:
    """Element -> count for simple formulas with parentheses and digit multipliers.

    Handles the shapes the two vocabularies actually contain: "CH3NH3", "(CH3)2NH2",
    "C(NH2)3", "O2F", "N3".  It is deliberately not a general chemistry parser — charges,
    hydrates and nested brackets beyond plain grouping are rejected loudly, because a
    formula this module silently mis-parses becomes a wrong descriptor row that no
    downstream audit can detect.
    """
    counts: dict[str, float] = {}
    stack: list[dict[str, float]] = [counts]
    pos = 0
    n = len(formula)
    while pos < n:
        ch = formula[pos]
        if ch in "([":
            stack.append({})
            pos += 1
            continue
        if ch in ")]":
            group = stack.pop()
            if not stack:
                raise ValueError(f"unbalanced parenthesis in {formula!r}")
            pos += 1
            mult, pos = _read_int(formula, pos)
            for el, c in group.items():
                stack[-1][el] = stack[-1].get(el, 0.0) + c * mult
            continue
        m = _TOKEN.match(formula, pos)
        if m is None or m.group(1) is None:
            raise ValueError(f"cannot parse {formula!r} at position {pos} ({formula[pos:]!r})")
        symbol = m.group(1)
        pos = m.end()
        mult, pos = _read_int(formula, pos)
        stack[-1][symbol] = stack[-1].get(symbol, 0.0) + mult
    if len(stack) != 1:
        raise ValueError(f"unbalanced parenthesis in {formula!r}")
    if not counts:
        raise ValueError(f"empty formula {formula!r}")
    return counts


def _read_int(text: str, pos: int) -> tuple[float, int]:
    start = pos
    while pos < len(text) and text[pos].isdigit():
        pos += 1
    return (float(text[start:pos]) if pos > start else 1.0), pos


def formula_for(label: str, formulas: Optional[Mapping[str, str]] = None) -> str:
    """Resolve a site label to a formula string: explicit map, then vocabulary, then self."""
    if formulas is not None and label in formulas:
        return formulas[label]
    if label in MOLECULAR_FORMULAS:
        return MOLECULAR_FORMULAS[label]
    return label


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _element_row(symbol: str) -> tuple[dict[str, float], list[str]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        el = Element(symbol)
        row = {name: fn(el) for name, fn in ELEMENT_FEATURES.items()}
    miss = [k for k, v in row.items() if not np.isfinite(v)]
    return row, miss


def physical_descriptors(
    labels: Iterable[str],
    kind: str = "element",
    site: str = "?",
    formulas: Optional[Mapping[str, str]] = None,
) -> ElementalDescriptors:
    """Descriptor table for a set of site levels.

    kind="element"  -> every label is a pymatgen `Element` symbol.
    kind="molecule" -> every label is a composite level, routed to
                       `composition_descriptors`.
    kind="auto"     -> per-label dispatch; the whole bundle then uses the composition
                       feature set if *any* label is composite, because a matrix whose
                       rows carry different features is not a matrix.
    """
    labels = list(dict.fromkeys(labels))
    if kind == "molecule":
        return composition_descriptors(labels, formulas, site=site)
    if kind == "auto":
        if any(not _is_element_symbol(lab) for lab in labels):
            return composition_descriptors(labels, formulas, site=site)
        kind = "element"
    if kind != "element":
        raise ValueError(f"kind must be element | molecule | auto, got {kind!r}")

    rows: dict[str, dict[str, float]] = {}
    missing: dict[str, list[str]] = {}
    for lab in labels:
        if not _is_element_symbol(lab):
            raise ValueError(
                f"site {site}: {lab!r} is not an element symbol; use kind='molecule' "
                "or kind='auto' so it is routed through composition_descriptors"
            )
        row, miss = _element_row(lab)
        rows[lab] = row
        if miss:
            missing[lab] = miss
    frame = pd.DataFrame.from_dict(rows, orient="index", columns=list(FEATURE_NAMES))
    frame.index.name = f"site_{site}"
    return ElementalDescriptors(site=site, frame=frame, kind="element", missing=missing)


def _is_element_symbol(label: str) -> bool:
    if not label or not label.isalpha() or len(label) > 2:
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Element(label)
    except Exception:
        return False
    return True


def composition_descriptors(
    labels: Iterable[str],
    formulas: Optional[Mapping[str, str]] = None,
    site: str = "?",
) -> ElementalDescriptors:
    """Stoichiometry-weighted descriptors for composite site levels.

    Produces `mean_<feature>` for every elemental feature, `sum_<feature>` for the
    extensive ones, and `n_atoms`.  The mean is weighted by stoichiometric count, so
    "O2F" is 2/3 oxygen and 1/3 fluorine rather than an unweighted average over the two
    distinct elements — the unweighted version would make "O2F" and "OF2" identical.

    When a constituent element is missing a feature, the mean is taken over the atoms
    that do have it and the label/feature pair is recorded in `.partial`; sums are set to
    NaN and recorded in `.missing`, because a partial sum of an extensive quantity is
    simply wrong rather than approximate.
    """
    labels = list(dict.fromkeys(labels))
    cache: dict[str, dict[str, float]] = {}
    rows: dict[str, dict[str, float]] = {}
    missing: dict[str, list[str]] = {}
    partial: dict[str, list[str]] = {}

    columns = [f"mean_{f}" for f in FEATURE_NAMES]
    columns += [f"sum_{f}" for f in EXTENSIVE_FEATURES]
    columns += ["n_atoms"]

    for lab in labels:
        formula = formula_for(lab, formulas)
        counts = parse_formula(formula)
        for sym in counts:
            if sym not in cache:
                if not _is_element_symbol(sym):
                    raise ValueError(
                        f"site {site}: formula {formula!r} for level {lab!r} contains "
                        f"unknown element {sym!r}"
                    )
                cache[sym], _ = _element_row(sym)
        total_atoms = float(sum(counts.values()))
        row: dict[str, float] = {"n_atoms": total_atoms}
        for feat in FEATURE_NAMES:
            weight = 0.0
            acc = 0.0
            complete = True
            for sym, c in counts.items():
                v = cache[sym][feat]
                if np.isfinite(v):
                    acc += c * v
                    weight += c
                else:
                    complete = False
            row[f"mean_{feat}"] = acc / weight if weight > 0 else float("nan")
            if feat in EXTENSIVE_FEATURES:
                row[f"sum_{feat}"] = acc if complete else float("nan")
            if not complete:
                if weight == 0:
                    missing.setdefault(lab, []).append(f"mean_{feat}")
                else:
                    partial.setdefault(lab, []).append(f"mean_{feat}")
                if feat in EXTENSIVE_FEATURES:
                    missing.setdefault(lab, []).append(f"sum_{feat}")
        rows[lab] = row

    frame = pd.DataFrame.from_dict(rows, orient="index")[columns]
    frame.index.name = f"site_{site}"
    bundle = ElementalDescriptors(
        site=site, frame=frame, kind="molecule", missing=missing, partial=partial
    )
    bundle.notes.append(
        "means are stoichiometry-weighted; sums are reported only for extensive features"
    )
    return bundle


def descriptors_for_frame(
    df: pd.DataFrame,
    site: str,
    formulas: Optional[Mapping[str, str]] = None,
) -> ElementalDescriptors:
    """Build the bundle for one site straight off a canonical sample frame.

    Uses `site_{s}_kind` when the ingest adapter filled it, so the element/molecule
    decision is the one the data contract already made rather than a re-guess here.
    """
    labels = sorted(df[f"site_{site}"].dropna().unique().tolist())
    kind_col = f"site_{site}_kind"
    kind = "auto"
    if kind_col in df.columns:
        kinds = set(df[kind_col].dropna().unique().tolist())
        if kinds == {"element"}:
            kind = "element"
        elif kinds == {"molecule"}:
            kind = "molecule"
    return physical_descriptors(labels, kind=kind, site=site, formulas=formulas)


# ---------------------------------------------------------------------------
# Standardisation
# ---------------------------------------------------------------------------


@dataclass
class Standardizer:
    """Column means and standard deviations fitted on a training matrix.

    Kept as an object instead of being recomputed inside each model because the held-out
    element in the extrapolation experiment must be transformed with *training*
    statistics.  Refitting on data that includes the held-out species leaks exactly the
    information the experiment is trying to withhold and shrinks its apparent distance
    from the training span.
    """

    mean: np.ndarray
    std: np.ndarray
    feature_names: tuple[str, ...] = ()
    zero_variance: tuple[str, ...] = ()

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        return (np.asarray(matrix, dtype=float) - self.mean) / self.std

    def inverse_transform(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray(matrix, dtype=float) * self.std + self.mean


def standardize(
    matrix: np.ndarray, feature_names: Sequence[str] = ()
) -> tuple[np.ndarray, Standardizer]:
    """z-score the columns; returns (z, fitted Standardizer)."""
    X = np.asarray(matrix, dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    zero = std <= 0
    # A constant column carries no information but must not become inf/NaN, which would
    # poison every downstream linear solve.  It is scaled by 1 and named in the result.
    std = np.where(zero, 1.0, std)
    names = tuple(feature_names)
    zero_names = tuple(names[i] for i in np.flatnonzero(zero)) if names else ()
    scaler = Standardizer(mean=mean, std=std, feature_names=names, zero_variance=zero_names)
    return scaler.transform(X), scaler


# ---------------------------------------------------------------------------
# Span diagnostics
# ---------------------------------------------------------------------------


@dataclass
class SpanDiagnostics:
    """Where a query species sits relative to the training descriptor row space.

    `relative_residual` is the x-axis of the falsification plot: the project's error
    decomposition predicts that response error grows with the component of the query
    descriptor that the training species could not have expressed, and only the relative
    version is comparable across sites and feature sets because the raw residual inherits
    the units and scale of the descriptor block.  `leverage` is the in-span counterpart —
    a species can be perfectly inside the span and still be extrapolatory along a weakly
    sampled direction, which is what a large leverage with a near-zero residual means.
    `smallest_singular_value` says how trustworthy that split is at all: when it is close
    to zero the training matrix is effectively rank deficient and small residuals should
    not be read as evidence of interpolation.
    """

    n_train: int
    n_features: int
    rank: int
    residual_norm: float
    query_norm: float
    relative_residual: float
    leverage: float
    smallest_singular_value: float
    largest_singular_value: float

    @property
    def condition_number(self) -> float:
        if self.smallest_singular_value <= 0:
            return float("inf")
        return self.largest_singular_value / self.smallest_singular_value

    @property
    def in_span(self) -> bool:
        return self.relative_residual <= 1e-8

    def render(self) -> str:
        lines = [
            "### span diagnostics",
            f"- training rows: {self.n_train}, features: {self.n_features}, rank: {self.rank}",
            f"- residual norm: {self.residual_norm:.4g} (relative: {self.relative_residual:.4g})",
            f"- leverage h: {self.leverage:.4g}",
            f"- singular values: sigma_min {self.smallest_singular_value:.4g}, "
            f"sigma_max {self.largest_singular_value:.4g}, cond {self.condition_number:.4g}",
            f"- verdict: {'inside the training span' if self.in_span else 'outside the training span'}",
        ]
        if self.rank < self.n_features:
            lines.append(
                "- note: the training matrix is rank deficient, so directions outside its "
                "row space are unconstrained and the residual below is a lower bound on "
                "the extrapolation actually being asked for"
            )
        return "\n".join(lines)


def span_diagnostics(
    train_matrix: np.ndarray, query_vector: np.ndarray, rcond: float = 1e-10
) -> SpanDiagnostics:
    """Projection residual, relative residual, leverage and sigma_min for one query row.

    The matrices are used as given: standardise both with the *same* fitted
    `Standardizer` before calling, or the residual mixes descriptor units together and
    the number is not comparable between sites.
    """
    X = np.atleast_2d(np.asarray(train_matrix, dtype=float))
    q = np.asarray(query_vector, dtype=float).ravel()
    if X.shape[1] != q.size:
        raise ValueError(f"query has {q.size} features, training matrix has {X.shape[1]}")

    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    tol = rcond * (s[0] if s.size else 0.0)
    keep = s > tol
    rank = int(keep.sum())
    V = Vt[keep].T                       # (d, rank) orthonormal basis of the row space
    proj = V @ (V.T @ q) if rank else np.zeros_like(q)
    residual = q - proj
    res_norm = float(np.linalg.norm(residual))
    q_norm = float(np.linalg.norm(q))
    rel = res_norm / q_norm if q_norm > 0 else 0.0

    s_keep = s[keep]
    coords = (Vt[keep] @ q)
    leverage = float(np.sum((coords / s_keep) ** 2)) if rank else float("inf")

    return SpanDiagnostics(
        n_train=int(X.shape[0]),
        n_features=int(X.shape[1]),
        rank=rank,
        residual_norm=res_norm,
        query_norm=q_norm,
        relative_residual=float(rel),
        leverage=leverage,
        smallest_singular_value=float(s_keep[-1]) if rank else 0.0,
        largest_singular_value=float(s[0]) if s.size else 0.0,
    )


def span_table(
    descriptors: ElementalDescriptors,
    train_labels: Sequence[str],
    query_labels: Sequence[str],
    scaler: Optional[Standardizer] = None,
) -> pd.DataFrame:
    """One `span_diagnostics` row per query label, all against the same training block.

    The scaler is fitted on the training labels only when not supplied, which is the
    correct default for the leave-one-element-out experiment.
    """
    train = descriptors.matrix(train_labels)
    if scaler is None:
        train_z, scaler = standardize(train, descriptors.feature_names)
    else:
        train_z = scaler.transform(train)
    records = []
    for lab in query_labels:
        q = scaler.transform(descriptors.matrix([lab]))[0]
        d = span_diagnostics(train_z, q)
        records.append({
            "label": lab,
            "site": descriptors.site,
            "residual_norm": d.residual_norm,
            "relative_residual": d.relative_residual,
            "leverage": d.leverage,
            "sigma_min": d.smallest_singular_value,
            "rank": d.rank,
            "in_span": d.in_span,
        })
    return pd.DataFrame.from_records(records)
