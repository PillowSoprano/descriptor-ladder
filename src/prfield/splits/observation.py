"""Observation designs: which compositions are computed, and which responses follow from them.

A density-functional calculation gives a property of a composition; a substitution response is
the difference of two such values. Every design here is therefore defined by which compositions
are hidden, and the test responses are those the hidden compositions induce.
`InducedEdges.leakage_fraction` measures how many of them could still be recovered by subtraction.

    random compositions  some compositions were never computed     rungs 1 and 2
    left-out element     every composition of one element is hidden rung 3
    block                a region of element combinations is hidden rung 1

`random_edges_with_seen_endpoints` hides responses rather than compositions and exists to measure
the leak of rung 0.

Hiding unit
-----------
When a composition has several structures, hiding one row while keeping its siblings would leak the
same chemistry, so the default hiding unit is the composition (A, B, X). `unit="row"` is for
datasets with one structure per composition, where the two coincide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..graph.typed_graph import accepted
from ..schema import SITES

COMP_KEYS = [f"site_{s}" for s in SITES]

MECHANISM_QUESTIONS = {
    "random_nodes": "can the field be recovered when some materials were never computed",
    "fiber": "can an element level absent from training be extrapolated to",
    "block": "can an absent *combination* of levels be extrapolated to, when each level is seen alone",
    "random_edges": "SANITY CHECK ONLY — endpoints stay visible, so this is a subtraction exercise",
}


def _comp_id(df: pd.DataFrame) -> pd.Series:
    return df[COMP_KEYS].astype(str).agg("|".join, axis=1)


@dataclass
class InducedEdges:
    """The edge supervision implied by a node observation, plus its leakage audit."""

    train: pd.DataFrame          # both endpoints observed -> derivable supervision
    test: pd.DataFrame           # at least one endpoint hidden -> a genuine question
    leaked: pd.DataFrame         # nominally test, but both endpoints observed
    n_candidate: int

    @property
    def leakage_fraction(self) -> float:
        denom = len(self.test) + len(self.leaked)
        return len(self.leaked) / denom if denom else 0.0

    def render(self) -> str:
        return (
            f"- derivable training edges: {len(self.train):,}\n"
            f"- test edges (>=1 hidden endpoint): {len(self.test):,}\n"
            f"- leaked (both endpoints visible): {len(self.leaked):,} "
            f"({self.leakage_fraction:.1%} of nominal test edges)"
        )


@dataclass
class ObservationDesign:
    """Omega — which nodes were observed, why, and what that implies for edges."""

    kind: str
    observed_ids: frozenset
    held_out_ids: frozenset
    description: str
    params: dict = field(default_factory=dict)
    unit: str = "composition"

    @property
    def observed_fraction(self) -> float:
        total = len(self.observed_ids) + len(self.held_out_ids)
        return len(self.observed_ids) / total if total else float("nan")

    def train_nodes(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df["sample_id"].isin(self.observed_ids)]

    def test_nodes(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df["sample_id"].isin(self.held_out_ids)]

    def induced_edges(self, edges: pd.DataFrame) -> InducedEdges:
        e = accepted(edges)
        src_in = e["src"].isin(self.observed_ids)
        dst_in = e["dst"].isin(self.observed_ids)
        both = src_in & dst_in
        touching = ~both
        return InducedEdges(
            train=e[both],
            test=e[touching],
            # For a node-based design this is empty by construction; for the edge-based
            # sanity check it is the entire point of the diagnostic.
            leaked=e[both & self._nominal_test(e)],
            n_candidate=len(e),
        )

    def _nominal_test(self, e: pd.DataFrame) -> pd.Series:
        held = self.params.get("held_out_edge_ids")
        if held is None:
            return pd.Series(False, index=e.index)
        return e["edge_id"].isin(held)

    def render(self) -> str:
        lines = [
            f"### {self.kind} — {MECHANISM_QUESTIONS.get(self.kind, '')}",
            f"- {self.description}",
            f"- observed {len(self.observed_ids):,} nodes of "
            f"{len(self.observed_ids) + len(self.held_out_ids):,} ({self.observed_fraction:.1%}), "
            f"hiding unit = {self.unit}",
        ]
        return "\n".join(lines)


def _split_ids(df: pd.DataFrame, hidden_mask: pd.Series, unit: str) -> tuple[frozenset, frozenset]:
    if unit == "composition":
        comp = _comp_id(df)
        hidden_comps = set(comp[hidden_mask])
        hidden_mask = comp.isin(hidden_comps)
    elif unit != "row":
        raise ValueError(f"unit must be 'composition' or 'row', got {unit!r}")
    return (
        frozenset(df.loc[~hidden_mask, "sample_id"]),
        frozenset(df.loc[hidden_mask, "sample_id"]),
    )


def random_nodes(
    df: pd.DataFrame, observed_fraction: float = 0.5, seed: int = 0, unit: str = "composition"
) -> ObservationDesign:
    """MCAR over compositions: the "we only computed part of the grid" regime."""
    if unit == "composition":
        comps = _comp_id(df).drop_duplicates()
        rng = np.random.default_rng(seed)
        hide = set(comps.sample(frac=1.0 - observed_fraction, random_state=int(rng.integers(1 << 31))))
        hidden = _comp_id(df).isin(hide)
    else:
        rng = np.random.default_rng(seed)
        hidden = pd.Series(rng.random(len(df)) >= observed_fraction, index=df.index)
    obs, held = _split_ids(df, hidden, unit)
    return ObservationDesign(
        kind="random_nodes",
        observed_ids=obs,
        held_out_ids=held,
        description=f"uniformly hidden at random, target observed fraction {observed_fraction:.0%}",
        params={"observed_fraction": observed_fraction, "seed": seed},
        unit=unit,
    )


def held_out_fiber(
    df: pd.DataFrame, site: str, species: str | Sequence[str], unit: str = "composition"
) -> ObservationDesign:
    """Hide an entire slice of the product grid: every material containing `species`.

    This is the left-out element of rung 3.  Nothing containing the level
    is observed, so the model cannot have seen its absolute property anywhere.
    """
    species = [species] if isinstance(species, str) else list(species)
    hidden = df[f"site_{site}"].isin(species)
    obs, held = _split_ids(df, hidden, unit)
    return ObservationDesign(
        kind="fiber",
        observed_ids=obs,
        held_out_ids=held,
        description=f"entire {site}-site fiber(s) hidden: {', '.join(map(str, species))}",
        params={"site": site, "species": species},
        unit=unit,
    )


def held_out_block(
    df: pd.DataFrame, spec: Mapping[str, Sequence[str]], unit: str = "composition"
) -> ObservationDesign:
    """Hide a sub-box of the grid: the *intersection* of level sets at two or more sites.

    Every level involved still appears in training through other combinations, so this
    isolates combinatorial extrapolation from level extrapolation.  A model that has
    genuinely learned separable site responses plus their interaction should reach into
    the block; a model that has memorised the observed region cannot.
    """
    if len(spec) < 2:
        raise ValueError("a block needs level sets at two or more sites; one site is a fiber")
    hidden = pd.Series(True, index=df.index)
    for site, levels in spec.items():
        hidden &= df[f"site_{site}"].isin(list(levels))
    obs, held = _split_ids(df, hidden, unit)
    desc = " and ".join(f"{s} in {{{', '.join(map(str, v))}}}" for s, v in spec.items())
    return ObservationDesign(
        kind="block",
        observed_ids=obs,
        held_out_ids=held,
        description=f"combination block hidden: {desc}",
        params={"spec": {k: list(v) for k, v in spec.items()}},
        unit=unit,
    )


def random_edges_with_seen_endpoints(
    df: pd.DataFrame, edges: pd.DataFrame, test_fraction: float = 0.2, seed: int = 0
) -> ObservationDesign:
    """SANITY CHECK ONLY. Hides edges while every node stays observed.

    Kept so the leakage can be measured rather than argued about: the resulting design
    observes 100% of nodes, so `induced_edges().leakage_fraction` is 1.0 by construction
    and any score obtained under it is a subtraction of two memorised numbers.
    """
    e = accepted(edges)
    rng = np.random.default_rng(seed)
    mask = rng.random(len(e)) < test_fraction
    return ObservationDesign(
        kind="random_edges",
        observed_ids=frozenset(df["sample_id"]),
        held_out_ids=frozenset(),
        description=f"{int(mask.sum()):,} edges hidden, all {len(df):,} nodes still observed",
        params={"held_out_edge_ids": set(e.loc[mask, "edge_id"]), "test_fraction": test_fraction},
        unit="row",
    )


def leakage_table(designs: Iterable[ObservationDesign], edges: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side leakage audit — the table that belongs in the paper's setup section."""
    rows = []
    for d in designs:
        ind = d.induced_edges(edges)
        rows.append({
            "design": d.kind,
            "observed_nodes": len(d.observed_ids),
            "observed_fraction": d.observed_fraction,
            "train_edges": len(ind.train),
            "test_edges": len(ind.test),
            "leaked_edges": len(ind.leaked),
            "leakage_fraction": ind.leakage_fraction,
            "question": MECHANISM_QUESTIONS.get(d.kind, ""),
        })
    return pd.DataFrame(rows)


def fiber_sweep(
    df: pd.DataFrame, site: str, min_hidden: int = 1, unit: str = "composition"
) -> list[ObservationDesign]:
    """One design per level at `site` — the sweep the unseen-level error decomposition
    is validated against, one point per element."""
    out = []
    for sp in sorted(df[f"site_{site}"].dropna().unique()):
        d = held_out_fiber(df, site, sp, unit=unit)
        if len(d.held_out_ids) >= min_hidden and len(d.observed_ids) > 0:
            out.append(d)
    return out


def random_block_sweep(
    df: pd.DataFrame,
    sites: tuple[str, str] = ("A", "B"),
    block_size: tuple[int, int] = (5, 5),
    n_blocks: int = 10,
    seed: int = 0,
    unit: str = "composition",
) -> list[ObservationDesign]:
    rng = np.random.default_rng(seed)
    out = []
    levels = {s: sorted(df[f"site_{s}"].dropna().unique()) for s in sites}
    for _ in range(n_blocks):
        spec = {
            s: list(rng.choice(levels[s], size=min(k, len(levels[s])), replace=False))
            for s, k in zip(sites, block_size)
        }
        d = held_out_block(df, spec, unit=unit)
        if len(d.held_out_ids) > 0:
            out.append(d)
    return out
