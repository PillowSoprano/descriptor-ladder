"""Substitution pairs and edit squares on the product space G_A x G_B x G_X (x G_S).

Two samples form a pair when they differ in exactly one site and are comparable: same prototype,
same functional for the property, and, where relaxation metadata exist, a structural mismatch below
threshold. A substitution response is meaningful only when the pair changed one thing, so
comparability is enforced here.

Pairs are directed by a fixed ordering of the two species labels, so Delta f is well defined; the
reverse pair is its negation and is not stored.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from ..schema import PROPERTY_FUNCTIONAL, SITES


@dataclass(frozen=True)
class ComparabilityPolicy:
    """What we are willing to call "the same structural context".

    Defaults are deliberately strict.  Audit A1 sweeps these thresholds and reports how
    the pair count decays with them.
    """

    require_same_prototype: bool = True
    require_same_functional: bool = True
    max_volume_change_frac: Optional[float] = 0.10   # per-endpoint, vs its own init structure
    max_rmsd_init_final: Optional[float] = 0.50      # A
    require_same_coordination: bool = True
    max_octahedral_distortion_delta: Optional[float] = None  # None = do not test
    allow_unknown_relaxation: bool = True            # keep, but tag edge as provisional

    def describe(self) -> str:
        bits = [
            f"prototype={'same' if self.require_same_prototype else 'any'}",
            f"functional={'same' if self.require_same_functional else 'any'}",
            f"dV/V<={self.max_volume_change_frac}",
            f"rmsd<={self.max_rmsd_init_final}",
            f"coordination={'same' if self.require_same_coordination else 'any'}",
            f"unknown_relax={'kept' if self.allow_unknown_relaxation else 'dropped'}",
        ]
        return " ".join(bits)


LOOSE = ComparabilityPolicy(
    max_volume_change_frac=None,
    max_rmsd_init_final=None,
    require_same_coordination=False,
)


def _endpoint_ok(row, policy: ComparabilityPolicy) -> tuple[bool, str]:
    """`row` is an itertuples record, not a Series — this runs once per candidate pair."""
    known = bool(getattr(row, "relaxation_known", False))
    if not known:
        if policy.allow_unknown_relaxation:
            return True, "unknown_relaxation"
        return False, "unknown_relaxation"
    if policy.max_volume_change_frac is not None:
        dv = getattr(row, "volume_change_frac", None)
        if dv is not None and pd.notna(dv) and abs(float(dv)) > policy.max_volume_change_frac:
            return False, "volume_change"
    if policy.max_rmsd_init_final is not None:
        r = getattr(row, "rmsd_init_final", None)
        if r is not None and pd.notna(r) and float(r) > policy.max_rmsd_init_final:
            return False, "rmsd"
    return True, "ok"


def build_edges(
    df: pd.DataFrame,
    prop: str,
    policy: ComparabilityPolicy = ComparabilityPolicy(),
    sites: Iterable[str] = SITES,
) -> pd.DataFrame:
    """All substitution pairs for one property.

    Returns one row per edge with columns:
        edge_id, site, src, dst, src_species, dst_species,
        site_A/site_B/site_X (the *shared* coordinates, with the varied one as NaN),
        prototype_id, delta (f(dst) - f(src)), provisional, reject_reason
    Rejected candidate pairs are returned too (delta = NaN, reject_reason set) so that
    audit A1 can count what the policy removed instead of silently shrinking the graph.
    """
    func_col = PROPERTY_FUNCTIONAL.get(prop)
    work = df[df[prop].notna()].copy()
    if work.empty:
        return _empty_edges()

    records: list[dict] = []
    for site in sites:
        others = [f"site_{s}" for s in SITES if s != site]
        group_cols = others + (["prototype_id"] if policy.require_same_prototype else [])
        for _, block in work.groupby(group_cols, dropna=False, sort=False):
            if len(block) < 2:
                continue
            rows = list(block.itertuples(index=False))
            for r1, r2 in combinations(rows, 2):
                s1 = getattr(r1, f"site_{site}")
                s2 = getattr(r2, f"site_{site}")
                if s1 == s2:
                    continue  # same species, different polymorph: not a substitution edge
                # canonical direction: lexicographic on species label
                src, dst = (r1, r2) if str(s1) < str(s2) else (r2, r1)
                src_s = getattr(src, f"site_{site}")
                dst_s = getattr(dst, f"site_{site}")

                reason = "ok"
                if policy.require_same_functional and func_col:
                    if getattr(src, func_col, None) != getattr(dst, func_col, None):
                        reason = "functional_mismatch"
                if reason == "ok" and policy.require_same_coordination:
                    c1, c2 = getattr(src, "b_x_coordination", pd.NA), getattr(dst, "b_x_coordination", pd.NA)
                    if pd.notna(c1) and pd.notna(c2) and c1 != c2:
                        reason = "coordination_mismatch"
                provisional = False
                if reason == "ok":
                    for endpoint in (src, dst):
                        ok, why = _endpoint_ok(pd.Series(endpoint._asdict()), policy)
                        if not ok:
                            reason = why
                            break
                        if why == "unknown_relaxation":
                            provisional = True
                if reason == "ok" and policy.max_octahedral_distortion_delta is not None:
                    d1 = getattr(src, "octahedral_distortion", pd.NA)
                    d2 = getattr(dst, "octahedral_distortion", pd.NA)
                    if pd.notna(d1) and pd.notna(d2):
                        if abs(float(d1) - float(d2)) > policy.max_octahedral_distortion_delta:
                            reason = "distortion_mismatch"

                delta = (
                    float(getattr(dst, prop)) - float(getattr(src, prop))
                    if reason == "ok"
                    else np.nan
                )
                rec = {
                    "edge_id": f"{site}|{src.sample_id}->{dst.sample_id}",
                    "site": site,
                    "src": src.sample_id,
                    "dst": dst.sample_id,
                    "src_species": src_s,
                    "dst_species": dst_s,
                    "prototype_id": src.prototype_id,
                    "property": prop,
                    "delta": delta,
                    "provisional": provisional,
                    "reject_reason": reason,
                }
                for s in SITES:
                    rec[f"site_{s}"] = pd.NA if s == site else getattr(src, f"site_{s}")
                records.append(rec)

    return pd.DataFrame.from_records(records) if records else _empty_edges()


def _empty_edges() -> pd.DataFrame:
    cols = [
        "edge_id", "site", "src", "dst", "src_species", "dst_species", "prototype_id",
        "property", "delta", "provisional", "reject_reason", "site_A", "site_B", "site_X",
    ]
    return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})


def accepted(edges: pd.DataFrame) -> pd.DataFrame:
    return edges[edges["reject_reason"] == "ok"]


def build_squares(df: pd.DataFrame, prop: str, site1: str, site2: str,
                  policy: ComparabilityPolicy = ComparabilityPolicy()) -> pd.DataFrame:
    """Complete 2x2 edit squares for the mixed second difference D_{site2} D_{site1} f.

    A square is four samples sharing the third site and prototype, with species
    (p,q), (p',q), (p,q'), (p',q').  Returns the mixed difference
        f(p',q') - f(p',q) - f(p,q') + f(p,q)
    which is the pair-interaction contrast.
    """
    if site1 == site2:
        raise ValueError("a square needs two distinct sites")
    fixed = [s for s in SITES if s not in (site1, site2)]
    func_col = PROPERTY_FUNCTIONAL.get(prop)
    work = df[df[prop].notna()].copy()
    group_cols = [f"site_{s}" for s in fixed] + ["prototype_id"]

    out: list[dict] = []
    for keys, block in work.groupby(group_cols, dropna=False, sort=False):
        if len(block) < 4:
            continue
        table = block.set_index([f"site_{site1}", f"site_{site2}"])[prop]
        table = table[~table.index.duplicated(keep="first")]
        levels1 = sorted({i[0] for i in table.index})
        levels2 = sorted({i[1] for i in table.index})
        if func_col is not None and policy.require_same_functional:
            funcs = set(block[func_col].dropna().unique())
            if len(funcs) > 1:
                continue
        for p, pp in combinations(levels1, 2):
            for q, qq in combinations(levels2, 2):
                corners = [(p, q), (pp, q), (p, qq), (pp, qq)]
                if not all(c in table.index for c in corners):
                    continue
                v = {c: float(table.loc[c]) for c in corners}
                out.append({
                    "property": prop,
                    "site1": site1, "site2": site2,
                    f"{site1}_from": p, f"{site1}_to": pp,
                    f"{site2}_from": q, f"{site2}_to": qq,
                    **{f"site_{s}": k for s, k in zip(fixed, keys if isinstance(keys, tuple) else (keys,))},
                    "prototype_id": (keys[-1] if isinstance(keys, tuple) else keys),
                    "mixed_delta": v[(pp, qq)] - v[(pp, q)] - v[(p, qq)] + v[(p, q)],
                    "d1_at_q": v[(pp, q)] - v[(p, q)],
                    "d1_at_qq": v[(pp, qq)] - v[(p, qq)],
                })
    return pd.DataFrame.from_records(out)


def square_completeness(df: pd.DataFrame, prop: str, site1: str, site2: str) -> dict:
    """How many of the theoretically possible 2x2 squares actually exist."""
    fixed = [f"site_{s}" for s in SITES if s not in (site1, site2)]
    work = df[df[prop].notna()]
    stats = {"possible": 0, "complete": 0, "contexts": 0}
    for _, block in work.groupby(fixed + ["prototype_id"], dropna=False, sort=False):
        l1 = block[f"site_{site1}"].nunique()
        l2 = block[f"site_{site2}"].nunique()
        if l1 < 2 or l2 < 2:
            continue
        stats["contexts"] += 1
        stats["possible"] += (l1 * (l1 - 1) // 2) * (l2 * (l2 - 1) // 2)
    sq = build_squares(df, prop, site1, site2)
    stats["complete"] = len(sq)
    stats["completeness"] = stats["complete"] / stats["possible"] if stats["possible"] else float("nan")
    return stats


def connected_components(edges: pd.DataFrame, nodes: Iterable[str]) -> pd.Series:
    """Component label per node over the accepted-edge graph (union-find)."""
    parent: dict[str, str] = {n: n for n in nodes}

    def find(a: str) -> str:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for row in accepted(edges).itertuples(index=False):
        ra, rb = find(row.src), find(row.dst)
        if ra != rb:
            parent[ra] = rb
    return pd.Series({n: find(n) for n in parent}, name="component")


def incidence_matrix(edges: pd.DataFrame, node_index: dict[str, int]) -> np.ndarray:
    """Edge-node incidence D with D[e, dst] = +1, D[e, src] = -1 (so D f = delta)."""
    acc = accepted(edges)
    D = np.zeros((len(acc), len(node_index)))
    for i, row in enumerate(acc.itertuples(index=False)):
        D[i, node_index[row.dst]] = 1.0
        D[i, node_index[row.src]] = -1.0
    return D
