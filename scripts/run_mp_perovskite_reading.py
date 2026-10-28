"""Read the ladder off the ABX3 perovskites that the Materials Project has computed.

The rule of the main text needs only which compositions have been computed. For every entry of
data/raw/mp_perovskites (fetched by download_mp_perovskites.py) the element present three times is
the anion X; entries whose X is a metal are anti-perovskites and are left out. Of the two cations,
B is the one whose six nearest anions are closest. Each distinct (A, B, X) is one computed
composition, and the grid is every combination of the elements seen at each site.

For the model of the main text (a mean, three site effects and three pair interactions) the
script counts compositions per identified parameter and the state of every direction: the
eigenvalues g of X'X in the sum-to-zero parameter space, undetermined for g = 0, barely determined
for 0 < g < 1 and determined many times over for g >= 1. The nonzero eigenvalues are those of
X P X' with P the projector onto that space, so the large parameter space is never formed densely.
It also counts the element pairs never computed together and, for the uncomputed compositions of
the grid, how many involve such a pair. The same function run on the complete alphabetical
Castelli subgrid recovers its 358 identified parameters, all determined.

Writes reports/mp_perovskite_reading.csv, reports/mp_perovskite_compositions.csv and
reports/AUDIT_MP_PEROVSKITE_READING.md.

Usage:  python3 scripts/run_mp_perovskite_reading.py
"""

from __future__ import annotations

import gzip
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pymatgen.core import Element, Structure

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prfield.models import hybrid  # noqa: E402
from prfield.report import write_report  # noqa: E402
from prfield.schema import SITES  # noqa: E402

RAW = Path("data/raw/mp_perovskites/mp_abx3_perovskites.json.gz")
PROV = Path("data/raw/mp_perovskites/provenance.json")
CSV = Path("reports/mp_perovskite_reading.csv")
COMPS = Path("reports/mp_perovskite_compositions.csv")
REPORT = Path("reports/AUDIT_MP_PEROVSKITE_READING.md")
PAIRS = tuple(a + b for a, b in combinations(SITES, 2))


def assign_sites(entry: dict) -> dict | None:
    s = Structure.from_dict(entry["structure"])
    amounts = s.composition.reduced_composition.get_el_amt_dict()
    x = [e for e, n in amounts.items() if abs(n - 3) < 1e-6]
    cations = [e for e, n in amounts.items() if abs(n - 1) < 1e-6]
    if len(x) != 1 or len(cations) != 2:
        return None
    x = x[0]
    if Element(x).is_metal:
        return None                                     # anti-perovskite: X is a metal
    mean6, cn = {}, {}
    for c in cations:
        site = next(i for i, sp_ in enumerate(s.species) if str(sp_) == c)
        d = sorted(n.nn_distance for n in s.get_neighbors(s[site], 5.0) if str(n.specie) == x)
        mean6[c] = float(np.mean(d[:6]))
        cn[c] = int(sum(v <= 1.2 * d[0] for v in d))
    b = min(cations, key=mean6.get)
    a = next(c for c in cations if c != b)
    return {"material_id": entry["material_id"], "formula": entry["formula"], "A": a, "B": b, "X": x,
            "mineral": entry["mineral"], "theoretical": entry["theoretical"],
            "e_above_hull": entry["e_above_hull"], "cn_B": cn[b], "cn_A": cn[a],
            "d_B": mean6[b], "d_A": mean6[a]}


def direction_states(comps: pd.DataFrame) -> dict:
    """Compositions per parameter and direction states for the main-text model."""
    frame = comps.rename(columns={s: f"site_{s}" for s in SITES})
    levels = {s: sorted(frame[f"site_{s}"].unique().tolist()) for s in SITES}
    X, slices = hybrid._design_and_slices(frame, levels)
    C = hybrid._constraints(levels, slices, X.shape[1]).toarray()
    rank_c = np.linalg.matrix_rank(C)
    p = X.shape[1] - rank_c
    Xd = X.toarray()
    XC = Xd @ C.T
    M = Xd @ Xd.T - XC @ np.linalg.pinv(C @ C.T) @ XC.T
    g = np.clip(np.linalg.eigvalsh((M + M.T) / 2), 0, None)
    g = g[g > 1e-8 * max(g.max(), 1)]
    n = len(frame)
    return {"n": n, "p": p, "n_over_p": n / p, "identified": len(g),
            "undetermined": p - len(g), "barely": int((g < 1).sum()), "determined": int((g >= 1).sum()),
            **{f"levels_{s}": len(levels[s]) for s in SITES}}


def pair_counts(comps: pd.DataFrame) -> dict:
    out = {}
    lv = {s: sorted(comps[s].unique()) for s in SITES}
    seen = {pq: set(map(tuple, comps[[pq[0], pq[1]]].drop_duplicates().to_numpy())) for pq in PAIRS}
    for pq in PAIRS:
        out[f"pairs_{pq}"] = len(seen[pq])
        out[f"possible_{pq}"] = len(lv[pq[0]]) * len(lv[pq[1]])
    grid = pd.MultiIndex.from_product([lv["A"], lv["B"], lv["X"]], names=list(SITES)).to_frame(index=False)
    done = set(map(tuple, comps[list(SITES)].to_numpy()))
    todo = grid[[tuple(r) not in done for r in grid.to_numpy()]]
    unseen = np.zeros(len(todo), dtype=bool)
    for pq in PAIRS:
        unseen |= np.array([(r[0], r[1]) not in seen[pq] for r in todo[[pq[0], pq[1]]].to_numpy()])
    out.update(grid=len(grid), uncomputed=len(todo), uncomputed_with_unseen_pair=int(unseen.sum()))
    ab_unseen = np.array([(a, b) not in seen["AB"] for a, b in todo[["A", "B"]].to_numpy()])
    out["uncomputed_with_unseen_AB"] = int(ab_unseen.sum())
    return out


def castelli_check() -> dict:
    frame = pd.read_parquet("data/interim/castelli_raw.parquet")
    a = sorted(frame.site_A.unique())[:14]
    b = sorted(frame.site_B.unique())[:14]
    sub = frame[frame.site_A.isin(a) & frame.site_B.isin(b)]
    comps = sub[["site_A", "site_B", "site_X"]].rename(columns=lambda c: c[-1])
    return direction_states(comps)


def main() -> None:
    with gzip.open(RAW, "rt") as fh:
        entries = json.load(fh)
    rows = [r for r in map(assign_sites, entries) if r is not None]
    assigned = pd.DataFrame(rows).sort_values("material_id")
    assigned.to_csv(COMPS, index=False)
    comps = assigned[list(SITES)].drop_duplicates().reset_index(drop=True)
    check = castelli_check()
    assert check["p"] == 358 and check["identified"] == 358 and check["undetermined"] == 0, check

    records = []
    for label, sub in [("all anions", comps)] + [
            (f"X = {x}", comps[comps.X == x]) for x, k in comps.X.value_counts().items() if k >= 30]:
        rec = {"subset": label, **direction_states(sub.reset_index(drop=True)), **pair_counts(sub)}
        per = {s: sub.groupby(s).size() for s in ("A", "B")}
        rec.update({f"median_per_{s}": float(per[s].median()) for s in per})
        rec.update({f"single_{s}": int((per[s] == 1).sum()) for s in per})
        records.append(rec)
    table = pd.DataFrame(records)
    table.to_csv(CSV, index=False)

    castelli = pd.read_parquet("data/interim/castelli_raw.parquet")
    missing = {s: sorted(set(castelli[f"site_{s}"].unique()) - set(comps[s])) for s in ("A", "B")}
    prov = json.loads(PROV.read_text())
    excluded = len(entries) - len(assigned)
    t0 = table.iloc[0]
    lines = [
        f"Source: Materials Project database {prov['database_version']}; "
        f"{len(entries)} ABC3 entries labelled perovskite, {excluded} anti-perovskites or unassigned left out, "
        f"{len(assigned)} entries, {len(comps)} distinct (A, B, X) compositions.",
        f"B sites with six nearest anions within 1.2 of the shortest distance: "
        f"{(assigned.cn_B == 6).mean():.1%}.",
        f"Castelli check (alphabetical subgrid, complete): p = {check['p']}, identified {check['identified']}, "
        f"undetermined {check['undetermined']}.",
        "",
        "## Reading for every subset",
        "",
        "```",
        table.to_string(index=False, float_format="%.3g"),
        "```",
        "",
        f"All anions: grid {t0.grid:,} compositions, coverage {t0.n / t0.grid:.2%}; "
        f"n/p = {t0.n_over_p:.3f}; undetermined {t0.undetermined / t0.p:.1%}, "
        f"barely {t0.barely / t0.p:.1%}, determined {t0.determined / t0.p:.1%} of the {t0.p:,} directions; "
        f"A-B pairs computed together {t0.pairs_AB / t0.possible_AB:.1%}; "
        f"uncomputed compositions with an A-B pair never computed together "
        f"{t0.uncomputed_with_unseen_AB / t0.uncomputed:.1%}, with any unseen pair "
        f"{t0.uncomputed_with_unseen_pair / t0.uncomputed:.1%}.",
        "",
        f"Castelli A-site elements never on the A site of an MP perovskite ({len(missing['A'])}): "
        f"{', '.join(missing['A'])}",
        f"Castelli B-site elements never on the B site of an MP perovskite ({len(missing['B'])}): "
        f"{', '.join(missing['B'])}",
        "",
        "Anion counts: " + ", ".join(f"{k} {v}" for k, v in comps.X.value_counts().items()),
    ]
    write_report(REPORT, "# MP perovskite reading\n\n" + "\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
