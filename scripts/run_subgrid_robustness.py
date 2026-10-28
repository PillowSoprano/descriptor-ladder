"""Does the block conclusion depend on which 14 x 14 x 7 subgrid was used?

The end-to-end sweep runs on one subgrid, the first fourteen A and B species alphabetically.
That rule is non-adaptive, so it cannot have been chosen on a result, but it establishes nothing
about chemical representativeness, and the full grid carries 52 levels per site. This runs the
block contrasts, which carry the paper's central claim, on several subgrids chosen by rules
fixed before any of them was evaluated.

Seven subgrids, three rules:

  alphabetical   the first fourteen of each site in sorted order, the sweep's own choice
  random-0..4    fourteen drawn uniformly without replacement, seeds 0 to 4
  spread         farthest-point sampling in standardised descriptor space, started from the
                 level nearest the centroid and thereafter deterministic

The last is included because a uniform draw and an alphabetical prefix can both under-cover the
descriptor space, and the question is whether the conclusion survives a subgrid deliberately
chosen to spread over it.

Only the block family is run: it is where the contrasts are significant and where the claim
lives. Contrasts, penalties, designs and seeds are otherwise identical to the main factorial.

Writes reports/AUDIT_SUBGRID_ROBUSTNESS.md and reports/subgrid_robustness.csv.

Usage:  PYTHONPATH=src python3 scripts/run_subgrid_robustness.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prfield.descriptors import elemental as el  # noqa: E402
from prfield.graph.typed_graph import ComparabilityPolicy, build_edges  # noqa: E402
from prfield.metrics import response_metrics  # noqa: E402
from prfield.models import hybrid  # noqa: E402
from prfield.models.baselines import (  # noqa: E402
    DifferenceFromNode, NodeModel, fit_node_model_response_loss,
)
from prfield.report import write_report  # noqa: E402
from prfield.splits import observation as obs  # noqa: E402

REPORT = Path("reports/AUDIT_SUBGRID_ROBUSTNESS.md")
CSV = Path("reports/subgrid_robustness.csv")
PROPS = ("heat_of_formation", "volume_per_fu")
POLICY = ComparabilityPolicy(max_volume_change_frac=None, max_rmsd_init_final=None,
                             require_same_coordination=False)
K = 14


def spread_levels(labels, k):
    """Farthest-point sampling in standardised descriptor space; deterministic."""
    d = el.physical_descriptors(list(labels), "element")
    M = np.nan_to_num(np.asarray(d.matrix(list(labels), allow_missing=True), float))
    Z = (M - M.mean(0)) / np.where(M.std(0) > 0, M.std(0), 1.0)
    chosen = [int(np.argmin(((Z - Z.mean(0)) ** 2).sum(1)))]
    while len(chosen) < k:
        dist = np.min(((Z[:, None, :] - Z[None, chosen, :]) ** 2).sum(-1), axis=1)
        dist[chosen] = -1
        chosen.append(int(np.argmax(dist)))
    return sorted(labels[i] for i in chosen)


def subgrids(allA, allB):
    out = [("alphabetical", sorted(allA)[:K], sorted(allB)[:K])]
    for seed in range(5):
        r = np.random.default_rng(seed)
        out.append((f"random-{seed}",
                    sorted(r.choice(allA, K, replace=False).tolist()),
                    sorted(r.choice(allB, K, replace=False).tolist())))
    out.append(("spread", spread_levels(np.array(sorted(allA)), K),
                spread_levels(np.array(sorted(allB)), K)))
    return out


def block_designs(sub, A, B):
    """Fifteen blocks by the same rule and seed as the main sweep.

    The draws cannot be identical across subgrids, since they are draws from different level
    sets; what is held fixed is the rule, the sizes and the seed.
    """
    rng = np.random.default_rng(0)
    out = []
    for k in (3, 5, 8):
        for rep in range(5):
            aa = list(rng.choice(A, size=k, replace=False))
            bb = list(rng.choice(B, size=k, replace=False))
            out.append((f"block {k}A x {k}B #{rep}", obs.held_out_block(sub, {"A": aa, "B": bb})))
    return out


def main() -> None:
    df = pd.read_parquet("data/interim/castelli_raw.parquet")
    allA, allB = sorted(df.site_A.unique()), sorted(df.site_B.unique())
    rows, t0 = [], time.time()

    for tag, A, B in subgrids(allA, allB):
        sub = df[df.site_A.isin(A) & df.site_B.isin(B)].reset_index(drop=True)
        desc = {s: el.physical_descriptors(sorted(sub[f"site_{s}"].unique().tolist()),
                                           "element" if s in ("A", "B") else "auto")
                for s in ("A", "B", "X")}
        for prop in PROPS:
            edges = build_edges(sub, prop, POLICY)
            for name, d in block_designs(sub, A, B):
                train = d.train_nodes(sub)
                te = d.induced_edges(edges).test
                if len(te) == 0 or len(train) == 0:
                    continue
                y = te["delta"].to_numpy(dtype=float)
                idx = sub.set_index("sample_id")
                src, dst = idx.loc[te["src"]].reset_index(), idx.loc[te["dst"]].reset_index()

                def mae(pred):
                    return response_metrics(y, np.asarray(pred, float)).mae

                nm = NodeModel(desc, prop, interactions=True).fit(train)
                cn = hybrid.fit(train, prop, desc)
                cr = hybrid.fit_response(train, prop, desc)
                rows.append({
                    "subgrid": tag, "property": prop, "design": name,
                    "desc_node": mae(DifferenceFromNode(nm).predict_delta(te)),
                    "desc_resp": mae(DifferenceFromNode(fit_node_model_response_loss(
                        desc, prop, train)).predict_delta(te)),
                    "cat_node": mae(cn.predict(dst) - cn.predict(src)),
                    "cat_resp": mae(cr.predict(dst) - cr.predict(src))})
        print(f"  {tag:14s} done  {time.time()-t0:.0f}s", flush=True)

    d = pd.DataFrame(rows)
    CSV.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(CSV, index=False)

    out = ["# Does the block conclusion depend on the subgrid?\n"]
    out.append(
        "_Generated by `scripts/run_subgrid_robustness.py`._ The block contrasts, which carry "
        "the paper's central claim, on seven subgrids chosen by rules fixed before any was "
        "evaluated: the sweep's alphabetical prefix, five uniform draws, and one farthest-point "
        "spread in descriptor space.\n")
    out.append("| property | subgrid | C1 potential, node | C2 potential, response "
               "| C3 loss, descriptor | C4 loss, categorical |")
    out.append("|---|---|---|---|---|---|")
    for prop in PROPS:
        for tag in d[d["property"] == prop]["subgrid"].unique():
            g = d[(d["property"] == prop) & (d["subgrid"] == tag)]
            c1 = 100 * np.median((g.cat_node - g.desc_node) / g.desc_node)
            c2 = 100 * np.median((g.cat_resp - g.desc_resp) / g.desc_resp)
            c3 = 100 * np.median((g.desc_resp - g.desc_node) / g.desc_node)
            c4 = 100 * np.median((g.cat_resp - g.cat_node) / g.cat_node)
            out.append(f"| {prop.replace('_',' ')} | {tag} | {c1:+.1f}% | {c2:+.1f}% "
                       f"| {c3:+.1f}% | {c4:+.1f}% |")
    write_report(REPORT, "\n".join(out))
    print(f"\nwrote {REPORT} and {CSV}")


if __name__ == "__main__":
    main()
