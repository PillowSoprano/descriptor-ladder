"""Rungs 1 and 2 of the descriptor ladder on a second complete DFT grid (Emery-Wolverton ABO3).

The same models and code paths as the Castelli analyses, on 73 x 73 ABO3 perovskites:

  rung 1  full grid, block holdouts hiding k of 73 A-site and B-site elements (k = 15, 27, 42, the
          hidden fractions of the Castelli blocks), five replicates: descriptor basis, free element
          effects, the similarity prior (weight 0.1) and gradient boosting, scored on every
          response that involves a hidden composition; plus the variance split into site effects
          and the A-B term, and the share of each site effect the ten descriptors capture.
  rung 2  seven 37 x 37 subgrids (alphabetical, five random, descriptor maximin), coverage 5 to
          90%, three seeds: untuned free effects, tuned ridge, and the similarity prior from real
          and from shuffled descriptors, every weight chosen by the sweep's inner selection, scored
          on responses with neither composition computed.

Writes reports/emery_rung1.csv, reports/emery_rung2.csv and reports/AUDIT_EMERY_REPLICATION.md.

Usage:  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 scripts/run_emery_replication.py --n-jobs 4
        python3 scripts/run_emery_replication.py --part summary   (rewrite the report from the CSVs)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_regularization_baselines as mh  # noqa: E402
from emery_grid import load_emery_grid  # noqa: E402
from prfield.descriptors import elemental as el  # noqa: E402
from prfield.graph.typed_graph import ComparabilityPolicy, accepted, build_edges  # noqa: E402
from prfield.metrics import response_metrics  # noqa: E402
from prfield.models import hybrid  # noqa: E402
from prfield.models.baselines import DifferenceFromNode, NodeModel  # noqa: E402
from prfield.report import write_report  # noqa: E402
from prfield.schema import SITES  # noqa: E402
from prfield.splits import observation as obs  # noqa: E402
from run_nonlinear_baseline import features, scalers  # noqa: E402

PROPS = ("heat_of_formation", "volume_per_fu")
POLICY = ComparabilityPolicy(max_volume_change_frac=None, max_rmsd_init_final=None,
                             require_same_coordination=False)
KS, REPS, SMOOTH, MAX_TEST = (15, 27, 42), 5, 0.1, 30_000
SUBGRID, COVERAGES, SEEDS = 37, (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90), (0, 1, 2)
RUNG1, RUNG2 = Path("reports/emery_rung1.csv"), Path("reports/emery_rung2.csv")
REPORT = Path("reports/AUDIT_EMERY_REPLICATION.md")


def descriptors(frame):
    return {s: el.physical_descriptors(sorted(frame[f"site_{s}"].unique().tolist()),
                                       "element" if s in ("A", "B") else "auto") for s in SITES}


def variance_split(grid, prop, desc):
    """Least-squares site effects, the A-B remainder, and the descriptor-captured share of each."""
    g = grid.dropna(subset=[prop])
    y = g[prop].to_numpy(float) - g[prop].mean()
    a_levels, b_levels = sorted(g.site_A.unique()), sorted(g.site_B.unique())
    xa = pd.get_dummies(g.site_A).reindex(columns=a_levels, fill_value=0).to_numpy(float)
    xb = pd.get_dummies(g.site_B).reindex(columns=b_levels, fill_value=0).to_numpy(float)
    beta = np.linalg.lstsq(np.hstack([xa, xb]), y, rcond=None)[0]
    ea, eb = beta[:len(a_levels)], beta[len(a_levels):]
    ea, eb = ea - ea.mean(), eb - eb.mean()
    fit = xa @ ea + xb @ eb
    total = float(np.sum((y - y.mean()) ** 2))
    out = {"property": prop, "A_share": float(np.sum((xa @ ea) ** 2)) / total,
           "B_share": float(np.sum((xb @ eb) ** 2)) / total,
           "AB_share": float(np.sum((y - fit) ** 2)) / total}
    for site, eff, levels in (("A", ea, a_levels), ("B", eb, b_levels)):
        m = np.nan_to_num(np.asarray(desc[site].matrix(levels, allow_missing=True), float))
        z = (m - m.mean(0)) / np.where(m.std(0) > 0, m.std(0), 1.0)
        z = np.hstack([np.ones((len(levels), 1)), z])
        proj = z @ np.linalg.lstsq(z, eff, rcond=None)[0]
        out[f"{site}_captured"] = float(np.sum(proj ** 2) / np.sum(eff ** 2))
    return out


def rung1_design(grid, desc, prop, edges, name, design):
    train = design.train_nodes(grid).dropna(subset=[prop]).reset_index(drop=True)
    te = design.induced_edges(edges).test
    if len(te) > MAX_TEST:
        te = te.iloc[np.random.default_rng(0).choice(len(te), MAX_TEST, replace=False)]
    y = te["delta"].to_numpy(float)
    idx = grid.set_index("sample_id")
    src, dst = idx.loc[te["src"]].reset_index(), idx.loc[te["dst"]].reset_index()
    mae = lambda pred: response_metrics(y, np.asarray(pred, float)).mae  # noqa: E731
    rec = {"property": prop, "design": name, "n_train": len(train), "n_test": len(te)}
    rec["basis"] = mae(DifferenceFromNode(NodeModel(desc, prop, interactions=True).fit(train)).predict_delta(te))
    sc = scalers(train, desc)
    hgb = HistGradientBoostingRegressor(random_state=0).fit(features(train, desc, sc), train[prop].to_numpy(float))
    rec["gradient_boosting"] = mae(hgb.predict(features(dst, desc, sc)) - hgb.predict(features(src, desc, sc)))
    for tag, g in (("free", 0.0), ("prior", SMOOTH)):
        eff = hybrid.fit_stage1(train, prop, smooth=g, desc=desc)
        maps, norm = hybrid.fit_stage2(eff, desc)
        model = hybrid.HybridModel(eff, maps, norm, desc)
        rec[tag] = mae(model.predict(dst) - model.predict(src))
    return rec


def subgrids(grid):
    all_a, all_b = sorted(grid.site_A.unique()), sorted(grid.site_B.unique())
    panels = [("alphabetical", all_a[:SUBGRID], all_b[:SUBGRID])]
    for seed in range(5):
        rng = np.random.default_rng(seed)
        panels.append((f"random-{seed}", sorted(rng.choice(all_a, SUBGRID, replace=False).tolist()),
                       sorted(rng.choice(all_b, SUBGRID, replace=False).tolist())))
    panels.append(("descriptor-maximin", mh.spread_levels(all_a, SUBGRID), mh.spread_levels(all_b, SUBGRID)))
    return panels


def rung2_panel(grid, panel, a_levels, b_levels):
    sub = grid[grid.site_A.isin(a_levels) & grid.site_B.isin(b_levels)].reset_index(drop=True)
    levels = {s: sorted(sub[f"site_{s}"].unique().tolist()) for s in SITES}
    desc = descriptors(sub)
    template, slices = hybrid._design_and_slices(sub.iloc[:1], levels)
    metrics = {"descriptor_metric": mh.metric_matrix(desc, levels, slices, template.shape[1]),
               "shuffled_metric": mh.metric_matrix(mh.descriptor_controls(desc)[0], levels, slices,
                                                   template.shape[1])}
    node_index = sub.set_index("sample_id")
    # identified parameters of the pair model on a single-anion grid: mean, two site effects, A-B
    n_params = 1 + (len(levels["A"]) - 1) + (len(levels["B"]) - 1) + (len(levels["A"]) - 1) * (len(levels["B"]) - 1)
    rows = []
    for prop in PROPS:
        edges = accepted(build_edges(sub, prop, POLICY)).reset_index(drop=True)
        for cov in COVERAGES:
            for seed in SEEDS:
                for r in mh.evaluate_outer(sub, prop, cov, seed, levels, edges, node_index, metrics,
                                           ("free_min_norm", "ordinary_ridge", "descriptor_metric",
                                            "shuffled_metric")):
                    rows.append({"panel": panel, **r, "n_over_p": r["n_train"] / n_params})
    return rows


def rung2_gains(r2):
    """Per subgrid and coverage: 1 - MAE / MAE(tuned ridge) for each seed, reduced to the median over
    seeds, as for the Castelli subgrids."""
    m = r2.pivot_table(index=["property", "panel", "coverage", "seed"], columns="model",
                       values="mae_both_hidden")
    g = pd.DataFrame({"prior": 1 - m.descriptor_metric / m.ordinary_ridge,
                      "shuffled": 1 - m.shuffled_metric / m.ordinary_ridge,
                      "untuned": 1 - m.free_min_norm / m.ordinary_ridge})
    return g.groupby(["property", "panel", "coverage"]).median().reset_index()


def write_summary() -> None:
    lines = ["# Descriptor ladder rungs 1 and 2 on the Emery-Wolverton ABO3 grid\n",
             "Generated by `scripts/run_emery_replication.py`. Errors are mean absolute errors of substitution "
             "responses.\n"]
    if RUNG1.exists():
        r1 = pd.read_csv(RUNG1)
        r1["k"] = r1.design.str.extract(r"block (\d+)A").astype(int)
        med = r1.groupby(["property", "k"])[["basis", "free", "prior", "gradient_boosting"]].median()
        lines += ["## Rung 1: block holdouts on the full grid (median over five replicates)\n",
                  "| property | hidden per site | basis | free effects | prior | gradient boosting |",
                  "|---|---|---|---|---|---|"]
        for (prop, k), r in med.iterrows():
            lines.append(f"| {prop} | {k} | {r.basis:.4f} | {r.free:.4f} | {r.prior:.4f} | {r.gradient_boosting:.4f} |")
        lines += ["", "| property | blocks with basis behind free effects | blocks with prior ahead of free effects |",
                  "|---|---|---|"]
        for prop, g in r1.groupby("property"):
            lines.append(f"| {prop} | {int((g.basis > g.free).sum())}/{len(g)} | {int((g.prior < g.free).sum())}/{len(g)} |")
        split = RUNG1.with_name("emery_variance_split.csv")
        if split.exists():
            lines += ["\n## Variance split\n", "| property | A | B | A-B | A captured | B captured |",
                      "|---|---|---|---|---|---|"]
            for _, r in pd.read_csv(split).iterrows():
                lines.append(f"| {r.property} | {r.A_share:.0%} | {r.B_share:.0%} | {r.AB_share:.0%} | "
                             f"{r.A_captured:.0%} | {r.B_captured:.0%} |")
    if RUNG2.exists():
        g = rung2_gains(pd.read_csv(RUNG2))
        by = g.groupby(["property", "coverage"]).agg(prior=("prior", "median"), positive=("prior", lambda x: (x > 0).sum()),
                                                     shuffled=("shuffled", "median"), untuned=("untuned", "median"))
        lines += ["\n## Rung 2: gain beyond tuned ridge on responses with neither composition computed "
                  "(median over seven subgrids)\n",
                  "| property | coverage | prior | subgrids with prior ahead | shuffled descriptors | untuned free effects |",
                  "|---|---|---|---|---|---|"]
        for (prop, cov), r in by.iterrows():
            lines.append(f"| {prop} | {cov:.0%} | {r.prior:+.1%} | {int(r.positive)}/7 | {r.shuffled:+.1%} | "
                         f"{r.untuned:+.1%} |")
    write_report(REPORT, "\n".join(lines))
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--part", choices=("all", "rung1", "rung2", "summary"), default="all")
    args = ap.parse_args()
    if args.part == "summary":
        write_summary()
        return
    t0 = time.time()
    grid = load_emery_grid()
    desc = descriptors(grid)
    if args.part in ("all", "rung1"):
        a_all, b_all = sorted(grid.site_A.unique()), sorted(grid.site_B.unique())
        rng = np.random.default_rng(0)
        designs = []
        for k in KS:
            for rep in range(REPS):
                aa = list(rng.choice(a_all, size=k, replace=False))
                bb = list(rng.choice(b_all, size=k, replace=False))
                designs.append((f"block {k}A x {k}B #{rep}", obs.held_out_block(grid, {"A": aa, "B": bb})))
        rows = []
        for prop in PROPS:
            edges = build_edges(grid, prop, POLICY)
            rows += Parallel(n_jobs=args.n_jobs, verbose=5)(
                delayed(rung1_design)(grid, desc, prop, edges, n, d) for n, d in designs)
        r1 = pd.DataFrame(rows)
        r1.to_csv(RUNG1, index=False)
        split = pd.DataFrame([variance_split(grid, p, desc) for p in PROPS])
        split.to_csv(RUNG1.with_name("emery_variance_split.csv"), index=False)
    if args.part in ("all", "rung2"):
        out = Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(rung2_panel)(grid, *p) for p in subgrids(grid))
        pd.DataFrame([r for rows in out for r in rows]).to_csv(RUNG2, index=False)
    write_summary()
    print(f"done in {(time.time() - t0) / 60:.0f} minutes")


if __name__ == "__main__":
    main()
