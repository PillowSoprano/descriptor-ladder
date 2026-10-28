"""Does the similarity prior's gain need a random sample of compositions?

Projects choose what to compute by chemical preference. Three such choices are imitated on the
seven 14 x 14 x 7 subgrids, next to a random choice of the same size:

  conventional valence first   compositions whose common oxidation states (pymatgen) balance the
                               anion charge are computed first, in random order
  oxides first                 anions are filled in the order O3, O2N, O2F, O2S, ON2, OFN, N3
  core and periphery           6 of the 14 A-site and 6 of the 14 B-site elements form a core whose
                               compositions are drawn with weight 9, the rest with weight 1
  random                       uniformly random

No choice reads a property value. For each design (5, 10, 20 and 30% coverage, seeds 3 to 5) the
tuned ridge and the similarity prior are fitted with the coverage sweep's own inner selection, and
both are scored on the substitution responses with neither composition computed.

Writes reports/project_like_designs.csv and reports/AUDIT_PROJECT_LIKE_DESIGNS.md.

Usage:  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 scripts/run_project_like_designs.py [--n-jobs 4]
        python3 scripts/run_project_like_designs.py --summary-only   (rewrite the report from the CSV)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_regularization_baselines as mh  # noqa: E402
from prfield.descriptors import elemental as el  # noqa: E402
from prfield.graph.typed_graph import accepted, build_edges  # noqa: E402
from prfield.models import hybrid  # noqa: E402
from prfield.report import write_report  # noqa: E402
from prfield.schema import SITES  # noqa: E402

CSV = Path("reports/project_like_designs.csv")
REPORT = Path("reports/AUDIT_PROJECT_LIKE_DESIGNS.md")
RULES = ("conventional_valence_first", "oxides_first", "core_and_periphery", "random")
COVERAGES = (0.05, 0.10, 0.20, 0.30)
SEEDS = (3, 4, 5)
PROPERTIES = ("heat_of_formation", "volume_per_fu")
MODELS = ("ordinary_ridge", "descriptor_metric")
ANION_CHARGE = {"O3": 6, "O2N": 7, "ON2": 8, "N3": 9, "O2F": 5, "OFN": 6, "O2S": 6}
ANION_ORDER = ("O3", "O2N", "O2F", "O2S", "ON2", "OFN", "N3")


def conventional_mask(sub: pd.DataFrame) -> np.ndarray:
    from pymatgen.core.periodic_table import Element
    ox = {e: [q for q in Element(e).common_oxidation_states if q > 0]
          for e in set(sub.site_A) | set(sub.site_B)}
    return np.array([any(qa + qb == ANION_CHARGE[x] for qa in ox[a] for qb in ox[b])
                     for a, b, x in zip(sub.site_A, sub.site_B, sub.site_X)])


def select(sub: pd.DataFrame, rule: str, m: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of the m computed compositions; no property value is read."""
    if rule == "conventional_valence_first":
        conv = conventional_mask(sub)
        first, rest = rng.permutation(np.flatnonzero(conv)), rng.permutation(np.flatnonzero(~conv))
        return np.concatenate([first, rest])[:m]
    if rule == "oxides_first":
        order = [rng.permutation(np.flatnonzero(sub.site_X.to_numpy() == x)) for x in ANION_ORDER]
        return np.concatenate(order)[:m]
    if rule == "core_and_periphery":
        core = {s: set(rng.choice(sorted(sub[f"site_{s}"].unique()), 6, replace=False)) for s in ("A", "B")}
        w = np.where(sub.site_A.isin(core["A"]) & sub.site_B.isin(core["B"]), 9.0, 1.0)
        return rng.choice(len(sub), size=m, replace=False, p=w / w.sum())
    if rule == "random":
        return rng.permutation(len(sub))[:m]
    raise KeyError(rule)


def task(frame, pi, panel, a_levels, b_levels, ri, rule, cov, seed):
    sub = frame[frame.site_A.isin(a_levels) & frame.site_B.isin(b_levels)].reset_index(drop=True)
    levels = {s: sorted(sub[f"site_{s}"].unique().tolist()) for s in SITES}
    desc = {s: el.physical_descriptors(levels[s], "element" if s in ("A", "B") else "auto") for s in SITES}
    template, slices = hybrid._design_and_slices(sub.iloc[:1], levels)
    metric_matrices = {"descriptor_metric": mh.metric_matrix(desc, levels, slices, template.shape[1])}
    node_index = sub.set_index("sample_id")
    rng = np.random.default_rng(10_000 * pi + 100 * ri + seed)
    computed = sub.iloc[select(sub, rule, int(round(cov * len(sub))), rng)]
    hidden = set(sub.sample_id) - set(computed.sample_id)
    rows = []
    for prop in PROPERTIES:
        edges = accepted(build_edges(sub, prop, mh.POLICY)).reset_index(drop=True)
        train = computed.dropna(subset=[prop]).reset_index(drop=True)
        test = mh.edge_subset(edges, hidden, both_hidden=True)
        mae = {}
        for model in MODELS:
            option, _, _ = mh.select_candidate(train, prop, levels, edges, node_index, model,
                                               metric_matrices, seed)
            beta = mh.fit_candidate(train, prop, levels, model, option, metric_matrices)
            mae[model] = mh.edge_mae(beta, levels, node_index, test)
        rows.append({"panel": panel, "rule": rule, "coverage": cov, "seed": seed, "property": prop,
                     "n_train": len(train), "n_test": len(test), "mae_ridge": mae["ordinary_ridge"],
                     "mae_prior": mae["descriptor_metric"],
                     "gain": 1 - mae["descriptor_metric"] / mae["ordinary_ridge"]})
    return rows


def write_summary(res: pd.DataFrame, minutes: float | None) -> None:
    agg = dict(designs=("gain", "size"), prior_beats_ridge=("gain", lambda g: (g > 0).mean()),
               mean_gain=("gain", "mean"))
    overall = res.groupby(["property", "rule"]).agg(**agg)
    by_cov = res.groupby(["property", "rule", "coverage"]).agg(**agg)
    lines = ["# Similarity prior against tuned ridge in project-like designs\n",
             "Generated by `scripts/run_project_like_designs.py`. `gain` is 1 - MAE(prior) / MAE(tuned "
             "ridge) on the substitution responses with neither composition computed.\n",
             "## By choice of compositions\n", "| property | choice | designs | prior beats ridge | mean gain |",
             "|---|---|---|---|---|"]
    for (prop, rule), r in overall.iterrows():
        lines.append(f"| {prop} | {rule} | {int(r.designs)} | {r.prior_beats_ridge:.0%} | {r.mean_gain:+.1%} |")
    lines += ["\n## By choice and coverage\n",
              "| property | choice | coverage | designs | prior beats ridge | mean gain |", "|---|---|---|---|---|---|"]
    for (prop, rule, cov), r in by_cov.iterrows():
        lines.append(f"| {prop} | {rule} | {cov:.0%} | {int(r.designs)} | {r.prior_beats_ridge:.0%} | "
                     f"{r.mean_gain:+.1%} |")
    if minutes is not None:
        lines.append(f"\nRuntime: {minutes:.0f} minutes.")
    write_report(REPORT, "\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--summary-only", action="store_true", help="rewrite the report from the existing CSV")
    args = ap.parse_args()
    if args.summary_only:
        write_summary(pd.read_csv(CSV), None)
        return
    t0 = time.time()
    frame = pd.read_parquet("data/interim/castelli_raw.parquet").drop(columns=["structure_blob"], errors="ignore")
    panels = mh.panel_definitions(frame)
    tasks = [(pi, *panels[pi], ri, rule, cov, seed) for pi in range(len(panels))
             for ri, rule in enumerate(RULES) for cov in COVERAGES for seed in SEEDS]
    out = Parallel(n_jobs=args.n_jobs, verbose=5)(delayed(task)(frame, *t) for t in tasks)
    res = pd.DataFrame([r for rows in out for r in rows])
    res.to_csv(CSV, index=False)
    write_summary(res, (time.time() - t0) / 60)
    print(f"wrote {CSV} and {REPORT}")


if __name__ == "__main__":
    main()
