"""Block holdouts on the complete 52 x 52 x 7 Castelli grid.

The categorical design of the full grid has 1 + 52 + 52 + 7 + 52*52 + 52*7 + 52*7 = 3544 columns, so
its normal equations are a dense 3544 x 3544 system and the subgrid comparison can be repeated on
the full grid. Four models per design:

  desc-node   the descriptor basis
  cat-node    free element effects
  cat-smooth  free element effects with the similarity prior
  hgb         gradient boosting on the same descriptors

Blocks hide the same fractions of the grid as on the subgrid: k in {11, 19, 30} of 52 against
{3, 5, 8} of 14, five replicates each, one seeded generator.

Writes reports/AUDIT_FULL_GRID.md and reports/full_grid.csv.

Usage:  PYTHONPATH=src python3 scripts/run_full_grid.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prfield.descriptors import elemental as el  # noqa: E402
from prfield.graph.typed_graph import ComparabilityPolicy, build_edges  # noqa: E402
from prfield.metrics import response_metrics  # noqa: E402
from prfield.models import hybrid  # noqa: E402
from prfield.models.baselines import DifferenceFromNode, NodeModel  # noqa: E402
from prfield.report import write_report  # noqa: E402
from prfield.splits import observation as obs  # noqa: E402
from run_nonlinear_baseline import features, scalers  # noqa: E402

REPORT = Path("reports/AUDIT_FULL_GRID.md")
CSV = Path("reports/full_grid.csv")
PROPS = ("heat_of_formation", "volume_per_fu")
POLICY = ComparabilityPolicy(max_volume_change_frac=None, max_rmsd_init_final=None,
                             require_same_coordination=False)
SMOOTH = 0.1          # the value that was best on the subgrid; not retuned here
KS = (11, 19, 30)     # the same hidden fractions as 3, 5, 8 out of 14
MAX_TEST = 30_000


def main() -> None:
    df = pd.read_parquet("data/interim/castelli_raw.parquet")
    A, B = sorted(df.site_A.unique()), sorted(df.site_B.unique())
    grid = df.reset_index(drop=True)
    desc = {s: el.physical_descriptors(sorted(grid[f"site_{s}"].unique().tolist()),
                                       "element" if s in ("A", "B") else "auto")
            for s in ("A", "B", "X")}
    print(f"{len(grid)} nodes, {len(A)} x {len(B)} x "
          f"{grid.site_X.nunique()} grid", flush=True)

    rng = np.random.default_rng(0)
    designs = []
    for k in KS:
        for rep in range(5):
            aa = list(rng.choice(A, size=k, replace=False))
            bb = list(rng.choice(B, size=k, replace=False))
            designs.append((f"block {k}A x {k}B #{rep}",
                            obs.held_out_block(grid, {"A": aa, "B": bb})))

    rows, t0 = [], time.time()
    for prop in PROPS:
        edges = build_edges(grid, prop, POLICY)
        print(f"\n== {prop}: {len(edges):,} candidate edges ==", flush=True)
        for name, d in designs:
            train = d.train_nodes(grid).dropna(subset=[prop]).reset_index(drop=True)
            te = d.induced_edges(edges).test
            if len(te) == 0 or len(train) == 0:
                continue
            if len(te) > MAX_TEST:
                te = te.iloc[np.random.default_rng(0).choice(len(te), MAX_TEST, replace=False)]
            y = te["delta"].to_numpy(dtype=float)
            idx = grid.set_index("sample_id")
            src, dst = idx.loc[te["src"]].reset_index(), idx.loc[te["dst"]].reset_index()

            def mae(pred):
                return response_metrics(y, np.asarray(pred, float)).mae

            rec = {"property": prop, "design": name, "n_train": len(train), "n_test": len(te)}
            t = time.time()
            nm = NodeModel(desc, prop, interactions=True).fit(train)
            rec["desc_node"] = mae(DifferenceFromNode(nm).predict_delta(te))
            sc = scalers(train, desc)
            hgb = HistGradientBoostingRegressor(random_state=0).fit(
                features(train, desc, sc), train[prop].to_numpy(float))
            rec["hgb"] = mae(hgb.predict(features(dst, desc, sc))
                             - hgb.predict(features(src, desc, sc)))
            for tag, g in (("cat_node", 0.0), ("cat_smooth", SMOOTH)):
                eff = hybrid.fit_stage1(train, prop, smooth=g, desc=desc)
                maps, norm = hybrid.fit_stage2(eff, desc)
                m = hybrid.HybridModel(eff, maps, norm, desc)
                rec[tag] = mae(m.predict(dst) - m.predict(src))
            rec["seconds"] = time.time() - t
            rows.append(rec)
            print(f"  {name:20s} n={len(train):6d}  {rec['seconds']:.0f}s  "
                  f"desc {rec['desc_node']:.4f}  cat {rec['cat_node']:.4f}  "
                  f"smooth {rec['cat_smooth']:.4f}  hgb {rec['hgb']:.4f}", flush=True)

    d = pd.DataFrame(rows)
    CSV.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(CSV, index=False)

    from scipy import stats
    out = ["# The block comparison on the complete grid\n"]
    out.append(
        "_Generated by `scripts/run_full_grid.py`._ The full $52\\times52\\times7$ grid, 18,928 "
        "compositions, against the $14\\times14\\times7$ subgrid the sweep uses. The categorical "
        "design has 3,544 columns here, so the normal equations are about 100 MB; the subgrid is "
        "a choice about cost, not a feasibility limit. Blocks hide the same fraction of the grid "
        "as on the subgrid. The smoothing strength is the subgrid's best value, carried over "
        "rather than retuned.\n")
    out.append("| property | descriptor linear | free categorical | + descriptor prior "
               "| gradient boosting | median fit time |")
    out.append("|---|---|---|---|---|---|")
    for prop in PROPS:
        g = d[d["property"] == prop]
        if g.empty:
            continue
        out.append(f"| {prop.replace('_',' ')} | {g.desc_node.median():.4f} "
                   f"| {g.cat_node.median():.4f} | {g.cat_smooth.median():.4f} "
                   f"| {g.hgb.median():.4f} | {g.seconds.median():.0f}s |")
    out.append("\n## Contrasts\n")
    out.append("| property | categorical vs descriptor | prior vs categorical "
               "| prior vs gradient boosting |")
    out.append("|---|---|---|---|")
    for prop in PROPS:
        g = d[d["property"] == prop]
        if g.empty:
            continue
        def rel(a, b):
            v = 100 * (g[a].to_numpy() / g[b].to_numpy() - 1)
            p = float(stats.wilcoxon(g[a] - g[b])[1])
            return f"{np.median(v):+.1f}% (p={p:.4f})"
        out.append(f"| {prop.replace('_',' ')} | {rel('cat_node','desc_node')} "
                   f"| {rel('cat_smooth','cat_node')} | {rel('cat_smooth','hgb')} |")
    write_report(REPORT, "\n".join(out))
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
