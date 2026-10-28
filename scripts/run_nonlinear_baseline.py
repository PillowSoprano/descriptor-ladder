"""Does a non-linear descriptor model close the gap to free categorical effects?

The subspace lemma is about *linear* designs: a composition-level descriptor design is X M, so
its column space sits inside the categorical one. A tree ensemble on the same descriptors is not
in that subspace. It can represent interactions the second-order linear descriptor model cannot,
so the lemma says nothing about it and the paper's claim has to be limited to the class the
lemma covers. Whether the gap survives a strong non-linear descriptor model is an empirical
question, and this answers it.

Two ensembles, both fitted to node properties from the same standardised descriptors the linear
model uses, then differenced:

  rf     random forest, 400 trees, defaults otherwise
  hgb    histogram gradient boosting, defaults

They are given the raw per-site descriptor blocks rather than the explicit outer products, since
finding interactions is what they are for. No hyperparameter is tuned on any held-out result;
tuning them and not the linear models would bias the comparison the other way.

Run on the block designs, which is where the paper's central contrast lives.

Writes reports/AUDIT_NONLINEAR_BASELINE.md and reports/nonlinear_baseline.csv.

Usage:  PYTHONPATH=src python3 scripts/run_nonlinear_baseline.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prfield.descriptors import elemental as el  # noqa: E402
from prfield.graph.typed_graph import ComparabilityPolicy, build_edges  # noqa: E402
from prfield.metrics import response_metrics  # noqa: E402
from prfield.models import hybrid  # noqa: E402
from prfield.models.baselines import DifferenceFromNode, NodeModel  # noqa: E402
from prfield.report import write_report  # noqa: E402
from prfield.splits import observation as obs  # noqa: E402
from run_subgrid_robustness import block_designs  # noqa: E402

REPORT = Path("reports/AUDIT_NONLINEAR_BASELINE.md")
CSV = Path("reports/nonlinear_baseline.csv")
PROPS = ("heat_of_formation", "volume_per_fu")
POLICY = ComparabilityPolicy(max_volume_change_frac=None, max_rmsd_init_final=None,
                             require_same_coordination=False)


def features(df, desc, scal):
    """Standardised per-site descriptor blocks, concatenated. No explicit interactions."""
    out = []
    for s in ("A", "B", "X"):
        M = np.nan_to_num(np.asarray(
            desc[s].matrix(df[f"site_{s}"].tolist(), allow_missing=True), float))
        mu, sd = scal[s]
        out.append((M - mu) / sd)
    return np.hstack(out)


def scalers(df, desc):
    sc = {}
    for s in ("A", "B", "X"):
        M = np.nan_to_num(np.asarray(
            desc[s].matrix(df[f"site_{s}"].tolist(), allow_missing=True), float))
        mu, sd = M.mean(0), M.std(0)
        sc[s] = (mu, np.where(sd > 0, sd, 1.0))
    return sc


def main() -> None:
    df = pd.read_parquet("data/interim/castelli_raw.parquet")
    A = sorted(df.site_A.unique())[:14]
    B = sorted(df.site_B.unique())[:14]
    sub = df[df.site_A.isin(A) & df.site_B.isin(B)].reset_index(drop=True)
    desc = {s: el.physical_descriptors(sorted(sub[f"site_{s}"].unique().tolist()),
                                       "element" if s in ("A", "B") else "auto")
            for s in ("A", "B", "X")}

    rows, t0 = [], time.time()
    for prop in PROPS:
        edges = build_edges(sub, prop, POLICY)
        designs = [("block", n, d) for n, d in block_designs(sub, A, B)]
        designs += [("fiber-B", f"fiber B={l}", obs.held_out_fiber(sub, "B", l)) for l in B]
        designs += [("fiber-A", f"fiber A={l}", obs.held_out_fiber(sub, "A", l)) for l in A]
        for kind, name, d in designs:
            train = d.train_nodes(sub).dropna(subset=[prop]).reset_index(drop=True)
            te = d.induced_edges(edges).test
            if len(te) == 0 or len(train) == 0:
                continue
            y = te["delta"].to_numpy(dtype=float)
            idx = sub.set_index("sample_id")
            src, dst = idx.loc[te["src"]].reset_index(), idx.loc[te["dst"]].reset_index()

            def mae(pred):
                return response_metrics(y, np.asarray(pred, float)).mae

            sc = scalers(train, desc)
            Xtr = features(train, desc, sc)
            ytr = train[prop].to_numpy(dtype=float)
            Xs, Xd = features(src, desc, sc), features(dst, desc, sc)

            rf = RandomForestRegressor(n_estimators=400, random_state=0, n_jobs=-1).fit(Xtr, ytr)
            hgb = HistGradientBoostingRegressor(random_state=0).fit(Xtr, ytr)
            nm = NodeModel(desc, prop, interactions=True).fit(train)
            cn = hybrid.fit(train, prop, desc)

            rows.append({
                "property": prop, "kind": kind, "design": name,
                "desc_linear": mae(DifferenceFromNode(nm).predict_delta(te)),
                "desc_rf": mae(rf.predict(Xd) - rf.predict(Xs)),
                "desc_hgb": mae(hgb.predict(Xd) - hgb.predict(Xs)),
                "cat_free": mae(cn.predict(dst) - cn.predict(src))})
            print(f"  {prop[:12]:12s} {name:18s} {time.time()-t0:.0f}s", flush=True)

    d = pd.DataFrame(rows)
    CSV.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(CSV, index=False)

    from scipy import stats
    out = ["# A non-linear descriptor baseline\n"]
    out.append(
        "_Generated by `scripts/run_nonlinear_baseline.py`._ The subspace lemma covers linear "
        "descriptor designs only. A tree ensemble on the same descriptors is outside it and can "
        "represent interactions the second-order linear model cannot, so whether the gap "
        "survives is empirical.\n")
    out.append("| property / design | descriptor, linear | descriptor, RF "
               "| descriptor, HGB | free categorical |")
    out.append("|---|---|---|---|---|")
    for prop in PROPS:
        for kind in ("block", "fiber-A", "fiber-B"):
            g = d[(d["property"] == prop) & (d["kind"] == kind)]
            if g.empty:
                continue
            out.append(f"| {prop.replace('_',' ')} / {kind} | {g.desc_linear.median():.4f} "
                       f"| {g.desc_rf.median():.4f} | {g.desc_hgb.median():.4f} "
                       f"| {g.cat_free.median():.4f} |")
    out.append("\n## Free categorical against the best non-linear descriptor model\n")
    out.append("| property / design | median change | wins | Wilcoxon p |")
    out.append("|---|---|---|---|")
    for prop in PROPS:
        for kind in ("block", "fiber-A", "fiber-B"):
            g = d[(d["property"] == prop) & (d["kind"] == kind)]
            if g.empty:
                continue
            best = g[["desc_rf", "desc_hgb"]].min(axis=1)
            pct = 100 * (g.cat_free - best) / best
            p = float(stats.wilcoxon(g.cat_free - best)[1])
            out.append(f"| {prop.replace('_',' ')} / {kind} | {np.median(pct):+.1f}% | "
                       f"{int((g.cat_free < best).sum())}/{len(g)} | {p:.4f} |")
    write_report(REPORT, "\n".join(out))
    print(f"\nwrote {REPORT} and {CSV}")


if __name__ == "__main__":
    main()
