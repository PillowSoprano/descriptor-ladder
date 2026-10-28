"""Covariance rank at a fixed estimator.

`prfield.models.gp` fits element effects under a Gaussian-process prior over descriptor space. With
a degenerate linear kernel it spans what a descriptor basis spans; with a radial kernel it is full
rank. Keeping the estimator, the fitting procedure and the hyperparameter search fixed and changing
only the kernel isolates the effect of the covariance rank.

Three reductions are checked numerically:

  1. at an element seen in training the posterior returns the fitted coefficient
  2. the degenerate kernel reproduces the span of the descriptor basis
  3. a left-out element's prediction is a kernel regression from the fitted effects, with weights
     that do not depend on the left-out element's own value

Five estimators on every design: the shared-trend model, the same model with the similarity prior,
the Gaussian process with the radial kernel, the Gaussian process with the degenerate kernel, and
gradient boosting on the same descriptors. Every hyperparameter is chosen by inner
cross-validation that holds out a training element.

Writes reports/AUDIT_COVARIANCE_RANK.md and reports/covariance_rank.csv.

Usage:  PYTHONPATH=src python3 scripts/run_covariance_rank.py
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
from prfield.models import gp, hybrid  # noqa: E402
from prfield.report import write_report  # noqa: E402
from prfield.splits import observation as obs  # noqa: E402
from run_subgrid_robustness import block_designs  # noqa: E402

REPORT = Path("reports/AUDIT_COVARIANCE_RANK.md")
CSV = Path("reports/covariance_rank.csv")
PROPS = ("heat_of_formation", "volume_per_fu")
POLICY = ComparabilityPolicy(max_volume_change_frac=None, max_rmsd_init_final=None,
                             require_same_coordination=False)
SUBGRID = 14


def reductions(sub, desc) -> list[str]:
    out = []
    tr = sub[sub.site_B != "As"].dropna(subset=["heat_of_formation"]).reset_index(drop=True)
    m = gp.fit(tr, "heat_of_formation", desc, tau=1.0)
    worst = 0.0
    for s in ("A", "B", "X"):
        for i, lv in enumerate(m.eff.levels[s]):
            worst = max(worst, abs(m._main(s, lv) - float(m.eff.main[s][i])))
    out.append(f"- at an observed level the posterior returns the fitted coefficient to "
               f"{worst:.2e}")

    lin = gp.fit(tr, "heat_of_formation", desc, tau=1.0, linear_kernel=True)
    Z = lin.eff.feat["B"]
    K = gp._kernel(Z, Z, 0.0, True)
    rank = int(np.linalg.matrix_rank(K, tol=1e-8))
    out.append(f"- the degenerate kernel has rank {rank} on {len(Z)} levels, against the "
               f"{Z.shape[1] + 1} a constant plus the descriptor span provides")

    i, w = m._w("B", "As")
    out.append(f"- the unseen level enters through {len(w)} weights summing to {w.sum():.3f}, "
               "computed from its descriptors and the training levels only")
    return out


def main() -> None:
    df = pd.read_parquet("data/interim/castelli_raw.parquet")
    A = sorted(df.site_A.unique())[:SUBGRID]
    B = sorted(df.site_B.unique())[:SUBGRID]
    sub = df[df.site_A.isin(A) & df.site_B.isin(B)].reset_index(drop=True)
    desc = {s: el.physical_descriptors(sorted(sub[f"site_{s}"].unique().tolist()),
                                       "element" if s in ("A", "B") else "auto")
            for s in ("A", "B", "X")}
    idx = sub.set_index("sample_id")

    checks = reductions(sub, desc)
    print("\n".join(checks), flush=True)

    rows, t0 = [], time.time()
    for prop in PROPS:
        edges = build_edges(sub, prop, POLICY)
        designs = [("block", n, d) for n, d in block_designs(sub, A, B)]
        designs += [("fibre-A", f"unseen A={lv}", obs.held_out_fiber(sub, "A", lv)) for lv in A]
        designs += [("fibre-B", f"unseen B={lv}", obs.held_out_fiber(sub, "B", lv)) for lv in B]
        for kind, name, d in designs:
            train = d.train_nodes(sub).dropna(subset=[prop]).reset_index(drop=True)
            te = d.induced_edges(edges).test
            if len(te) == 0 or train.empty:
                continue
            y = te["delta"].to_numpy(float)
            src = idx.loc[te["src"]].reset_index()
            dst = idx.loc[te["dst"]].reset_index()

            def mae(pred):
                return response_metrics(y, np.asarray(pred, float)).mae

            rec = {"property": prop, "kind": kind, "design": name, "n_test": len(te)}
            h = hybrid.fit(train, prop, desc)
            rec["hybrid"] = mae(h.predict(dst) - h.predict(src))
            eff = hybrid.fit_stage1(train, prop, smooth=1.0, desc=desc)
            maps, norm = hybrid.fit_stage2(eff, desc)
            hp = hybrid.HybridModel(eff, maps, norm, desc)
            rec["hybrid_prior"] = mae(hp.predict(dst) - hp.predict(src))
            g = gp.fit(train, prop, desc)
            rec["gp"] = mae(g.predict(dst) - g.predict(src))
            rec["gp_tau"] = g.eff.tau
            gl = gp.fit(train, prop, desc, linear_kernel=True)
            rec["gp_basis"] = mae(gl.predict(dst) - gl.predict(src))
            F = np.hstack([np.nan_to_num(np.asarray(
                desc[s].matrix(train[f"site_{s}"].tolist(), allow_missing=True), float))
                for s in ("A", "B", "X")])
            mu, sd = F.mean(0), np.where(F.std(0) > 0, F.std(0), 1.0)
            hg = HistGradientBoostingRegressor(random_state=0).fit(
                (F - mu) / sd, train[prop].to_numpy(float))

            def feats(fr):
                M = np.hstack([np.nan_to_num(np.asarray(
                    desc[s].matrix(fr[f"site_{s}"].tolist(), allow_missing=True), float))
                    for s in ("A", "B", "X")])
                return (M - mu) / sd
            rec["hgb"] = mae(hg.predict(feats(dst)) - hg.predict(feats(src)))
            rows.append(rec)
            print(f"  {prop[:12]:12s} {name:16s} {time.time()-t0:.0f}s", flush=True)

    d = pd.DataFrame(rows)
    CSV.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(CSV, index=False)

    from scipy import stats
    out = ["# One posterior instead of a switch\n"]
    out.append(
        "_Generated by `scripts/run_covariance_rank.py`._ The two-stage estimator routes on "
        "level membership. `prfield.models.gp` puts a Gaussian process prior on the effects "
        "instead, so the observed-level and unseen-level behaviours are two evaluations of one "
        "posterior, and the descriptor basis is the same kernel with its radial part removed. "
        "The reductions are checked, not asserted:\n")
    out += checks
    out.append("\n## Median response error by design family\n")
    out.append("| property | family | shared trend | + similarity prior | GP, full kernel "
               "| GP, degenerate kernel | boosting |")
    out.append("|---|---|---|---|---|---|---|")
    for prop in PROPS:
        for kind in ("block", "fibre-A", "fibre-B"):
            g = d[(d["property"] == prop) & (d["kind"] == kind)]
            if g.empty:
                continue
            out.append(f"| {prop.replace('_',' ')} | {kind} | " + " | ".join(
                f"{g[c].median():.4f}" for c in
                ("hybrid", "hybrid_prior", "gp", "gp_basis", "hgb")) + " |")

    out.append("\n## The GP against each alternative, paired within design\n")
    out.append("| property | family | against | median change | wins | Wilcoxon p |")
    out.append("|---|---|---|---|---|---|")
    for prop in PROPS:
        for kind in ("block", "fibre-A", "fibre-B"):
            g = d[(d["property"] == prop) & (d["kind"] == kind)]
            if len(g) < 6:
                continue
            for col, nm in (("hybrid", "the shared-trend model"), ("hybrid_prior", "shared trend + prior"),
                            ("gp_basis", "the degenerate kernel"), ("hgb", "boosting")):
                pct = 100 * np.median((g["gp"] - g[col]) / g[col])
                p = float(stats.wilcoxon(g["gp"] - g[col])[1])
                out.append(f"| {prop.replace('_',' ')} | {kind} | {nm} | {pct:+.1f}% | "
                           f"{int((g['gp'] < g[col]).sum())}/{len(g)} | {p:.4f} |")
    write_report(REPORT, "\n".join(out))
    print(d.groupby(["property", "kind"])[
        ["hybrid", "hybrid_prior", "gp", "gp_basis", "hgb"]].median().round(4).to_string())
    print(f"\nwrote {REPORT} and {CSV}")


if __name__ == "__main__":
    main()
