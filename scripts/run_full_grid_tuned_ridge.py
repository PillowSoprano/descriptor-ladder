"""Tuned-ridge control for the full-grid block comparison.

run_full_grid.py compares free element effects (lambda = 1e-3) with the same effects under the
descriptor similarity prior (strength 0.1). The prior adds shrinkage as well as chemistry, so
this asks whether a ridge penalty tuned by inner cross-validation reaches the prior on its own,
and whether a prior built from shuffled descriptors does.

Five arms on the same 15 block designs and the same test edges as run_full_grid.py:

  free            lambda = 1e-3, no prior                          (reproduces cat_node)
  ridge_tuned     lambda chosen from nine values by inner CV
  prior           lambda = 1e-3, prior strength 0.1                (reproduces cat_smooth)
  prior_tuned     lambda = 1e-3, prior strength chosen by inner CV
  shuffled_tuned  as prior_tuned, descriptor rows reassigned among elements

Inner CV: three block holdouts of the outer size inside each outer training set. The score is
the response MAE on edges that touch an inner-hidden composition and have both endpoints in the
outer training set, the same estimand as the outer test.

Writes reports/full_grid_tuned_ridge.csv and reports/AUDIT_FULL_GRID_TUNED_RIDGE.md.

Usage:  VECLIB_MAXIMUM_THREADS=2 nice -n 19 python3 scripts/run_full_grid_tuned_ridge.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prfield.descriptors import elemental as el  # noqa: E402
from prfield.graph.typed_graph import accepted, build_edges  # noqa: E402
from prfield.models import hybrid  # noqa: E402
from prfield.report import write_report  # noqa: E402
from prfield.schema import SITES  # noqa: E402
from prfield.splits import observation as obs  # noqa: E402
from run_full_grid import KS, MAX_TEST, POLICY, PROPS, SMOOTH  # noqa: E402
from run_regularization_baselines import (  # noqa: E402
    LAMBDA_GRID, descriptor_controls, metric_matrix)

REPORT = Path("reports/AUDIT_FULL_GRID_TUNED_RIDGE.md")
CSV = Path("reports/full_grid_tuned_ridge.csv")
BASE_LAMBDA = 1e-3
INNER_FOLDS = 3
INNER_SEED = 200_000
MAX_VALID = 30_000

# arm -> candidate (lambda, metric, strength) tuples; one candidate means no tuning
ARMS = {
    "free": [(BASE_LAMBDA, None, 0.0)],
    "ridge_tuned": [(lam, None, 0.0) for lam in LAMBDA_GRID],
    "prior": [(BASE_LAMBDA, "real", SMOOTH)],
    "prior_tuned": [(BASE_LAMBDA, "real", g) for g in LAMBDA_GRID],
    "shuffled_tuned": [(BASE_LAMBDA, "shuffled", g) for g in LAMBDA_GRID],
}


def solve(gram, rhs, lam, metric, strength):
    system = gram.copy()
    system[np.diag_indices_from(system)] += lam
    if metric is not None and strength:
        system += strength * metric
    try:
        factor = scipy.linalg.cho_factor(system, lower=True, overwrite_a=True, check_finite=False)
        return scipy.linalg.cho_solve(factor, rhs, check_finite=False)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(gram + lam * np.eye(len(gram))
                               + (strength * metric if metric is not None else 0.0),
                               rhs, rcond=None)[0]


def main() -> None:
    df = pd.read_parquet("data/interim/castelli_raw.parquet")
    A, B = sorted(df.site_A.unique()), sorted(df.site_B.unique())
    grid = df.reset_index(drop=True)
    levels = {s: sorted(grid[f"site_{s}"].unique().tolist()) for s in SITES}
    desc = {s: el.physical_descriptors(sorted(grid[f"site_{s}"].unique().tolist()),
                                       "element" if s in ("A", "B") else "auto")
            for s in SITES}
    shuffled, _ = descriptor_controls(desc)

    X_all, slices = hybrid._design_and_slices(grid, levels)
    X_all = X_all.tocsr()
    n_cols = X_all.shape[1]
    CtC = hybrid._constraints(levels, slices, n_cols)
    CtC = (CtC.T @ CtC).toarray()
    metrics = {"real": metric_matrix(desc, levels, slices, n_cols),
               "shuffled": metric_matrix(shuffled, levels, slices, n_cols)}
    pos = pd.Series(np.arange(len(grid)), index=grid["sample_id"])
    print(f"{len(grid)} nodes, {n_cols} columns", flush=True)

    # identical design draws to run_full_grid.py
    rng = np.random.default_rng(0)
    designs = []
    for k in KS:
        for rep in range(5):
            aa = list(rng.choice(A, size=k, replace=False))
            bb = list(rng.choice(B, size=k, replace=False))
            designs.append((f"block {k}A x {k}B #{rep}", k,
                            obs.held_out_block(grid, {"A": aa, "B": bb})))

    def system(ids, y_all):
        p = pos.loc[list(ids)].to_numpy()
        p = p[~np.isnan(y_all[p])]
        X = X_all[p]
        w2 = 100.0 * len(p)
        return (X.T @ X).toarray() + w2 * CtC, np.asarray(X.T @ y_all[p]).ravel(), len(p)

    def delta(edges):
        s = pos.loc[edges["src"]].to_numpy()
        d = pos.loc[edges["dst"]].to_numpy()
        return (X_all[d] - X_all[s]).tocsr(), edges["delta"].to_numpy(float)

    rows, t0 = [], time.time()
    for prop in PROPS:
        y_all = grid[prop].to_numpy(float)
        edges = accepted(build_edges(grid, prop, POLICY)).reset_index(drop=True)
        print(f"\n== {prop}: {len(edges):,} accepted edges ==", flush=True)
        for di, (name, k, d) in enumerate(designs):
            t = time.time()
            train_ids = d.observed_ids
            te = d.induced_edges(edges).test
            if len(te) > MAX_TEST:
                te = te.iloc[np.random.default_rng(0).choice(len(te), MAX_TEST, replace=False)]
            dX_te, y_te = delta(te)

            # inner folds: same-size blocks inside the outer training set
            in_train = edges["src"].isin(train_ids) & edges["dst"].isin(train_ids)
            losses = {key: [] for arm, c in ARMS.items() if len(c) > 1 for key in c}
            for f in range(INNER_FOLDS):
                r = np.random.default_rng(INNER_SEED + 100 * di + f)
                ia = set(r.choice(A, size=k, replace=False))
                ib = set(r.choice(B, size=k, replace=False))
                sub = grid[grid.sample_id.isin(train_ids)]
                hidden = set(sub.loc[sub.site_A.isin(ia) & sub.site_B.isin(ib), "sample_id"])
                val = edges[in_train & (edges["src"].isin(hidden) | edges["dst"].isin(hidden))]
                if len(val) > MAX_VALID:
                    val = val.iloc[r.choice(len(val), MAX_VALID, replace=False)]
                gram, rhs, _ = system(train_ids - hidden, y_all)
                dX, yv = delta(val)
                for key in losses:
                    lam, m, g = key
                    beta = solve(gram, rhs, lam, metrics.get(m), g)
                    losses[key].append(float(np.mean(np.abs(yv - dX @ beta))))

            gram, rhs, n_train = system(train_ids, y_all)
            rec = {"property": prop, "design": name, "k": k,
                   "n_train": n_train, "n_test": len(te)}
            for arm, cands in ARMS.items():
                key = cands[0] if len(cands) == 1 else min(
                    cands, key=lambda c: (np.mean(losses[c]), cands.index(c)))
                lam, m, g = key
                beta = solve(gram, rhs, lam, metrics.get(m), g)
                rec[arm] = float(np.mean(np.abs(y_te - dX_te @ beta)))
                rec[f"{arm}_lambda"], rec[f"{arm}_strength"] = lam, g
            rec["seconds"] = time.time() - t
            rows.append(rec)
            print(f"  {name:20s} {rec['seconds']:4.0f}s  free {rec['free']:.4f}  "
                  f"ridge {rec['ridge_tuned']:.4f} (l={rec['ridge_tuned_lambda']:g})  "
                  f"prior {rec['prior']:.4f}  tuned {rec['prior_tuned']:.4f} "
                  f"(g={rec['prior_tuned_strength']:g})  shuffled {rec['shuffled_tuned']:.4f} "
                  f"(g={rec['shuffled_tuned_strength']:g})  total {(time.time()-t0)/60:.0f} min",
                  flush=True)
            pd.DataFrame(rows).to_csv(CSV, index=False)

    res = pd.DataFrame(rows)
    out = ["# Tuned-ridge control on the complete grid\n"]
    out.append(
        "_Generated by `scripts/run_full_grid_tuned_ridge.py`._ The 15 block designs and test "
        "edges of `run_full_grid.py`. Tuned arms choose their penalty from nine values "
        "($10^{-6}$ to $10^{2}$) by three inner block holdouts of the outer size. Median "
        "response MAE across designs.\n")
    out.append("| property | free | tuned ridge | prior 0.1 | prior tuned | shuffled prior tuned |")
    out.append("|---|---|---|---|---|---|")
    for prop in PROPS:
        g = res[res.property == prop]
        out.append(f"| {prop.replace('_', ' ')} | " + " | ".join(
            f"{g[a].median():.4f}" for a in ARMS) + " |")
    out.append("\n## Paired contrasts (median of per-design change, Wilcoxon p)\n")
    pairs = [("ridge_tuned", "free"), ("prior", "ridge_tuned"), ("prior_tuned", "ridge_tuned"),
             ("shuffled_tuned", "ridge_tuned"), ("prior_tuned", "shuffled_tuned")]
    out.append("| property | " + " | ".join(f"{a} vs {b}" for a, b in pairs) + " |")
    out.append("|---|" + "---|" * len(pairs))
    for prop in PROPS:
        g = res[res.property == prop]
        cells = []
        for a, b in pairs:
            v = 100 * (g[a].to_numpy() / g[b].to_numpy() - 1)
            p = float(stats.wilcoxon(g[a] - g[b])[1])
            cells.append(f"{np.median(v):+.1f}% (p={p:.4f}, {int((v < 0).sum())}/{len(v)})")
        out.append(f"| {prop.replace('_', ' ')} | " + " | ".join(cells) + " |")
    out.append("\n## Selected penalties (count over designs)\n")
    for prop in PROPS:
        g = res[res.property == prop]
        for arm, col in (("ridge_tuned", "ridge_tuned_lambda"),
                         ("prior_tuned", "prior_tuned_strength"),
                         ("shuffled_tuned", "shuffled_tuned_strength")):
            counts = g[col].value_counts().sort_index()
            out.append(f"- {prop.replace('_', ' ')}, {arm}: "
                       + ", ".join(f"{v:g} x{c}" for v, c in counts.items()))
    out.append(f"\nRuntime: {(time.time() - t0) / 60:.0f} minutes.")
    write_report(REPORT, "\n".join(out))
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
