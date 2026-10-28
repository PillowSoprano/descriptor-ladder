"""Does the interpolation peak grow with the share of interaction parameters?

Rung 2 of the descriptor manuscript says the error peak near n = p appears only in models whose
parameters are mostly interactions. This refits untuned free element effects on the alphabetical
14 x 14 x 7 subgrid with different sets of pair-interaction blocks, places each model's coverage
sweep on its own n/p axis, and measures the peak as the largest rise of the untuned error as data
are added: max over coverages of error / (lowest error at any smaller coverage) - 1. The excess
over a tuned ridge at the same coverage is reported alongside. Seven models: additive, AX, BX, AB, AX+BX, AB+AX and all pairs.

Responses are scored when both compositions are hidden, as in rung 2. Ridge weights are chosen from
the same nine values by three inner random holdouts on training compositions, as in
run_regularization_baselines.py.

Writes reports/peak_vs_interaction_share.csv and reports/AUDIT_PEAK_VS_INTERACTION_SHARE.md.

Usage:  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 scripts/run_peak_vs_interaction_share.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_regularization_baselines as mh  # noqa: E402
from prfield.graph.typed_graph import accepted, build_edges  # noqa: E402
from prfield.models import hybrid  # noqa: E402
from prfield.report import write_report  # noqa: E402
from prfield.schema import SITES  # noqa: E402
from prfield.splits import observation as obs  # noqa: E402

REPORT = Path("reports/AUDIT_PEAK_VS_INTERACTION_SHARE.md")
CSV = Path("reports/peak_vs_interaction_share.csv")
PROPS = ("heat_of_formation", "volume_per_fu")
MODELS = {"additive": (), "AX": ("AX",), "BX": ("BX",), "AB": ("AB",),
          "AX+BX": ("AX", "BX"), "AB+AX": ("AB", "AX"), "all pairs": ("AB", "AX", "BX")}
NP_TARGETS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0, 1.15, 1.3, 1.6, 2.0, 3.0)
SEEDS = tuple(range(5))


def keep_mask(slices, n_cols, pairs):
    keep = np.zeros(n_cols, bool)
    keep[0] = True
    for s in SITES:
        keep[slices[s]] = True
    for p in pairs:
        keep[slices[p]] = True
    return keep


def identified(levels, pairs):
    size = {s: len(levels[s]) - 1 for s in SITES}
    mains = 1 + sum(size.values())
    inter = sum(size[p[0]] * size[p[1]] for p in pairs)
    return mains + inter, inter


def fit(gram, rhs, keep, lam, n_cols, main_cols):
    """Ridge on the kept columns; lam applies to every effect, not the intercept."""
    g, r = gram[np.ix_(keep, keep)], rhs[keep]
    diag = np.zeros(keep.sum())
    diag[1:] = lam
    beta = np.zeros(n_cols)
    beta[keep] = mh.solve(g, r, diag)
    return beta


def evaluate(sub, prop, model, frac, seed, levels, edges, node_index):
    pairs = MODELS[model]
    split = obs.random_nodes(sub, frac, seed=seed)
    train = split.train_nodes(sub).dropna(subset=[prop]).reset_index(drop=True)
    test = mh.edge_subset(edges, split.held_out_ids, both_hidden=True)
    gram, rhs, slices = mh.normal_equations(train, prop, levels)
    n_cols = gram.shape[0]
    keep = keep_mask(slices, n_cols, pairs)

    # inner selection of the ridge weight, mirroring the coverage sweep
    losses = np.zeros(len(mh.LAMBDA_GRID))
    folds = 0
    train_ids = set(train.sample_id)
    for fold in range(mh.INNER_FOLDS):
        inner = obs.random_nodes(train, mh.INNER_TRAIN_FRACTION, seed=100_000 + 100 * seed + fold)
        val = mh.edge_subset(edges, inner.held_out_ids, both_hidden=True)
        val = val[val["src"].isin(train_ids) & val["dst"].isin(train_ids)]
        if val.empty:
            continue
        g_in, r_in, _ = mh.normal_equations(inner.train_nodes(train), prop, levels)
        for i, lam in enumerate(mh.LAMBDA_GRID):
            losses[i] += mh.edge_mae(fit(g_in, r_in, keep, lam, n_cols, None), levels, node_index, val)
        folds += 1
    lam = mh.LAMBDA_GRID[int(np.argmin(losses))] if folds else 1.0

    untuned = mh.edge_mae(fit(gram, rhs, keep, 0.0, n_cols, None), levels, node_index, test)
    ridge = mh.edge_mae(fit(gram, rhs, keep, lam, n_cols, None), levels, node_index, test)
    return dict(property=prop, model=model, coverage=frac, seed=seed, n_train=len(train),
                n_test=len(test), untuned=untuned, ridge=ridge, ridge_lambda=lam)


def main() -> None:
    frame = pd.read_parquet("data/interim/castelli_raw.parquet")
    _, a_levels, b_levels = mh.panel_definitions(frame)[0]
    sub = frame[frame.site_A.isin(a_levels) & frame.site_B.isin(b_levels)].reset_index(drop=True)
    levels = {s: sorted(sub[f"site_{s}"].unique().tolist()) for s in SITES}
    node_index = sub.set_index("sample_id")
    edges = {p: accepted(build_edges(sub, p, mh.POLICY)).reset_index(drop=True) for p in PROPS}

    tasks = []
    for model, pairs in MODELS.items():
        p, _ = identified(levels, pairs)
        for target in NP_TARGETS:
            frac = min(0.95, target * p / len(sub))
            tasks += [(prop, model, frac, seed) for prop in PROPS for seed in SEEDS]
    t0 = time.time()
    rows = Parallel(n_jobs=6, verbose=5)(
        delayed(evaluate)(sub, prop, model, frac, seed, levels, edges[prop], node_index)
        for prop, model, frac, seed in tasks)
    res = pd.DataFrame(rows)
    for model, pairs in MODELS.items():
        p, inter = identified(levels, pairs)
        m = res.model == model
        res.loc[m, "p"] = p
        res.loc[m, "interaction_share"] = inter / p
    res["n_over_p"] = res.n_train / res.p
    CSV.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(CSV, index=False)

    curve = (res.groupby(["property", "model", "coverage"])
             .agg(n_over_p=("n_over_p", "median"), untuned=("untuned", "median"),
                  ridge=("ridge", "median"), share=("interaction_share", "first"), p=("p", "first"))
             .reset_index())
    curve["excess_pct"] = 100 * (curve.untuned / curve.ridge - 1)
    curve = curve.sort_values(["property", "model", "coverage"])
    rows_peak = []
    for (prop, model), g in curve.groupby(["property", "model"]):
        u = g.untuned.to_numpy()
        prior_min = np.minimum.accumulate(np.r_[np.inf, u[:-1]])
        rise = 100 * (u / prior_min - 1)
        k = int(np.nanargmax(rise))
        rows_peak.append(dict(property=prop, model=model, share=g.share.iloc[0], p=g.p.iloc[0],
                              rise_pct=max(rise[k], 0.0), rise_at=g.n_over_p.iloc[k],
                              excess_pct=g.excess_pct.max(),
                              excess_at=g.n_over_p.iloc[int(g.excess_pct.argmax())]))
    peak = pd.DataFrame(rows_peak).sort_values(["property", "share"])

    out = ["# Interpolation peak against the share of interaction parameters\n"]
    out.append(
        "_Generated by `scripts/run_peak_vs_interaction_share.py`._ Alphabetical 14x14x7 subgrid; "
        "untuned free effects (numerical ridge only) against a ridge tuned by three inner holdouts; "
        "responses with both compositions hidden; five seeds per coverage, coverages placed at "
        "n/p = " + ", ".join(f"{t:g}" for t in NP_TARGETS) + " for each model. The peak is the "
        "largest excess error of the untuned model over the tuned ridge, medians over seeds.\n")
    out.append("| property | model | identified parameters | interaction share | rise of error | at n/p "
               "| largest excess over tuned ridge | at n/p |")
    out.append("|---|---|---|---|---|---|---|---|")
    for _, r in peak.iterrows():
        out.append(f"| {r.property.replace('_', ' ')} | {r.model} | {int(r.p)} | {r.share:.2f} "
                   f"| {r.rise_pct:+.1f}% | {r.rise_at:.2f} | {r.excess_pct:+.1f}% | {r.excess_at:.2f} |")
    for prop in PROPS:
        g = peak[peak.property == prop]
        for col in ("rise_pct", "excess_pct"):
            rho, pval = stats.spearmanr(g.share, g[col])
            out.append(f"\n{prop.replace('_', ' ')}: Spearman rho(share, {col}) = {rho:.2f} "
                       f"(p = {pval:.3f}, {len(g)} models).")
    peak.to_csv(CSV.with_name("peak_vs_interaction_share_summary.csv"), index=False)
    out.append("\n## Excess error by n/p\n")
    piv = curve.pivot_table(index=["property", "model"], columns=curve.n_over_p.round(2),
                            values="excess_pct")
    out.append(piv.round(1).to_string())
    out.append(f"\nRuntime: {(time.time() - t0) / 60:.1f} minutes.")
    write_report(REPORT, "\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
