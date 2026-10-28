"""How the error of the descriptor map depends on the elements it is learned from.

On the 14 x 14 x 7 subgrid the descriptor map fits eleven parameters (an intercept and ten
descriptors) to the effects of thirteen elements and extrapolates to a fourteenth. The complete
grid has 52 elements per site, so the map can be learned from more elements than the subgrid holds.
Three changes are compared by leave-one-element-out error on the exact main effects:

  elements   how many other elements the map is learned from, 13 up to 51
  ridge      the penalty, fixed at 1e-2 or chosen by inner leave-one-out
  shrink     shrinking the prediction toward the mean effect by the inner estimate of the variance
             the map explains

The effects are exact, so the error measures the descriptor map alone.

Writes reports/AUDIT_TRANSFER_SAMPLE_SIZE.md and reports/transfer_sample_size.csv.

Usage:  PYTHONPATH=src python3 scripts/run_transfer_sample_size.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prfield.descriptors import elemental as el  # noqa: E402
from prfield.models.hybrid import _phi, _ridge  # noqa: E402
from prfield.report import write_report  # noqa: E402

REPORT = Path("reports/AUDIT_TRANSFER_SAMPLE_SIZE.md")
CSV = Path("reports/transfer_sample_size.csv")
PROP = "heat_of_formation"
LAM_FIXED = 1e-2
LAM_GRID = [10.0 ** k for k in range(-3, 4)]
N_LEVELS = [13, 20, 30, 40, 51]
SEEDS = range(12)


def exact_main_effects(df: pd.DataFrame, site: str) -> dict:
    """Exact main effect of each level of `site` on the complete grid."""
    g = df.dropna(subset=[PROP])
    mu = g[PROP].mean()
    # the grid is complete, so the site marginal is the plain mean over the other two
    return (g.groupby(f"site_{site}")[PROP].mean() - mu).to_dict()


def loo_error(y, P, i, keep, lam, shrink):
    """Error predicting level i from `keep`, with penalty lam and optional shrinkage."""
    P_tr, mu_, sd_ = _phi_rows(P, keep)
    c = _ridge(P_tr, y[keep], lam)
    P_te, _, _ = _phi_rows(P, [i], mu_, sd_)
    pred = float((P_te @ c)[0])
    if shrink:
        # how much of the variance does the map explain out of sample, inside the training
        # levels? use that as the shrinkage weight toward the training mean
        inner = []
        for j in keep:
            sub = [k for k in keep if k != j]
            Pi, m2, s2 = _phi_rows(P, sub)
            ci = _ridge(Pi, y[sub], lam)
            Pj, _, _ = _phi_rows(P, [j], m2, s2)
            inner.append(y[j] - float((Pj @ ci)[0]))
        var_y = float(np.var(y[keep]))
        r2 = 1.0 - float(np.mean(np.square(inner))) / var_y if var_y > 0 else 0.0
        w = max(0.0, min(1.0, r2))
        pred = w * pred + (1 - w) * float(np.mean(y[keep]))
    return abs(y[i] - pred)


def _phi_rows(P, idx, mean=None, std=None):
    M = P[list(idx)]
    if mean is None:
        mean, std = M.mean(0), np.where(M.std(0) > 0, M.std(0), 1.0)
    Z = (M - mean) / std
    return np.hstack([np.ones((len(idx), 1)), Z]), mean, std


def tune_lambda(y, P, keep):
    """Inner leave-one-out over the training levels only; never sees the held-out level."""
    best, best_err = LAM_FIXED, np.inf
    for lam in LAM_GRID:
        errs = [loo_error(y, P, j, [k for k in keep if k != j], lam, False) for j in keep]
        e = float(np.mean(np.square(errs)))
        if e < best_err:
            best, best_err = lam, e
    return best


def main() -> None:
    df = pd.read_parquet("data/interim/castelli_raw.parquet")
    rng = np.random.default_rng(0)
    rows = []
    for site in ("A", "B"):
        eff = exact_main_effects(df, site)
        labels = sorted(eff)
        desc = el.physical_descriptors(labels, "element")
        P = np.nan_to_num(np.asarray(desc.matrix(labels, allow_missing=True), dtype=float))
        y = np.array([eff[l] for l in labels])
        targets = list(range(min(14, len(labels))))  # the subgrid levels are the first 14
        for n_lev in N_LEVELS:
            if n_lev > len(labels) - 1:
                continue
            for seed in SEEDS:
                r = np.random.default_rng(1000 * seed + n_lev)
                for i in targets:
                    pool = [j for j in range(len(labels)) if j != i]
                    keep = list(r.choice(pool, n_lev, replace=False))
                    for mode in ("fixed", "tuned", "tuned+shrink"):
                        lam = LAM_FIXED if mode == "fixed" else tune_lambda(y, P, keep)
                        e = loo_error(y, P, i, keep, lam, mode.endswith("shrink"))
                        rows.append({"site": site, "n_levels": n_lev, "seed": seed,
                                     "level": labels[i], "mode": mode, "abs_err": e})
            print(f"  {site}: {n_lev} training levels done", flush=True)

    d = pd.DataFrame(rows)
    CSV.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(CSV, index=False)

    piv = d.groupby(["site", "mode", "n_levels"])["abs_err"].median().unstack()
    out = ["# What actually reduces unseen-element transfer error\n"]
    out.append(
        "_Generated by `scripts/run_transfer_sample_size.py`._ Transfer error is the quantity "
        "that predicts how hard a left-out element is, so this measures interventions against it directly. "
        "Median absolute error predicting a level's exact main effect from the descriptors of "
        "other levels, over 12 draws of the training set and the 14 evaluation levels.\n")
    out.append("| site | mode | " + " | ".join(f"{n} levels" for n in piv.columns) + " |")
    out.append("|---" * (len(piv.columns) + 2) + "|")
    for (site, mode), r in piv.iterrows():
        out.append(f"| {site} | {mode} | " + " | ".join(f"{v:.4f}" for v in r) + " |")

    base = piv.xs("fixed", level="mode")
    out.append("\n## Reading\n")
    for site in ("A", "B"):
        lo, hi = base.loc[site, min(piv.columns)], base.loc[site, max(piv.columns)]
        out.append(f"- Site {site}: going from {min(piv.columns)} to {max(piv.columns)} "
                   f"training levels moves the median error from {lo:.4f} to {hi:.4f}, "
                   f"a change of {100 * (hi / lo - 1):+.0f}%.")
    write_report(REPORT, "\n".join(out))
    print(piv.to_string())
    print(f"\nwrote {REPORT} and {CSV}")


if __name__ == "__main__":
    main()
