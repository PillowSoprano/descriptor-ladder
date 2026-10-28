"""Where in parameter space does the similarity prior earn its gain over tuned ridge?

The coverage sweep compares a ridge-tuned free-effect model with the descriptor similarity prior.
Their coefficient difference is split along the eigenvectors of the data information matrix X'X
restricted to the sum-to-zero parameter space, and each eigenvector is labelled by how much the
computed compositions determine it:

  undetermined   eigenvalue 0 (the data say nothing about this direction)
  weak           0 < eigenvalue < 1 (less information than one composition gives one parameter)
  determined     eigenvalue >= 1

The reduction in squared error of the held-out responses splits exactly over these classes:
MSE_ridge - MSE_prior = (2/n) sum_c <delta_c, y - (p_ridge + p_prior)/2>, where delta_c is the
change in predicted response carried by class c. Penalties are the ones the sweep selected by
inner cross-validation (reports/regularization_panel_curves.csv); nothing is refitted or tuned.
Thresholds 0.5 and 2 are reported as a sensitivity check.

With --shuffled the prior is built from descriptors reassigned among elements, on the alphabetical
subgrid, with the weights the sweep selected for that control (reports/regularization_baselines.csv).

Writes reports/gain_by_direction_state.csv and reports/AUDIT_GAIN_BY_DIRECTION_STATE.md
(with --shuffled: the same names with _shuffled).

Usage:  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 scripts/run_gain_by_direction_state.py [--shuffled]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_regularization_baselines as mh  # noqa: E402
from prfield.descriptors import elemental as el  # noqa: E402
from prfield.graph.typed_graph import accepted, build_edges  # noqa: E402
from prfield.models import hybrid  # noqa: E402
from prfield.report import write_report  # noqa: E402
from prfield.schema import SITES  # noqa: E402
from prfield.splits import observation as obs  # noqa: E402

REPORT = Path("reports/AUDIT_GAIN_BY_DIRECTION_STATE.md")
CSV = Path("reports/gain_by_direction_state.csv")
CURVES = Path("reports/regularization_panel_curves.csv")
THRESHOLDS = (0.5, 1.0, 2.0)
CLASSES = ("undetermined", "weak", "determined")


def lam(label: str) -> float:
    return float(label.split("=")[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffled", action="store_true")
    args = ap.parse_args()
    global REPORT, CSV
    prior_model = "shuffled_metric" if args.shuffled else "descriptor_metric"
    if args.shuffled:
        REPORT = REPORT.with_name(REPORT.stem + "_SHUFFLED.md")
        CSV = CSV.with_name(CSV.stem + "_shuffled.csv")
    t0 = time.time()
    frame = pd.read_parquet("data/interim/castelli_raw.parquet")
    if "magnetic_moment" not in frame or frame["magnetic_moment"].isna().all():
        source = "_magmom_signed" if "_magmom_signed" in frame else "_magmom"
        frame["magnetic_moment"] = frame[source].abs()
    curves = pd.read_csv(Path("reports/regularization_baselines.csv") if args.shuffled else CURVES)
    if args.shuffled:
        curves = curves.assign(panel="alphabetical")
        curves = curves[curves.seed.isin([0, 1, 2])]
    rows = []
    panels = mh.panel_definitions(frame)[:1] if args.shuffled else mh.panel_definitions(frame)
    for panel, a_levels, b_levels in panels:
        sub = frame[frame.site_A.isin(a_levels) & frame.site_B.isin(b_levels)].reset_index(drop=True)
        levels = {s: sorted(sub[f"site_{s}"].unique().tolist()) for s in SITES}
        desc = {s: el.physical_descriptors(levels[s], "element" if s in ("A", "B") else "auto")
                for s in SITES}
        X_all, slices = hybrid._design_and_slices(sub, levels)
        X_all = X_all.tocsr()
        n_cols = X_all.shape[1]
        if args.shuffled:
            desc = mh.descriptor_controls(desc)[0]
        metric = mh.metric_matrix(desc, levels, slices, n_cols)
        C = hybrid._constraints(levels, slices, n_cols).toarray()
        Q = scipy.linalg.null_space(C)                       # sum-to-zero parameter space
        pos = pd.Series(np.arange(len(sub)), index=sub["sample_id"])
        for prop in mh.PROPERTIES:
            edges = accepted(build_edges(sub, prop, mh.POLICY)).reset_index(drop=True)
            cur = curves[(curves.panel == panel) & (curves.property == prop)]
            for (frac, seed), g in cur.groupby(["coverage", "seed"]):
                sel = g.set_index("model").selected
                lam_r, lam_m = lam(sel["ordinary_ridge"]), lam(sel[prior_model])
                split = obs.random_nodes(sub, frac, seed=int(seed))
                train = split.train_nodes(sub).dropna(subset=[prop]).reset_index(drop=True)
                test = mh.edge_subset(edges, split.held_out_ids, both_hidden=True)
                gram, rhs, _ = mh.normal_equations(train, prop, levels)
                b0 = mh.solve(gram, rhs, mh.diagonal_penalty(slices, n_cols, lam_r, lam_r))
                b1 = mh.solve(gram, rhs, np.zeros(n_cols), metric=metric, metric_strength=lam_m)
                s = pos.loc[test["src"]].to_numpy()
                d = pos.loc[test["dst"]].to_numpy()
                E = (X_all[d] - X_all[s]).tocsr()
                y = test["delta"].to_numpy(float)
                p0, p1 = E @ b0, E @ b1
                ref = g.set_index("model").mae_both_hidden
                check = max(abs(np.mean(np.abs(y - p0)) - ref["ordinary_ridge"]),
                            abs(np.mean(np.abs(y - p1)) - ref[prior_model]))
                Xt = X_all[pos.loc[train["sample_id"]].to_numpy()]
                info = Q.T @ (Xt.T @ Xt).toarray() @ Q
                ev, U = np.linalg.eigh((info + info.T) / 2)
                ev = np.clip(ev, 0, None)
                coords = U.T @ (Q.T @ (b1 - b0))
                outside = (b1 - b0) - Q @ (Q.T @ (b1 - b0))
                mid = y - (p0 + p1) / 2
                total = float(np.mean((y - p0) ** 2) - np.mean((y - p1) ** 2))
                zero = ev <= ev.max() * 1e-10
                for thr in THRESHOLDS:
                    masks = {"undetermined": zero, "weak": ~zero & (ev < thr), "determined": ev >= thr}
                    rec = {"panel": panel, "property": prop, "coverage": frac, "seed": seed,
                           "threshold": thr, "n_train": len(train), "lam_ridge": lam_r,
                           "lam_prior": lam_m, "mse_gain": total, "repro_mae_error": check}
                    parts = 0.0
                    for c in CLASSES:
                        m = masks[c]
                        delta = E @ (Q @ (U[:, m] @ coords[m]))
                        rec[f"gain_{c}"] = float(2 * np.mean(delta * mid))
                        rec[f"dims_{c}"] = int(m.sum())
                        parts += rec[f"gain_{c}"]
                    rec["gain_outside"] = float(2 * np.mean((E @ outside) * mid))
                    rec["closure_error"] = total - parts - rec["gain_outside"]
                    rows.append(rec)
        print(f"panel {panel} done, {(time.time() - t0) / 60:.1f} min", flush=True)

    res = pd.DataFrame(rows)
    CSV.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(CSV, index=False)

    out = ["# Where the prior's gain over tuned ridge sits\n"]
    out.append(
        "_Generated by `scripts/run_gain_by_direction_state.py`._ Seven subgrids, three seeds, the "
        "penalties the coverage sweep selected. Each share is a class's term summed over the 21 "
        "designs at one coverage, divided by the summed total reduction in squared error of the "
        "both-hidden responses. Dimensions are the mean fraction of the identified parameter space "
        f"in each class. Largest reproduction error of the stored MAE: {res.repro_mae_error.max():.2e}; "
        f"largest closure error of the split: {res.closure_error.abs().max():.2e}; largest share "
        f"outside the sum-to-zero space: "
        f"{(res.gain_outside.abs() / res.mse_gain.abs().clip(lower=1e-30)).max():.2e}.\n")
    for thr in THRESHOLDS:
        out.append(f"\n## Threshold {thr:g}\n")
        out.append("| property | coverage | total gain | undetermined | weak | determined "
                   "| dims undetermined | dims weak | dims determined |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        r = res[res.threshold == thr]
        for (prop, frac), g in r.groupby(["property", "coverage"]):
            tot = g.mse_gain.sum()
            dims = sum(g[f"dims_{c}"].mean() for c in CLASSES)
            cells = " | ".join(f"{100 * g[f'gain_{c}'].sum() / tot:+.0f}%" for c in CLASSES)
            dcells = " | ".join(f"{100 * g[f'dims_{c}'].mean() / dims:.0f}%" for c in CLASSES)
            out.append(f"| {prop.replace('_', ' ')} | {frac:.0%} | {tot:+.4g} | {cells} | {dcells} |")
    out.append(f"\nRuntime: {(time.time() - t0) / 60:.1f} minutes.")
    write_report(REPORT, "\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
