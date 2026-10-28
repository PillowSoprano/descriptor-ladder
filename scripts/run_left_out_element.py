"""Responses of a left-out element with a wider and a non-linear descriptor map.

Two changes to the descriptor map act on different things and are combined here:

  wider        the map is learned from all 51 other elements of the site rather than the 13 of
               the subgrid
  non-linear   a radial kernel plus the degenerate term `1 + x'y` of the descriptor basis; the
               radial part localises and the degenerate part extrapolates to elements whose
               effect lies outside the range of the others

Bandwidths are chosen by inner leave-one-element-out over the training elements only.

Four arms and a reference, on the same designs and test pairs:

  narrow-linear   the shared-trend model on the subgrid
  wide-linear     wider descriptor map, linear
  narrow-kernel   subgrid descriptor map, combined kernel
  wide-kernel     both
  hgb             gradient boosting on the same descriptors

Only the left-out site is widened, because a main effect is an average over the other two sites;
`hybrid.fit` checks that the left-out element is absent from the added compositions.

Writes reports/AUDIT_LEFT_OUT_ELEMENT.md and reports/left_out_element.csv.

Usage:  PYTHONPATH=src python3 scripts/run_left_out_element.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prfield.descriptors import elemental as el  # noqa: E402
from prfield.graph.typed_graph import ComparabilityPolicy, build_edges  # noqa: E402
from prfield.metrics import response_metrics  # noqa: E402
from prfield.models import hybrid  # noqa: E402
from prfield.report import write_report  # noqa: E402
from prfield.splits import observation as obs  # noqa: E402

REPORT = Path("reports/AUDIT_LEFT_OUT_ELEMENT.md")
CSV = Path("reports/left_out_element.csv")
PROPS = ("heat_of_formation", "volume_per_fu")
POLICY = ComparabilityPolicy(max_volume_change_frac=None, max_rmsd_init_final=None,
                             require_same_coordination=False)
SUBGRID = 14
GAMMAS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
LAM2 = 1e-2


def pick_gamma(desc_site, levels, y) -> float:
    """Inner leave-one-level-out over the training elements of the descriptor map. Never sees the held out."""
    P, _, _ = hybrid._phi(desc_site, list(levels))
    Z = P[:, 1:]
    best, best_e = GAMMAS[0], np.inf
    for g in GAMMAS:
        errs = []
        for i in range(len(Z)):
            k = [j for j in range(len(Z)) if j != i]
            K = hybrid._kmat(Z[k], Z[k], g, True) + LAM2 * np.eye(len(k))
            a = np.linalg.lstsq(K, y[k], rcond=None)[0]
            kk = hybrid._kmat(Z[[i]], Z[k], g, True)
            errs.append(y[i] - float((kk @ a)[0]))
        e = float(np.mean(np.square(errs)))
        if e < best_e:
            best, best_e = g, e
    return best


def main() -> None:
    full = pd.read_parquet("data/interim/castelli_raw.parquet")
    A = sorted(full.site_A.unique())[:SUBGRID]
    B = sorted(full.site_B.unique())[:SUBGRID]
    sub = full[full.site_A.isin(A) & full.site_B.isin(B)].reset_index(drop=True)
    desc = {s: el.physical_descriptors(sorted(full[f"site_{s}"].unique().tolist()),
                                       "element" if s in ("A", "B") else "auto")
            for s in ("A", "B", "X")}
    idx = sub.set_index("sample_id")

    rows, t0 = [], time.time()
    for prop in PROPS:
        edges = build_edges(sub, prop, POLICY)
        f = full.dropna(subset=[prop])
        for site, levels in (("A", A), ("B", B)):
            other = {"A": ("B", B), "B": ("A", A)}[site]
            for lv in levels:
                d = obs.held_out_fiber(sub, site, lv)
                train = d.train_nodes(sub).dropna(subset=[prop]).reset_index(drop=True)
                te = d.induced_edges(edges).test
                if len(te) == 0 or train.empty:
                    continue
                y = te["delta"].to_numpy(float)
                src = idx.loc[te["src"]].reset_index()
                dst = idx.loc[te["dst"]].reset_index()

                def mae(pred):
                    return response_metrics(y, np.asarray(pred, float)).mae

                donor = f[(f[f"site_{other[0]}"].isin(other[1]))
                          & (f[f"site_{site}"] != lv)]

                # bandwidths come from the elements each arm's descriptor map is trained on
                eff_n = hybrid.fit_stage1(train, prop)
                g_n = pick_gamma(desc[site], eff_n.levels[site], eff_n.main[site])
                eff_w = hybrid.fit_stage1(donor, prop)
                g_w = pick_gamma(desc[site], eff_w.levels[site], eff_w.main[site])

                rec = {"property": prop, "site": site, "level": lv, "n_test": len(te),
                       "gamma_narrow": g_n, "gamma_wide": g_w}
                arms = {
                    "narrow_linear": dict(),
                    "wide_linear": dict(donor=donor, forbid=[(site, lv)]),
                    "narrow_kernel": dict(gamma=g_n, combined=True),
                    "wide_kernel": dict(donor=donor, forbid=[(site, lv)],
                                        gamma=g_w, combined=True),
                }
                for tag, kw in arms.items():
                    m = hybrid.fit(train, prop, desc, **kw)
                    rec[tag] = mae(m.predict(dst) - m.predict(src))
                    rec[f"{tag}_sign"] = float(np.mean(
                        np.sign(y) == np.sign(m.predict(dst) - m.predict(src))))

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
                pr = hg.predict(feats(dst)) - hg.predict(feats(src))
                rec["hgb"] = mae(pr)
                rec["hgb_sign"] = float(np.mean(np.sign(y) == np.sign(pr)))
                rows.append(rec)
                print(f"  {prop[:12]:12s} {site}={lv:3s} {time.time()-t0:.0f}s", flush=True)

    d = pd.DataFrame(rows)
    CSV.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(CSV, index=False)

    from scipy import stats
    ARMS = ("narrow_linear", "wide_linear", "narrow_kernel", "wide_kernel", "hgb")
    out = ["# Do a wider and a non-linear descriptor map combine?\n"]
    out.append(
        "_Generated by `scripts/run_left_out_element.py`._ Widening the descriptor map's element set and "
        "widening its function class act on different things, so they should compose. Both are "
        "follow-ups the manuscript's frozen estimator does not contain. All 28 unseen-level "
        "designs, both properties, same test pairs throughout.\n")
    out.append("| property | site | " + " | ".join(a.replace('_', ' ') for a in ARMS) + " |")
    out.append("|---|---|" + "---|" * len(ARMS))
    for prop in PROPS:
        for site in ("A", "B"):
            g = d[(d["property"] == prop) & (d["site"] == site)]
            if g.empty:
                continue
            out.append(f"| {prop.replace('_',' ')} | {site} | "
                       + " | ".join(f"{g[a].median():.4f}" for a in ARMS) + " |")

    out.append("\n## Against the manuscript's estimator, paired within design\n")
    out.append("| property | site | arm | median change | wins | Wilcoxon p |")
    out.append("|---|---|---|---|---|---|")
    for prop in PROPS:
        for site in ("A", "B"):
            g = d[(d["property"] == prop) & (d["site"] == site)]
            if len(g) < 6:
                continue
            b = g["narrow_linear"]
            for a in ARMS[1:]:
                pct = 100 * np.median((g[a] - b) / b)
                p = float(stats.wilcoxon(g[a] - b)[1])
                out.append(f"| {prop.replace('_',' ')} | {site} | {a.replace('_',' ')} | "
                           f"{pct:+.1f}% | {int((g[a] < b).sum())}/{len(g)} | {p:.4f} |")

    out.append("\n## The composed estimator against gradient boosting\n")
    out.append("| property | site | median change | wins | Wilcoxon p | sign, composed "
               "| sign, boosting |")
    out.append("|---|---|---|---|---|---|---|")
    for prop in PROPS:
        for site in ("A", "B"):
            g = d[(d["property"] == prop) & (d["site"] == site)]
            if len(g) < 6:
                continue
            pct = 100 * np.median((g["wide_kernel"] - g["hgb"]) / g["hgb"])
            p = float(stats.wilcoxon(g["wide_kernel"] - g["hgb"])[1])
            out.append(f"| {prop.replace('_',' ')} | {site} | {pct:+.1f}% | "
                       f"{int((g['wide_kernel'] < g['hgb']).sum())}/{len(g)} | {p:.4f} | "
                       f"{g['wide_kernel_sign'].median():.3f} | {g['hgb_sign'].median():.3f} |")

    out.append("\n## Do they compose?\n")
    out.append("| property | site | widening alone | kernel alone | sum of the two "
               "| both together |")
    out.append("|---|---|---|---|---|---|")
    for prop in PROPS:
        for site in ("A", "B"):
            g = d[(d["property"] == prop) & (d["site"] == site)]
            if g.empty:
                continue
            b = g["narrow_linear"]
            w = 100 * np.median((g["wide_linear"] - b) / b)
            k = 100 * np.median((g["narrow_kernel"] - b) / b)
            j = 100 * np.median((g["wide_kernel"] - b) / b)
            out.append(f"| {prop.replace('_',' ')} | {site} | {w:+.1f}% | {k:+.1f}% | "
                       f"{w+k:+.1f}% | **{j:+.1f}%** |")
    write_report(REPORT, "\n".join(out))
    print(d.groupby(["property", "site"])[list(ARMS)].median().round(4).to_string())
    print(f"\nwrote {REPORT} and {CSV}")


if __name__ == "__main__":
    main()
