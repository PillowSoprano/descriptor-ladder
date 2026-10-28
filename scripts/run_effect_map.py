"""A non-linear descriptor map and a descriptor map weighted by how well each effect is determined.

The descriptor map carries every prediction for a left-out element. Two changes are compared with
the linear, equally weighted map:

  non-linear   a kernel map from element properties to site effects in place of the linear one
  weighted     each element's fitted effect weighted by its posterior precision, so an element
               fixed by ninety compositions counts more than one fixed by four

Three measurements:

  1. reduction checks: with a linear kernel the kernel map reproduces the linear one, and with
     uniform weights the weighted map reproduces the unweighted one
  2. leave-one-element-out error on the exact main effects of the complete grid, which measures the
     descriptor map alone
  3. response error on the leave-one-element-out designs

Kernel bandwidths are chosen by inner leave-one-out over the training elements only.

Writes reports/AUDIT_EFFECT_MAP.md and reports/effect_map.csv.

Usage:  PYTHONPATH=src python3 scripts/run_effect_map.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prfield.descriptors import elemental as el  # noqa: E402
from prfield.metrics import response_metrics  # noqa: E402
from prfield.models import hybrid  # noqa: E402
from prfield.report import write_report  # noqa: E402

REPORT = Path("reports/AUDIT_EFFECT_MAP.md")
CSV = Path("reports/effect_map.csv")
PROPS = ("heat_of_formation", "volume_per_fu")
SUBGRID = 14
LAM2 = 1e-2
GAMMAS = (0.003, 0.01, 0.03, 0.1, 0.3)
# The kernel gets its bandwidth chosen by inner cross-validation, so the linear map is given
# the matching courtesy: an arm whose ridge is chosen the same way, on the same folds. Without
# it a win for the kernel could just be a win for having been tuned.
LAMS = tuple(10.0 ** k for k in range(-3, 4))


# --------------------------------------------------------------------------- reduction checks

def reduction_checks(sub, desc) -> list[str]:
    """Both changes must be strict generalisations; checked numerically."""
    out = []
    rng = np.random.default_rng(0)
    train = sub[rng.random(len(sub)) < 0.35].reset_index(drop=True)
    eff = hybrid.fit_stage1(train, "heat_of_formation", with_variance=True)

    # (a) uniform weights must reproduce the unweighted fit
    m0, _ = hybrid.fit_stage2(eff, desc, LAM2)
    flat = hybrid.Stage1Effects(
        grand=eff.grand, levels=eff.levels, main=eff.main, inter=eff.inter,
        n_obs=eff.n_obs, rank_deficient=eff.rank_deficient,
        main_var={s: np.ones_like(v) for s, v in eff.main_var.items()},
        inter_var=eff.inter_var)
    m1, _ = hybrid.fit_stage2(flat, desc, LAM2, precision_weighted=True)
    d_w = max(float(np.abs(m0.main_coef[s] - m1.main_coef[s]).max()) for s in "ABX")
    out.append(f"- uniform weights reproduce the unweighted map to {d_w:.2e}")

    # (b) a linear kernel must reproduce the primal ridge. Patch the kernel rather than add a
    #     second code path, so what is checked is the code that actually runs.
    real_rbf = hybrid._rbf
    hybrid._rbf = lambda A, B, g: A @ B.T
    try:
        mk, nk = hybrid.fit_stage2(eff, desc, LAM2, gamma=1.0)
        lin, nl = hybrid.fit_stage2(eff, desc, LAM2)
        worst = 0.0
        for s in "ABX":
            Z, _, _ = hybrid._phi(desc[s], eff.levels[s], *nl[s])
            # the primal path carries an intercept column the kernel path drops, so compare
            # predictions on centred features, which is where the two are the same estimator
            k = hybrid._rbf(Z[:, 1:], mk.train_feat[s], 1.0)
            pk = k @ mk.main_dual[s]
            pp = Z[:, 1:] @ np.linalg.lstsq(
                Z[:, 1:].T @ Z[:, 1:] + LAM2 * np.eye(Z.shape[1] - 1),
                Z[:, 1:].T @ eff.main[s], rcond=None)[0]
            worst = max(worst, float(np.abs(pk - pp).max()))
    finally:
        hybrid._rbf = real_rbf
    out.append(f"- a linear kernel reproduces the primal ridge to {worst:.2e}")
    return out


# --------------------------------------------------------------------- exact-effect transfer

def exact_main_effects(df, site, prop):
    g = df.dropna(subset=[prop])
    return (g.groupby(f"site_{site}")[prop].mean() - g[prop].mean()).to_dict()


def _std(M, keep):
    mu, sd = M[keep].mean(0), M[keep].std(0)
    return (M - mu) / np.where(sd > 0, sd, 1.0)


def pick_lambda(Z, y, keep):
    """Inner leave-one-out over the training levels, for the linear map's ridge."""
    best, best_e = LAM2, np.inf
    for lam in LAMS:
        errs = []
        for j in keep:
            sub_ = [k for k in keep if k != j]
            Ptr = np.hstack([np.ones((len(sub_), 1)), Z[sub_]])
            c = hybrid._ridge(Ptr, y[sub_], lam)
            errs.append(y[j] - float(np.hstack([[1.0], Z[j]]) @ c))
        e = float(np.mean(np.square(errs)))
        if e < best_e:
            best, best_e = lam, e
    return best


def pick_gamma(Z, y, keep):
    """Inner leave-one-out over the training levels; never sees the target level."""
    best, best_e = GAMMAS[0], np.inf
    for g in GAMMAS:
        errs = []
        for j in keep:
            sub_ = [k for k in keep if k != j]
            K = np.exp(-g * ((Z[sub_][:, None] - Z[sub_][None]) ** 2).sum(-1))
            a = np.linalg.lstsq(K + LAM2 * np.eye(len(K)), y[sub_], rcond=None)[0]
            kk = np.exp(-g * ((Z[[j]][:, None] - Z[sub_][None]) ** 2).sum(-1))
            errs.append(y[j] - float((kk @ a)[0]))
        e = float(np.mean(np.square(errs)))
        if e < best_e:
            best, best_e = g, e
    return best


def transfer_maps(df, desc_all) -> list[dict]:
    rows = []
    for prop in PROPS:
        for site in ("A", "B"):
            eff = exact_main_effects(df, site, prop)
            labels = sorted(eff)
            d = el.physical_descriptors(labels, "element")
            M = np.nan_to_num(np.asarray(d.matrix(labels, allow_missing=True), float))
            y = np.array([eff[lv] for lv in labels])
            for i in range(len(labels)):
                keep = [k for k in range(len(labels)) if k != i]
                Z = _std(M, keep)
                pred = {}
                Ptr = np.hstack([np.ones((len(keep), 1)), Z[keep]])
                Pte = np.hstack([[1.0], Z[i]])
                pred["linear"] = float(Pte @ hybrid._ridge(Ptr, y[keep], LAM2))
                lam = pick_lambda(Z, y, keep)
                pred["linear_tuned"] = float(Pte @ hybrid._ridge(Ptr, y[keep], lam))
                g = pick_gamma(Z, y, keep)
                K = np.exp(-g * ((Z[keep][:, None] - Z[keep][None]) ** 2).sum(-1))
                a = np.linalg.lstsq(K + LAM2 * np.eye(len(K)), y[keep], rcond=None)[0]
                kk = np.exp(-g * ((Z[[i]][:, None] - Z[keep][None]) ** 2).sum(-1))
                pred["rbf"] = float((kk @ a)[0])
                pred["rf"] = float(RandomForestRegressor(
                    n_estimators=300, random_state=0, n_jobs=-1
                ).fit(Z[keep], y[keep]).predict(Z[[i]])[0])
                pred["hgb"] = float(HistGradientBoostingRegressor(
                    random_state=0).fit(Z[keep], y[keep]).predict(Z[[i]])[0])
                pred["knn"] = float(KNeighborsRegressor(
                    n_neighbors=3, weights="distance"
                ).fit(Z[keep], y[keep]).predict(Z[[i]])[0])
                for k, v in pred.items():
                    rows.append({"stage": "exact_effect", "property": prop, "site": site,
                                 "level": labels[i], "map": k, "gamma": g if k == "rbf" else np.nan,
                                 "abs_err": abs(y[i] - v)})
    return rows


# ------------------------------------------------------------------------------- end-to-end

def fibre_response(sub, desc) -> list[dict]:
    rows = []
    for prop in PROPS:
        s = sub.dropna(subset=[prop]).reset_index(drop=True)
        for site in ("A", "B"):
            for lv in sorted(s[f"site_{site}"].unique()):
                train = s[s[f"site_{site}"] != lv].reset_index(drop=True)
                test = s[s[f"site_{site}"] == lv].reset_index(drop=True)
                if train.empty or len(test) < 2:
                    continue
                pairs = hybrid.substitution_edges(test)
                if len(pairs) == 0:
                    continue
                true = test[prop].to_numpy(float)
                dt = true[pairs[:, 1]] - true[pairs[:, 0]]
                eff = hybrid.fit_stage1(train, prop, with_variance=True)

                # bandwidth from the training levels only, on their estimated effects
                d = el.physical_descriptors(eff.levels[site], "element")
                Mz = np.nan_to_num(np.asarray(
                    d.matrix(eff.levels[site], allow_missing=True), float))
                Zg = _std(Mz, list(range(len(Mz))))
                g = pick_gamma(Zg, eff.main[site], list(range(len(Zg))))

                variants = {
                    "linear": dict(),
                    "weighted": dict(precision_weighted=True),
                    "rbf": dict(gamma=g),
                    "rbf+weighted": dict(gamma=g, precision_weighted=True),
                }
                for tag, kw in variants.items():
                    maps, norm = hybrid.fit_stage2(eff, desc, LAM2, **kw)
                    m = hybrid.HybridModel(eff, maps, norm, desc)
                    pr = m.predict(test)
                    dp = pr[pairs[:, 1]] - pr[pairs[:, 0]]
                    ok = np.isfinite(dt) & np.isfinite(dp)
                    rows.append({
                        "stage": "response", "property": prop, "site": site, "level": lv,
                        "map": tag, "gamma": g if "rbf" in tag else np.nan,
                        "n_pairs": int(ok.sum()),
                        "abs_err": float(response_metrics(dt[ok], dp[ok]).mae),
                        "sign": float(np.mean(np.sign(dt[ok]) == np.sign(dp[ok])))})
            print(f"  response {prop[:12]:12s} {site} done", flush=True)
    return rows


def main() -> None:
    full = pd.read_parquet("data/interim/castelli_raw.parquet")
    A = sorted(full.site_A.unique())[:SUBGRID]
    B = sorted(full.site_B.unique())[:SUBGRID]
    sub = full[full.site_A.isin(A) & full.site_B.isin(B)].reset_index(drop=True)
    desc = {s: el.physical_descriptors(sorted(full[f"site_{s}"].unique().tolist()),
                                       "element" if s in ("A", "B") else "auto")
            for s in ("A", "B", "X")}

    t0 = time.time()
    checks = reduction_checks(sub, desc)
    print("\n".join(checks), flush=True)
    rows = transfer_maps(full, desc)
    print(f"  exact-effect transfer done  {time.time()-t0:.0f}s", flush=True)
    rows += fibre_response(sub, desc)

    d = pd.DataFrame(rows)
    CSV.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(CSV, index=False)

    from scipy import stats
    out = ["# A non-linear descriptor map, and one weighted by how well each effect is determined\n"]
    out.append(
        "_Generated by `scripts/run_effect_map.py`._ Both changes are strict "
        "generalisations of the linear descriptor map, and the reductions are checked rather than "
        "assumed:\n")
    out += checks
    out.append("\n## Transfer on exact main effects\n")
    out.append(
        "The full grid is complete, so a level's main effect is exact and leave-one-level-out "
        "error measures the descriptor map alone. All 52 levels per site.\n")
    out.append("| property | site | linear | linear, tuned ridge | RBF kernel | random forest "
               "| boosting | 3-NN |")
    out.append("|---|---|---|---|---|---|---|---|")
    ex = d[d["stage"] == "exact_effect"]
    for prop in PROPS:
        for site in ("A", "B"):
            g = ex[(ex["property"] == prop) & (ex["site"] == site)]
            if g.empty:
                continue
            cells = [f"{g[g['map'] == k]['abs_err'].median():.4f}"
                     for k in ("linear", "linear_tuned", "rbf", "rf", "hgb", "knn")]
            out.append(f"| {prop.replace('_',' ')} | {site} | " + " | ".join(cells) + " |")
    out.append("\nPaired against the linear map, over all 52 levels:\n")
    out.append("| property | site | map | median change | wins | Wilcoxon p |")
    out.append("|---|---|---|---|---|---|")
    for prop in PROPS:
        for site in ("A", "B"):
            base = ex[(ex["property"] == prop) & (ex["site"] == site)
                      & (ex["map"] == "linear")].set_index("level")["abs_err"]
            for k in ("linear_tuned", "rbf", "rf", "hgb", "knn"):
                o = ex[(ex["property"] == prop) & (ex["site"] == site)
                       & (ex["map"] == k)].set_index("level")["abs_err"]
                o = o.reindex(base.index)
                pct = 100 * np.median((o - base) / base)
                p = float(stats.wilcoxon(o - base)[1])
                out.append(f"| {prop.replace('_',' ')} | {site} | {k} | {pct:+.1f}% | "
                           f"{int((o < base).sum())}/{len(base)} | {p:.4f} |")

    out.append("\n## Response error on the leave-one-element-out designs\n")
    out.append("| property | site | linear | + precision weights | RBF | RBF + weights |")
    out.append("|---|---|---|---|---|---|")
    rs = d[d["stage"] == "response"]
    for prop in PROPS:
        for site in ("A", "B"):
            g = rs[(rs["property"] == prop) & (rs["site"] == site)]
            if g.empty:
                continue
            cells = [f"{g[g['map'] == k]['abs_err'].median():.4f}"
                     for k in ("linear", "weighted", "rbf", "rbf+weighted")]
            out.append(f"| {prop.replace('_',' ')} | {site} | " + " | ".join(cells) + " |")
    out.append("\nPaired against the linear descriptor map, over the 14 held-out levels:\n")
    out.append("| property | site | variant | median change | wins | Wilcoxon p |")
    out.append("|---|---|---|---|---|---|")
    for prop in PROPS:
        for site in ("A", "B"):
            base = rs[(rs["property"] == prop) & (rs["site"] == site)
                      & (rs["map"] == "linear")].set_index("level")["abs_err"]
            if base.empty:
                continue
            for k in ("weighted", "rbf", "rbf+weighted"):
                o = rs[(rs["property"] == prop) & (rs["site"] == site)
                       & (rs["map"] == k)].set_index("level")["abs_err"].reindex(base.index)
                pct = 100 * np.median((o - base) / base)
                p = float(stats.wilcoxon(o - base)[1]) if len(base) > 5 else float("nan")
                out.append(f"| {prop.replace('_',' ')} | {site} | {k} | {pct:+.1f}% | "
                           f"{int((o < base).sum())}/{len(base)} | {p:.4f} |")
    write_report(REPORT, "\n".join(out))
    print(f"\nwrote {REPORT} and {CSV}")


if __name__ == "__main__":
    main()
