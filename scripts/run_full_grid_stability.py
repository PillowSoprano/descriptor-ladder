"""How far the full-grid block holdouts sit from stability, and what a formula-only model can see.

Reruns the fifteen formation-energy block holdouts of scripts/run_full_grid.py with the same four
models and keeps every scored response, then splits the errors by the heat of formation of the two
compositions of each response relative to 0.21 eV per atom, the threshold used to screen the
database. It also counts the compositions that share their chemical formula with the composition
that swaps the A- and B-site elements, which a model reading only the formula cannot tell apart.

Writes reports/full_grid_stability.csv, reports/full_grid_stability_bands.csv and
reports/AUDIT_FULL_GRID_STABILITY.md.

Usage:  PYTHONPATH=src python3 scripts/run_full_grid_stability.py
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
from prfield.graph.typed_graph import build_edges  # noqa: E402
from prfield.models import hybrid  # noqa: E402
from prfield.models.baselines import DifferenceFromNode, NodeModel  # noqa: E402
from prfield.report import write_report  # noqa: E402
from prfield.splits import observation as obs  # noqa: E402
from run_full_grid import KS, MAX_TEST, POLICY, SMOOTH  # noqa: E402
from run_nonlinear_baseline import features, scalers  # noqa: E402

REPORT = Path("reports/AUDIT_FULL_GRID_STABILITY.md")
CSV = Path("reports/full_grid_stability.csv")
BANDS = Path("reports/full_grid_stability_bands.csv")
PROP = "heat_of_formation"
WINDOW = 0.21         # eV per atom above the reference phases
MODELS = ("desc_node", "cat_node", "cat_smooth", "hgb")


def formula_collisions(grid: pd.DataFrame) -> dict:
    off = grid[grid.site_A != grid.site_B]
    key = [tuple(sorted((a, b))) + (x,) for a, b, x in zip(off.site_A, off.site_B, off.site_X)]
    groups = off.assign(key=key).groupby("key")[PROP]
    pairs = groups.filter(lambda s: len(s) == 2)
    gap = groups.agg(lambda s: abs(s.iloc[0] - s.iloc[1]) if len(s) == 2 else np.nan).dropna()
    return {"n_compositions": len(grid), "n_sharing_formula": len(pairs),
            "median_swap_change": float(gap.median()),
            "n_in_window": int((grid[PROP] <= WINDOW).sum())}


def main() -> None:
    df = pd.read_parquet("data/interim/castelli_raw.parquet")
    A, B = sorted(df.site_A.unique()), sorted(df.site_B.unique())
    grid = df.reset_index(drop=True)
    desc = {s: el.physical_descriptors(sorted(grid[f"site_{s}"].unique().tolist()),
                                       "element" if s in ("A", "B") else "auto")
            for s in ("A", "B", "X")}
    rng = np.random.default_rng(0)
    designs = []
    for k in KS:
        for rep in range(5):
            aa = list(rng.choice(A, size=k, replace=False))
            bb = list(rng.choice(B, size=k, replace=False))
            designs.append((f"block {k}A x {k}B #{rep}",
                            obs.held_out_block(grid, {"A": aa, "B": bb})))

    edges = build_edges(grid, PROP, POLICY)
    idx = grid.set_index("sample_id")
    frames, t0 = [], time.time()
    for name, d in designs:
        train = d.train_nodes(grid).dropna(subset=[PROP]).reset_index(drop=True)
        te = d.induced_edges(edges).test
        if len(te) > MAX_TEST:
            te = te.iloc[np.random.default_rng(0).choice(len(te), MAX_TEST, replace=False)]
        src, dst = idx.loc[te["src"]].reset_index(), idx.loc[te["dst"]].reset_index()
        out = pd.DataFrame({"design": name, "y": te["delta"].to_numpy(float),
                            "hf_src": src[PROP].to_numpy(float), "hf_dst": dst[PROP].to_numpy(float)})
        nm = NodeModel(desc, PROP, interactions=True).fit(train)
        out["desc_node"] = np.asarray(DifferenceFromNode(nm).predict_delta(te), float)
        sc = scalers(train, desc)
        hgb = HistGradientBoostingRegressor(random_state=0).fit(
            features(train, desc, sc), train[PROP].to_numpy(float))
        out["hgb"] = hgb.predict(features(dst, desc, sc)) - hgb.predict(features(src, desc, sc))
        for tag, g in (("cat_node", 0.0), ("cat_smooth", SMOOTH)):
            eff = hybrid.fit_stage1(train, PROP, smooth=g, desc=desc)
            maps, norm = hybrid.fit_stage2(eff, desc)
            m = hybrid.HybridModel(eff, maps, norm, desc)
            out[tag] = np.asarray(m.predict(dst) - m.predict(src), float)
        frames.append(out)
        print(f"  {name:20s} {time.time() - t0:.0f}s", flush=True)
    d = pd.concat(frames, ignore_index=True)
    d["in_window"] = d[["hf_src", "hf_dst"]].max(axis=1) <= WINDOW

    rows = []
    for (design, inside), g in d.groupby(["design", "in_window"]):
        rec = {"design": design, "both_in_window": bool(inside), "n": len(g),
               "mean_abs_response": float(np.abs(g.y).mean())}
        rec.update({m: float(np.abs(g[m] - g.y).mean()) for m in MODELS})
        rows.append(rec)
    per = pd.DataFrame(rows)
    allrows = d.groupby("design").apply(
        lambda g: pd.Series({"n": len(g), "mean_abs_response": float(np.abs(g.y).mean()),
                             **{m: float(np.abs(g[m] - g.y).mean()) for m in MODELS}}),
        include_groups=False).reset_index().assign(both_in_window="all")
    per = pd.concat([allrows, per], ignore_index=True)
    CSV.parent.mkdir(parents=True, exist_ok=True)
    per.to_csv(CSV, index=False)

    win = per[per.both_in_window.astype(str) == "True"]
    total = per[per.both_in_window.astype(str) == "all"]
    bands = []
    for label, g in (("all responses", total), ("both compositions within 0.21 eV/atom", win)):
        rec = {"responses": label, "n": int(g.n.sum()),
               "mean_abs_response": g.mean_abs_response.median()}
        rec.update({m: g[m].median() for m in MODELS})
        rec["basis_vs_free_pct"] = float(np.median(100 * (g.desc_node / g.cat_node - 1)))
        rec["prior_vs_free_pct"] = float(np.median(100 * (g.cat_smooth / g.cat_node - 1)))
        rec["prior_wins"] = int((g.cat_smooth < g.cat_node).sum())
        rec["basis_losses"] = int((g.desc_node > g.cat_node).sum())
        rec["prior_beats_no_change"] = int((g.cat_smooth < g.mean_abs_response).sum())
        rec["designs"] = len(g)
        bands.append(rec)
    bands = pd.DataFrame(bands)
    bands.to_csv(BANDS, index=False)
    fc = formula_collisions(grid)

    out = ["# Distance from stability on the full grid\n",
           "_Generated by `scripts/run_full_grid_stability.py`._ The fifteen formation-energy block "
           "holdouts of `scripts/run_full_grid.py`, split by whether both compositions of a scored "
           f"response lie within {WINDOW} eV per atom of their reference phases. Errors are the median "
           "over designs of the per-design mean absolute error (eV per atom).\n",
           "| responses | n | mean abs. response | basis | free effects | prior | gradient boosting "
           "| basis vs free | prior vs free | prior wins | basis losses | prior beats no change |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in bands.itertuples():
        out.append(f"| {r.responses} | {r.n:,} | {r.mean_abs_response:.3f} | {r.desc_node:.3f} "
                   f"| {r.cat_node:.3f} | {r.cat_smooth:.3f} | {r.hgb:.3f} "
                   f"| {r.basis_vs_free_pct:+.1f}% | {r.prior_vs_free_pct:+.1f}% "
                   f"| {r.prior_wins}/{r.designs} | {r.basis_losses}/{r.designs} "
                   f"| {r.prior_beats_no_change}/{r.designs} |")
    out += ["\n## The grid and the chemical formula\n",
            f"- compositions: {fc['n_compositions']:,}; within {WINDOW} eV per atom of their "
            f"reference phases: {fc['n_in_window']:,}",
            f"- compositions sharing their formula with the A/B swap: {fc['n_sharing_formula']:,}",
            f"- median change in formation energy on the swap: {fc['median_swap_change']:.2f} eV per atom"]
    write_report(REPORT, "\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
