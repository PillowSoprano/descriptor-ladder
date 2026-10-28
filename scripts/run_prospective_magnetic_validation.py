"""Execute the locked magnetic-moment validation without target-specific tuning.

The decision rules were written down before this script was run.  This program does not choose thresholds,
models, fractions or seeds from the results; it only evaluates the frozen choices.

Writes reports/prospective_magnetic_coverage.csv and
reports/PROSPECTIVE_MAGNETIC_VALIDATION.md.

Usage: PYTHONPATH=src python3 scripts/run_prospective_magnetic_validation.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prfield.audits.variance import decompose  # noqa: E402
from prfield.descriptors import elemental as el  # noqa: E402
from prfield.graph.typed_graph import LOOSE, build_edges  # noqa: E402
from prfield.metrics import response_metrics  # noqa: E402
from prfield.models import hybrid  # noqa: E402
from prfield.models.baselines import DifferenceFromNode, NodeModel  # noqa: E402
from prfield.report import write_report  # noqa: E402
from prfield.splits import observation as obs  # noqa: E402

PROP = "magnetic_moment"
FRACS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90)
SEEDS = tuple(range(100, 110))
SMOOTH = 0.1
CSV = Path("reports/prospective_magnetic_coverage.csv")
REPORT = Path("reports/PROSPECTIVE_MAGNETIC_VALIDATION.md")


def load_target() -> pd.DataFrame:
    df = pd.read_parquet("data/interim/castelli_raw.parquet").copy()
    if PROP not in df or df[PROP].isna().all():
        source = "_magmom_signed" if "_magmom_signed" in df else "_magmom"
        if source not in df:
            raise KeyError("cached Castelli frame has no magnetic-moment source column")
        df[PROP] = df[source].abs()
    if (df[PROP] < 0).any():
        raise ValueError("magnetic-moment magnitude must be non-negative")
    return df


def descriptors(df: pd.DataFrame):
    return {
        s: el.physical_descriptors(
            sorted(df[f"site_{s}"].unique().tolist()), "element" if s in ("A", "B") else "auto"
        )
        for s in ("A", "B", "X")
    }


def mae(true, pred) -> float:
    return response_metrics(np.asarray(true, float), np.asarray(pred, float)).mae


def main() -> None:
    full = load_target()
    A = sorted(full.site_A.unique())[:14]
    B = sorted(full.site_B.unique())[:14]
    sub = full[full.site_A.isin(A) & full.site_B.isin(B)].reset_index(drop=True)
    desc = descriptors(sub)
    edges = build_edges(sub, PROP, LOOSE)
    idx = sub.set_index("sample_id")

    rows = []
    t0 = time.time()
    for frac in FRACS:
        for seed in SEEDS:
            design = obs.random_nodes(sub, frac, seed=seed)
            train = design.train_nodes(sub).dropna(subset=[PROP]).reset_index(drop=True)
            test_edges = design.induced_edges(edges).test
            if train.empty or test_edges.empty:
                raise RuntimeError(f"empty train/test at fraction={frac}, seed={seed}")
            src = idx.loc[test_edges["src"]].reset_index()
            dst = idx.loc[test_edges["dst"]].reset_index()
            true = test_edges["delta"].to_numpy(float)

            basis = DifferenceFromNode(NodeModel(desc, PROP, interactions=True).fit(train))
            basis_mae = mae(true, basis.predict_delta(test_edges))

            fitted = {}
            for name, smooth in (("free", 0.0), ("metric", SMOOTH)):
                eff = hybrid.fit_stage1(train, PROP, smooth=smooth, desc=desc if smooth else None)
                maps, norm = hybrid.fit_stage2(eff, desc)
                model = hybrid.HybridModel(eff, maps, norm, desc)
                fitted[name] = (eff, mae(true, model.predict(dst) - model.predict(src)))

            eff = fitted["free"][0]
            raw, constraints, p_nominal = hybrid.parameter_dimensions(eff.levels)
            free_mae, metric_mae = fitted["free"][1], fitted["metric"][1]
            rows.append({
                "fraction": frac,
                "seed": seed,
                "n_train": len(train),
                "raw_parameters": raw,
                "constraints": constraints,
                "p_nominal": p_nominal,
                "n_over_p": len(train) / p_nominal,
                "basis_mae": basis_mae,
                "free_mae": free_mae,
                "metric_mae": metric_mae,
                "metric_benefit_pct": 100 * (free_mae - metric_mae) / free_mae,
                "basis_cost_pct": 100 * (basis_mae - metric_mae) / metric_mae,
                "n_test_edges": len(test_edges),
            })
        print(f"fraction={frac:.2f} complete  elapsed={time.time()-t0:.0f}s", flush=True)

    result = pd.DataFrame(rows)
    CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(CSV, index=False)

    summary = result.groupby("fraction", as_index=False).agg(
        n_train=("n_train", "median"),
        n_over_p=("n_over_p", "median"),
        basis_mae=("basis_mae", "median"),
        free_mae=("free_mae", "median"),
        metric_mae=("metric_mae", "median"),
        metric_benefit_pct=("metric_benefit_pct", "median"),
        basis_cost_pct=("basis_cost_pct", "median"),
    )
    in_band = summary[summary.n_over_p.between(0.7, 1.5)]
    dense = summary[summary.n_over_p >= 2.5]
    peak = summary.loc[summary.metric_benefit_pct.idxmax()]
    band_max = float(in_band.metric_benefit_pct.max()) if len(in_band) else float("nan")
    dense_median = float(dense.metric_benefit_pct.median()) if len(dense) else float("nan")
    decisions = {
        "1 metric direction": int((summary.metric_benefit_pct > 0).sum()) >= 5,
        "2 near-threshold maximum": 0.7 <= float(peak.n_over_p) <= 1.5,
        "3 dense-regime decay": dense_median < 0.5 * band_max,
        "4 no basis crossing": bool((summary.basis_cost_pct > 0).all()),
    }

    vd = decompose(full, PROP)
    lines = [
        "# Prospective magnetic-moment validation\n",
        "Generated by `scripts/run_prospective_magnetic_validation.py` under rules fixed "
        "before the run.\n",
        "## Target description\n",
        f"- observations: {len(full):,}; zero fraction: {(full[PROP] == 0).mean():.1%}",
        f"- median: {full[PROP].median():.4g}; 95th percentile: {full[PROP].quantile(.95):.4g}; "
        f"99th percentile: {full[PROP].quantile(.99):.4g} Bohr magnetons",
        f"- exact uniform-product ANOVA: main {vd.main:.1%}, pairs {vd.pair:.1%}, "
        f"triple {vd.triple:.1%}\n",
        "## Coverage results\n",
        "| observed | n/p | descriptor basis | free | + metric | metric benefit | basis cost |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary.itertuples(index=False):
        lines.append(
            f"| {r.fraction:.0%} | {r.n_over_p:.3f} | {r.basis_mae:.4f} | "
            f"{r.free_mae:.4f} | {r.metric_mae:.4f} | {r.metric_benefit_pct:+.1f}% | "
            f"{r.basis_cost_pct:+.1f}% |"
        )
    lines += [
        "\n## Locked decisions\n",
        f"- peak metric benefit: {peak.metric_benefit_pct:+.1f}% at n/p={peak.n_over_p:.3f}",
        f"- maximum in the locked threshold band: {band_max:+.1f}%",
        f"- median for n/p >= 2.5: {dense_median:+.1f}%",
    ]
    for name, passed in decisions.items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — {name}")
    primary = all(decisions[k] for k in list(decisions)[:3])
    lines += [
        "\n## Reading\n",
        ("The prospective n/p claim passes all three primary rules."
         if primary else
         "The prospective n/p claim does not pass all three primary rules; the failed rules "
         "remain part of the result and the manuscript claim must be narrowed accordingly."),
    ]
    write_report(REPORT, "\n".join(lines))
    print(f"wrote {CSV} and {REPORT}")


if __name__ == "__main__":
    main()
