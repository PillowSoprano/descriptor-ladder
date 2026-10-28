"""Semiparametric kernel transfer for every left-out B-site element of the complete grid.

Decision rules fixed before the run.

Writes model, uncertainty, support, phase, inner-CV and report files.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import optimize, stats
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prfield.descriptors import elemental as el  # noqa: E402
from prfield.models.semiparametric import (  # noqa: E402
    LinearHybridBModel,
    SemiparametricBModel,
    _descriptor_matrix,
    _standardise,
)
from run_as_cs_mechanism_audit import descriptor_features, sign_accuracy  # noqa: E402

DETAIL = Path("reports/semiparametric_kernel_validation.csv")
RISK = Path("reports/semiparametric_kernel_risk_coverage.csv")
SUPPORT = Path("reports/semiparametric_kernel_support.csv")
PHASE = Path("reports/semiparametric_kernel_phase_data.csv")
INNER = Path("reports/semiparametric_kernel_inner_cv.csv")
REPORT = Path("reports/AUDIT_SEMIPARAMETRIC_KERNEL.md")

TARGETS = ("heat_of_formation", "magnetic_moment", "volume_per_fu")
PRIMARY = ("heat_of_formation", "magnetic_moment")
MODEL_NAMES = ("hybrid", "node random forest", "node gradient boosting", "semiparametric kernel")
RISK_COVERAGES = (0.6, 0.8, 1.0)
RF_TREES = 400


def load_full_grid() -> pd.DataFrame:
    frame = pd.read_parquet("data/interim/castelli_raw.parquet").copy()
    frame["magnetic_moment"] = frame["_magmom"].abs().astype(float)
    return frame.reset_index(drop=True)


def descriptors(frame: pd.DataFrame):
    return {
        site: el.physical_descriptors(
            sorted(frame[f"site_{site}"].unique()),
            "element" if site in ("A", "B") else "auto",
        )
        for site in ("A", "B", "X")
    }


def held_b_test_edges(frame: pd.DataFrame, prop: str, held_level: str) -> pd.DataFrame:
    """All accepted substitution pairs involving one B-site element on this complete grid.

    The Castelli table has one prototype and one energy functional. Under the locked LOOSE
    policy, the test topology is therefore determined entirely by the 52 x 52 x 7 product
    coordinates. Constructing the 28,938 relevant edges directly is exactly equivalent to
    enumerating the million-edge full graph and filtering it, but avoids repeating that
    property-independent work three times.
    """
    a_levels = sorted(frame.site_A.unique().tolist())
    b_levels = sorted(frame.site_B.unique().tolist())
    x_levels = sorted(frame.site_X.unique().tolist())
    if frame.prototype_id.nunique(dropna=False) != 1:
        raise ValueError("direct full-grid edge construction requires one prototype")
    lookup = frame.set_index(["site_A", "site_B", "site_X"])["sample_id"].to_dict()
    values = frame.set_index("sample_id")[prop]
    records = []
    # A and X substitutions stay within the compositions of the left-out B-site element.
    for x in x_levels:
        for i, source_a in enumerate(a_levels):
            for destination_a in a_levels[i + 1:]:
                records.append((
                    "A", lookup[(source_a, held_level, x)],
                    lookup[(destination_a, held_level, x)], source_a, destination_a))
    for a in a_levels:
        for i, source_x in enumerate(x_levels):
            for destination_x in x_levels[i + 1:]:
                records.append((
                    "X", lookup[(a, held_level, source_x)],
                    lookup[(a, held_level, destination_x)], source_x, destination_x))
    # B substitutions connect the left-out element to every training element.
    for a in a_levels:
        for x in x_levels:
            for partner in b_levels:
                if partner == held_level:
                    continue
                source_b, destination_b = sorted([held_level, partner])
                records.append((
                    "B", lookup[(a, source_b, x)], lookup[(a, destination_b, x)],
                    source_b, destination_b))
    edges = pd.DataFrame(
        records, columns=["site", "src", "dst", "src_species", "dst_species"])
    edges["delta"] = values.loc[edges.dst].to_numpy(float) - values.loc[edges.src].to_numpy(float)
    return edges


def convex_hull_distance(train_z: np.ndarray, query_z: np.ndarray) -> float:
    """Euclidean distance to the convex hull of the outer-training descriptors."""
    n = len(train_z)
    objective = lambda w: float(np.sum((w @ train_z - query_z) ** 2))
    gradient = lambda w: 2.0 * (w @ train_z - query_z) @ train_z.T
    result = optimize.minimize(
        objective,
        np.repeat(1.0 / n, n),
        jac=gradient,
        bounds=[(0.0, 1.0)] * n,
        constraints={"type": "eq", "fun": lambda w: float(w.sum() - 1.0),
                     "jac": lambda w: np.ones_like(w)},
        method="SLSQP",
        options={"ftol": 1e-10, "maxiter": 500},
    )
    return float(np.sqrt(max(result.fun, 0.0)))


def oriented_b_margin(test: pd.DataFrame, values: np.ndarray, held_level: str) -> float:
    is_b = test.site.eq("B").to_numpy()
    if not is_b.any():
        return float("nan")
    b = test.loc[is_b]
    v = np.asarray(values, float)[is_b]
    held_is_src = b.src_species.astype(str).eq(held_level).to_numpy()
    held_minus_partner = np.where(held_is_src, -v, v)
    return float(np.mean(held_minus_partner))


def score_predictions(
    prop: str,
    level: str,
    test: pd.DataFrame,
    prediction: np.ndarray,
    model: str,
):
    truth = test.delta.to_numpy(float)
    is_b = test.site.eq("B").to_numpy()
    true_margin = oriented_b_margin(test, truth, level)
    pred_margin = oriented_b_margin(test, prediction, level)
    catastrophic = int(
        np.isfinite(true_margin) and np.isfinite(pred_margin)
        and np.sign(true_margin) != np.sign(pred_margin)
        and true_margin != 0 and pred_margin != 0
    )
    return {
        "property": prop,
        "held_site": "B",
        "held_level": level,
        "model": model,
        "n_edges": len(test),
        "mae": float(np.mean(np.abs(prediction - truth))),
        "sign_accuracy": sign_accuracy(truth, prediction),
        "b_mae": float(np.mean(np.abs(prediction[is_b] - truth[is_b]))),
        "b_sign_accuracy": sign_accuracy(truth[is_b], prediction[is_b]),
        "true_b_margin": true_margin,
        "pred_b_margin": pred_margin,
        "catastrophic_sign_error": catastrophic,
    }


def risk_coverage_rows(
    prop: str,
    level: str,
    truth: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
):
    standard_deviation = np.sqrt(np.maximum(variance, 1e-12))
    positive_probability = stats.norm.cdf(mean / standard_deviation)
    confidence = stats.norm.cdf(np.abs(mean) / standard_deviation)
    signed = np.abs(truth) > 0
    rows = []
    for coverage in RISK_COVERAGES:
        count = max(1, int(np.ceil(coverage * len(truth))))
        keep = np.argsort(-confidence, kind="stable")[:count]
        keep_signed = keep[signed[keep]]
        rows.append({
            "property": prop,
            "held_level": level,
            "coverage": coverage,
            "n_retained": count,
            "mae": float(np.mean(np.abs(mean[keep] - truth[keep]))),
            "sign_error": float(np.mean(np.sign(mean[keep_signed]) != np.sign(truth[keep_signed])))
            if len(keep_signed) else float("nan"),
            "mean_confidence": float(np.mean(confidence[keep])),
            "brier": float(np.mean((positive_probability[keep] - (truth[keep] > 0)) ** 2)),
            "interval90_coverage": float(np.mean(
                np.abs(truth[keep] - mean[keep]) <= 1.6448536269514722 * standard_deviation[keep])),
        })
    return rows


def run_outer(
    frame: pd.DataFrame,
    desc,
    prop: str,
    level: str,
):
    start = time.time()
    train = frame[frame.site_B.ne(level)].dropna(subset=[prop]).reset_index(drop=True)
    test = held_b_test_edges(frame, prop, level)
    index = frame.set_index("sample_id")
    src = index.loc[test.src].reset_index()
    dst = index.loc[test.dst].reset_index()
    truth = test.delta.to_numpy(float)

    hybrid = LinearHybridBModel.fit(train, prop, desc["B"])
    hybrid_prediction = hybrid.predict(dst) - hybrid.predict(src)

    semiparametric = SemiparametricBModel.fit(train, prop, desc["B"])
    semiparametric_prediction, semiparametric_variance = semiparametric.predict_edges(src, dst)

    x_train, fitted = descriptor_features(train, desc)
    x_src, _ = descriptor_features(src, desc, fitted)
    x_dst, _ = descriptor_features(dst, desc, fitted)
    y_train = train[prop].to_numpy(float)
    forest = RandomForestRegressor(
        n_estimators=RF_TREES, random_state=0, n_jobs=1
    ).fit(x_train, y_train)
    boosting = HistGradientBoostingRegressor(random_state=0).fit(x_train, y_train)
    forest_prediction = forest.predict(x_dst) - forest.predict(x_src)
    boosting_prediction = boosting.predict(x_dst) - boosting.predict(x_src)

    rows = [
        score_predictions(prop, level, test, hybrid_prediction, "hybrid"),
        score_predictions(prop, level, test, forest_prediction, "node random forest"),
        score_predictions(prop, level, test, boosting_prediction, "node gradient boosting"),
        score_predictions(prop, level, test, semiparametric_prediction, "semiparametric kernel"),
    ]
    semiparametric_row = rows[-1]
    standard_deviation = np.sqrt(np.maximum(semiparametric_variance, 1e-12))
    positive_probability = stats.norm.cdf(semiparametric_prediction / standard_deviation)
    semiparametric_row.update({
        "brier": float(np.mean((positive_probability - (truth > 0)) ** 2)),
        "interval90_coverage": float(np.mean(
            np.abs(truth - semiparametric_prediction)
            <= 1.6448536269514722 * standard_deviation)),
        "gamma": semiparametric.hyperparameters.gamma,
        "rho": semiparametric.hyperparameters.rho,
        "uncertainty_multiplier": semiparametric.hyperparameters.uncertainty_multiplier,
        "inner_score": semiparametric.hyperparameters.inner_score,
        "effective_df": semiparametric.effective_degrees_of_freedom(),
        "n_over_p_eff": len(semiparametric.b_levels)
        / semiparametric.effective_degrees_of_freedom(),
    })

    support = semiparametric.support([level]).iloc[0].to_dict()
    raw_train = _descriptor_matrix(desc["B"], semiparametric.b_levels)
    train_z, mean, scale = _standardise(raw_train)
    query_z, _, _ = _standardise(_descriptor_matrix(desc["B"], [level]), mean, scale)
    support.update({
        "property": prop,
        "held_level": level,
        "convex_hull_distance": convex_hull_distance(train_z, query_z[0]),
        "gamma": semiparametric.hyperparameters.gamma,
        "rho": semiparametric.hyperparameters.rho,
        "effective_df": semiparametric.effective_degrees_of_freedom(),
        "n_over_p_eff": len(semiparametric.b_levels)
        / semiparametric.effective_degrees_of_freedom(),
        "true_fibre_mean": float(frame.loc[frame.site_B.eq(level), prop].mean()),
        "training_mean": float(train[prop].mean()),
        "hybrid_fibre_mean": float(hybrid.predict(
            frame[frame.site_B.eq(level)].reset_index(drop=True)).mean()),
        "rf_fibre_mean": float(forest.predict(descriptor_features(
            frame[frame.site_B.eq(level)].reset_index(drop=True), desc, fitted)[0]).mean()),
        "hgb_fibre_mean": float(boosting.predict(descriptor_features(
            frame[frame.site_B.eq(level)].reset_index(drop=True), desc, fitted)[0]).mean()),
        "semiparametric_fibre_mean": float(semiparametric.predict(
            frame[frame.site_B.eq(level)].reset_index(drop=True)).mean()),
    })
    risk = risk_coverage_rows(
        prop, level, truth, semiparametric_prediction, semiparametric_variance)
    inner = semiparametric.inner_grid.copy()
    inner["property"] = prop
    inner["held_level"] = level
    inner["selected"] = (
        inner.gamma.eq(semiparametric.hyperparameters.gamma)
        & inner.rho.eq(semiparametric.hyperparameters.rho)
        & inner.uncertainty_multiplier.eq(
            semiparametric.hyperparameters.uncertainty_multiplier)
    )
    elapsed = time.time() - start
    for row in rows:
        row["seconds"] = elapsed
    print(
        f"{prop[:12]:12s} B={level:2s}  shared-trend={rows[0]['mae']:.3f} "
        f"RF={rows[1]['mae']:.3f} HGB={rows[2]['mae']:.3f} "
        f"semi={rows[3]['mae']:.3f}  {elapsed:.1f}s",
        flush=True,
    )
    return rows, risk, support, inner


def paired_summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for prop in detail.property.unique():
        candidate = detail[(detail.property.eq(prop))
                           & (detail.model.eq("semiparametric kernel"))].set_index("held_level")
        for comparator in ("hybrid", "node random forest", "node gradient boosting"):
            reference = detail[(detail.property.eq(prop))
                               & (detail.model.eq(comparator))].set_index("held_level")
            joined = candidate[["mae", "sign_accuracy", "catastrophic_sign_error"]].join(
                reference[["mae", "sign_accuracy", "catastrophic_sign_error"]],
                lsuffix="_candidate", rsuffix="_reference")
            change = 100.0 * (joined.mae_candidate - joined.mae_reference) / joined.mae_reference
            try:
                pvalue = float(stats.wilcoxon(
                    joined.mae_candidate - joined.mae_reference).pvalue)
            except ValueError:
                pvalue = 1.0
            rows.append({
                "property": prop,
                "comparator": comparator,
                "n_levels": len(joined),
                "candidate_median_mae": float(joined.mae_candidate.median()),
                "reference_median_mae": float(joined.mae_reference.median()),
                "median_pct_change": float(change.median()),
                "wins": int((joined.mae_candidate < joined.mae_reference).sum()),
                "median_sign_change": float(
                    (joined.sign_accuracy_candidate - joined.sign_accuracy_reference).median()),
                "candidate_catastrophic": int(joined.catastrophic_sign_error_candidate.sum()),
                "reference_catastrophic": int(joined.catastrophic_sign_error_reference.sum()),
                "wilcoxon_p": pvalue,
            })
    return pd.DataFrame(rows)


def build_report(detail: pd.DataFrame, risk: pd.DataFrame, support: pd.DataFrame) -> str:
    comparison = paired_summary(detail)
    primary_comparison = comparison[comparison.property.isin(PRIMARY)]

    check1 = all(
        primary_comparison[primary_comparison.comparator.eq("hybrid")]
        .set_index("property").loc[list(PRIMARY), "median_pct_change"] < 0
    )
    check2 = all(
        primary_comparison[primary_comparison.comparator.eq("node random forest")]
        .set_index("property").loc[list(PRIMARY), "median_pct_change"] <= 5
    )
    catastrophic = (detail[detail.property.isin(PRIMARY)]
                    .groupby("model").catastrophic_sign_error.sum())
    check3 = int(catastrophic.get("semiparametric kernel", 0)) \
        < int(catastrophic.get("hybrid", 0))
    formation_cs = detail[(detail.property.eq("heat_of_formation"))
                          & detail.held_level.eq("Cs")].set_index("model")
    check4 = (
        formation_cs.loc["semiparametric kernel", "mae"]
        <= formation_cs.loc["hybrid", "mae"]
        and formation_cs.loc["semiparametric kernel", "sign_accuracy"]
        >= formation_cs.loc["hybrid", "sign_accuracy"]
    )
    risk_summary = (risk[risk.property.isin(PRIMARY)]
                    .groupby(["property", "coverage"], as_index=False)
                    .agg(sign_error=("sign_error", "median"), mae=("mae", "median"),
                         brier=("brier", "median"),
                         interval90_coverage=("interval90_coverage", "median")))
    check5 = True
    for prop in PRIMARY:
        values = risk_summary[risk_summary.property.eq(prop)].set_index("coverage")
        check5 &= bool(values.loc[0.6, "sign_error"] < values.loc[1.0, "sign_error"])
        check5 &= bool(values.loc[0.8, "sign_error"] < values.loc[1.0, "sign_error"])
    checks = [
        (check1, "lower median MAE than the shared-trend model on both primary targets"),
        (check2, "within 5% of random forest on both primary targets"),
        (check3, "fewer direction reversals on the primary targets than the shared-trend model"),
        (check4, "formation-Cs MAE and sign are no worse than the shared-trend model"),
        (check5, "60% and 80% selective prediction lower sign error on both primary targets"),
    ]
    passed = all(value for value, _ in checks)

    lines = [
        "# Full-grid semiparametric kernel validation\n",
        "Decision rules fixed before the run.\n",
        "## Locked decision\n",
    ]
    lines.extend(f"- {'PASS' if value else 'FAIL'} — {label}" for value, label in checks)
    lines.append(
        f"\n**Semiparametric-kernel status: "
        f"{'PRIMARY BREAKTHROUGH CANDIDATE' if passed else 'FUTURE WORK'}.**\n")
    lines.extend([
        "## Paired outer-B comparisons\n",
        "| target | comparator | semi MAE | reference MAE | median change | wins | "
        "sign change | reversals semi/ref | p |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.property.replace('_', ' ')} | {row.comparator} | "
            f"{row.candidate_median_mae:.4f} | {row.reference_median_mae:.4f} | "
            f"{row.median_pct_change:+.1f}% | {row.wins}/{row.n_levels} | "
            f"{row.median_sign_change:+.3f} | {row.candidate_catastrophic}/"
            f"{row.reference_catastrophic} | {row.wilcoxon_p:.4g} |"
        )
    lines.extend([
        "\n## Risk-coverage summary\n",
        "| target | coverage | sign error | MAE | Brier | 90% interval coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in risk_summary.itertuples(index=False):
        lines.append(
            f"| {row.property.replace('_', ' ')} | {row.coverage:.0%} | "
            f"{row.sign_error:.3f} | {row.mae:.4f} | {row.brier:.3f} | "
            f"{row.interval90_coverage:.3f} |"
        )
    lines.extend([
        "\n## As/Cs cells\n",
        "| target | level | model | MAE | sign | B-edge MAE | B-edge sign | margin |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    selected = detail[detail.held_level.isin(["As", "Cs"])]
    for row in selected.itertuples(index=False):
        lines.append(
            f"| {row.property.replace('_', ' ')} | {row.held_level} | {row.model} | "
            f"{row.mae:.4f} | {row.sign_accuracy:.3f} | {row.b_mae:.4f} | "
            f"{row.b_sign_accuracy:.3f} | {row.pred_b_margin:+.4f} |"
        )
    lines.extend([
        "\n## Formation-Cs support diagnosis\n",
        "| kNN distance | kernel leverage | extrapolation instability | convex-hull distance "
        "| true/train/shared-trend/RF/HGB/semi element means |",
        "|---:|---:|---:|---:|---|",
    ])
    cs = support[(support.property.eq("heat_of_formation"))
                 & support.held_level.eq("Cs")].iloc[0]
    lines.append(
        f"| {cs.knn_distance:.3f} | {cs.kernel_leverage:.3f} | "
        f"{cs.extrapolation_instability:.3f} | {cs.convex_hull_distance:.3f} | "
        f"{cs.true_fibre_mean:.3f} / {cs.training_mean:.3f} / {cs.hybrid_fibre_mean:.3f} / "
        f"{cs.rf_fibre_mean:.3f} / {cs.hgb_fibre_mean:.3f} / "
        f"{cs.semiparametric_fibre_mean:.3f} |"
    )
    lines.append(
        "\nThe support diagnostics test shrinkage and extrapolation explanations only. They "
        "do not establish a caesium or arsenic mechanism.\n")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", help="comma-separated B levels; default is all 52")
    parser.add_argument("--targets", help="comma-separated targets; default is the locked three")
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()

    frame = load_full_grid()
    desc = descriptors(frame)
    levels = sorted(frame.site_B.unique().tolist())
    if args.levels:
        requested = [value.strip() for value in args.levels.split(",") if value.strip()]
        missing = sorted(set(requested) - set(levels))
        if missing:
            raise ValueError(f"unknown B levels: {missing}")
        levels = requested
    targets = list(TARGETS)
    if args.targets:
        targets = [value.strip() for value in args.targets.split(",") if value.strip()]
        missing = sorted(set(targets) - set(TARGETS))
        if missing:
            raise ValueError(f"targets outside locked set: {missing}")

    DETAIL.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for prop in targets:
        target_results = Parallel(n_jobs=args.n_jobs, prefer="threads", verbose=5)(
            delayed(run_outer)(frame, desc, prop, level)
            for level in levels
        )
        results.extend(target_results)
        # Target-level checkpoints make the 52-level run recoverable and inspectable without
        # changing the locked fit or evaluation order.
        pd.DataFrame([row for result in results for row in result[0]]).to_csv(
            DETAIL, index=False)
        pd.DataFrame([row for result in results for row in result[1]]).to_csv(
            RISK, index=False)
        pd.DataFrame([result[2] for result in results]).to_csv(SUPPORT, index=False)
        pd.concat([result[3] for result in results], ignore_index=True).to_csv(
            INNER, index=False)
        print(f"checkpointed {prop}", flush=True)

    detail = pd.DataFrame([row for result in results for row in result[0]])
    risk = pd.DataFrame([row for result in results for row in result[1]])
    support = pd.DataFrame([result[2] for result in results])
    inner = pd.concat([result[3] for result in results], ignore_index=True)

    best = (detail.sort_values(["property", "held_level", "mae", "model"])
            .groupby(["property", "held_level"], as_index=False).first()
            [["property", "held_level", "model", "mae", "sign_accuracy"]]
            .rename(columns={"model": "best_model", "mae": "best_mae",
                             "sign_accuracy": "best_sign_accuracy"}))
    phase = support.merge(best, on=["property", "held_level"], how="left")
    phase.to_csv(PHASE, index=False)

    # A trial run checks the runtime; the decision rules apply to the full run only.
    if set(levels) == set(frame.site_B.unique()) and set(targets) == set(TARGETS):
        REPORT.write_text(build_report(detail, risk, support))
        print(f"wrote {DETAIL}, {RISK}, {SUPPORT}, {PHASE}, {INNER}, and {REPORT}")
    else:
        print(f"trial run only; wrote {DETAIL}, {RISK}, {SUPPORT}, {PHASE}, and {INNER}")


if __name__ == "__main__":
    main()
