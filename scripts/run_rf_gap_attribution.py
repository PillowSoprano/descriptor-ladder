"""Deterministic attribution of the random-forest versus semiparametric gap.

Deterministic diagnostics computed after the results.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prfield.models.semiparametric import (  # noqa: E402
    SemiparametricBModel,
    decompose_two_way_field,
    project_cube_to_second_order,
)
from run_as_cs_mechanism_audit import descriptor_features  # noqa: E402
from run_semiparametric_kernel_validation import (  # noqa: E402
    RF_TREES,
    TARGETS,
    descriptors,
    held_b_test_edges,
    load_full_grid,
    score_predictions,
)

VALIDATION = Path("reports/rf_gap_attribution_validation.csv")
COMPONENTS = Path("reports/rf_gap_attribution_components.csv")
REPORT = Path("reports/AUDIT_RF_GAP_ATTRIBUTION.md")

MODELS = (
    "semiparametric kernel",
    "node random forest",
    "RF second-order projection",
    "semiparametric + true held triple oracle",
)


def cube_axes(frame: pd.DataFrame):
    return (
        sorted(frame.site_A.unique().tolist()),
        sorted(frame.site_B.unique().tolist()),
        sorted(frame.site_X.unique().tolist()),
    )


def values_to_cube(frame: pd.DataFrame, values: np.ndarray, axes=None) -> np.ndarray:
    """Map values aligned with frame rows to a B-by-A-by-X cube."""
    if axes is None:
        axes = cube_axes(frame)
    a_levels, b_levels, x_levels = axes
    ai = {value: index for index, value in enumerate(a_levels)}
    bi = {value: index for index, value in enumerate(b_levels)}
    xi = {value: index for index, value in enumerate(x_levels)}
    cube = np.empty((len(b_levels), len(a_levels), len(x_levels)), dtype=float)
    for row, value in zip(frame.itertuples(index=False), np.asarray(values, dtype=float)):
        cube[bi[row.site_B], ai[row.site_A], xi[row.site_X]] = value
    return cube


def cube_to_values(frame: pd.DataFrame, cube: np.ndarray, axes=None) -> np.ndarray:
    if axes is None:
        axes = cube_axes(frame)
    a_levels, b_levels, x_levels = axes
    ai = {value: index for index, value in enumerate(a_levels)}
    bi = {value: index for index, value in enumerate(b_levels)}
    xi = {value: index for index, value in enumerate(x_levels)}
    return np.asarray([
        cube[bi[row.site_B], ai[row.site_A], xi[row.site_X]]
        for row in frame.itertuples(index=False)
    ])


def edge_deltas(test: pd.DataFrame, frame: pd.DataFrame, node_values: np.ndarray) -> np.ndarray:
    positions = pd.Series(np.arange(len(frame)), index=frame.sample_id)
    src = positions.loc[test.src].to_numpy(int)
    dst = positions.loc[test.dst].to_numpy(int)
    values = np.asarray(node_values, dtype=float)
    return values[dst] - values[src]


def balanced_b_edge_check(test: pd.DataFrame, frame: pd.DataFrame, held_level: str):
    b_edges = test[test.site.eq("B")]
    indexed = frame.set_index("sample_id")
    src = indexed.loc[b_edges.src].reset_index()
    dst = indexed.loc[b_edges.dst].reset_index()
    combinations = []
    for source, destination in zip(src.itertuples(index=False), dst.itertuples(index=False)):
        if source.site_B == held_level:
            held, partner = source, destination.site_B
        elif destination.site_B == held_level:
            held, partner = destination, source.site_B
        else:
            raise AssertionError("a scored B pair does not involve the left-out element")
        if source.site_A != destination.site_A or source.site_X != destination.site_X:
            raise AssertionError("a B substitution changed another site")
        combinations.append((held.site_A, held.site_X, partner))
    counts = pd.Series(combinations).value_counts()
    a_levels, b_levels, x_levels = cube_axes(frame)
    expected = len(a_levels) * len(x_levels) * (len(b_levels) - 1)
    balanced = len(combinations) == expected and len(counts) == expected and bool(counts.eq(1).all())
    return len(combinations), expected, balanced


def cube_margin(cube: np.ndarray, held_index: int) -> float:
    return float(cube[held_index].mean() - np.delete(cube, held_index, axis=0).mean())


def run_outer(frame, desc, prop: str, held_level: str, true_triple: np.ndarray, axes):
    start = time.time()
    a_levels, b_levels, x_levels = axes
    held_index = b_levels.index(held_level)
    train = frame[frame.site_B.ne(held_level)].dropna(subset=[prop]).reset_index(drop=True)
    test = held_b_test_edges(frame, prop, held_level)

    semiparametric = SemiparametricBModel.fit(train, prop, desc["B"])
    semi_values = semiparametric.predict(frame)
    semi_cube = values_to_cube(frame, semi_values, axes)

    x_train, fitted = descriptor_features(train, desc)
    x_full, _ = descriptor_features(frame, desc, fitted)
    forest = RandomForestRegressor(
        n_estimators=RF_TREES, random_state=0, n_jobs=1
    ).fit(x_train, train[prop].to_numpy(float))
    rf_values = forest.predict(x_full)
    rf_cube = values_to_cube(frame, rf_values, axes)
    projected_rf_cube, predicted_rf_triple = project_cube_to_second_order(rf_cube)
    projected_rf_values = cube_to_values(frame, projected_rf_cube, axes)

    oracle_cube = semi_cube.copy()
    oracle_cube[held_index] += true_triple[held_index]
    oracle_values = cube_to_values(frame, oracle_cube, axes)

    predictions = {
        "semiparametric kernel": edge_deltas(test, frame, semi_values),
        "node random forest": edge_deltas(test, frame, rf_values),
        "RF second-order projection": edge_deltas(test, frame, projected_rf_values),
        "semiparametric + true held triple oracle": edge_deltas(test, frame, oracle_values),
    }
    validation = []
    for model in MODELS:
        row = score_predictions(prop, held_level, test, predictions[model], model)
        row["seconds"] = time.time() - start
        validation.append(row)

    difference = rf_cube[held_index] - semi_cube[held_index]
    mean, a_effect, x_effect, ax_residual = decompose_two_way_field(difference)
    block_mse = {
        "fibre_mean_mse": mean ** 2,
        "a_conditional_mse": float(np.mean(a_effect ** 2)),
        "x_conditional_mse": float(np.mean(x_effect ** 2)),
        "ax_residual_mse": float(np.mean(ax_residual ** 2)),
    }
    total_mse = float(np.mean(difference ** 2))
    b_count, b_expected, b_balanced = balanced_b_edge_check(test, frame, held_level)
    semi_margin = cube_margin(semi_cube, held_index)
    oracle_margin = cube_margin(oracle_cube, held_index)
    truth_cube = values_to_cube(frame, frame[prop].to_numpy(float), axes)
    true_margin = cube_margin(truth_cube, held_index)
    scored_truth_margin = validation[0]["true_b_margin"]
    component = {
        "property": prop,
        "held_level": held_level,
        "rf_minus_semi_fibre_mean": mean,
        **block_mse,
        "total_gap_field_mse": total_mse,
        "orthogonality_error": sum(block_mse.values()) - total_mse,
        "fibre_mean_share": block_mse["fibre_mean_mse"] / total_mse if total_mse else 0.0,
        "a_conditional_share": block_mse["a_conditional_mse"] / total_mse if total_mse else 0.0,
        "x_conditional_share": block_mse["x_conditional_mse"] / total_mse if total_mse else 0.0,
        "ax_residual_share": block_mse["ax_residual_mse"] / total_mse if total_mse else 0.0,
        "rf_predicted_triple_mse_held": float(np.mean(predicted_rf_triple[held_index] ** 2)),
        "true_triple_mse_held": float(np.mean(true_triple[held_index] ** 2)),
        "true_triple_fibre_mean": float(true_triple[held_index].mean()),
        "semi_fibre_mean": float(semi_cube[held_index].mean()),
        "rf_fibre_mean": float(rf_cube[held_index].mean()),
        "true_fibre_mean": float(truth_cube[held_index].mean()),
        "b_edge_count": b_count,
        "b_edge_expected": b_expected,
        "b_edge_balanced": b_balanced,
        "true_margin_from_edges": scored_truth_margin,
        "true_margin_from_cube": true_margin,
        "true_margin_weighting_error": scored_truth_margin - true_margin,
        "semi_margin_from_cube": semi_margin,
        "triple_oracle_margin_from_cube": oracle_margin,
        "triple_oracle_margin_change": oracle_margin - semi_margin,
    }
    print(
        f"{prop[:12]:12s} B={held_level:2s} semi={validation[0]['mae']:.3f} "
        f"RF={validation[1]['mae']:.3f} RF2={validation[2]['mae']:.3f} "
        f"oracle={validation[3]['mae']:.3f} {time.time() - start:.1f}s",
        flush=True,
    )
    return validation, component


def diagnostic_summary(validation: pd.DataFrame) -> pd.DataFrame:
    pivot = validation.pivot(index=["property", "held_level"], columns="model", values="mae")
    rows = []
    for prop, group in pivot.groupby(level="property"):
        semi = group["semiparametric kernel"]
        rf = group["node random forest"]
        projected = group["RF second-order projection"]
        oracle = group["semiparametric + true held triple oracle"]
        advantage = semi - rf
        positive = advantage > 0
        rows.append({
            "property": prop,
            "n_positive_rf_gaps": int(positive.sum()),
            "semi_median_mae": float(semi.median()),
            "rf_median_mae": float(rf.median()),
            "projected_rf_median_mae": float(projected.median()),
            "triple_oracle_median_mae": float(oracle.median()),
            "median_projection_advantage_retained": float(
                ((semi[positive] - projected[positive]) / advantage[positive]).median()
            ) if positive.any() else float("nan"),
            "median_triple_gap_closed": float(
                ((semi[positive] - oracle[positive]) / advantage[positive]).median()
            ) if positive.any() else float("nan"),
            "median_rf_projection_cost": float((projected - rf).median()),
            "median_triple_oracle_gain": float((semi - oracle).median()),
        })
    return pd.DataFrame(rows)


def build_report(validation: pd.DataFrame, components: pd.DataFrame) -> str:
    summary = diagnostic_summary(validation)
    mse_columns = [
        "fibre_mean_mse", "a_conditional_mse", "x_conditional_mse", "ax_residual_mse"
    ]
    pooled = components.groupby("property")[mse_columns + ["total_gap_field_mse"]].sum()
    block_summary = pooled[mse_columns].div(pooled.total_gap_field_mse, axis=0)
    lines = [
        "# RF--semiparametric gap attribution\n",
        "Deterministic diagnostics computed after the results.\n",
        "## Oracle comparison\n",
        "| target | semi MAE | RF MAE | projected-RF MAE | true-triple oracle MAE | "
        "RF gaps > 0 | projected RF advantage retained | triple gap closed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.property.replace('_', ' ')} | {row.semi_median_mae:.4f} | "
            f"{row.rf_median_mae:.4f} | {row.projected_rf_median_mae:.4f} | "
            f"{row.triple_oracle_median_mae:.4f} | {row.n_positive_rf_gaps}/52 | "
            f"{row.median_projection_advantage_retained:.1%} | "
            f"{row.median_triple_gap_closed:.1%} |"
        )
    lines.extend([
        "\nFractions use only outer levels where RF beats the semiparametric model and are not "
        "clipped. `Projected RF advantage retained` measures how much of RF's level-wise MAE "
        "advantage survives removal of its predicted triple field. `Triple gap closed` is the "
        "unusable truth oracle's level-wise upper-bound diagnostic.\n",
        "## Random forest minus semiparametric model, left-out element\n",
        "Pooled share of squared node-prediction difference by orthogonal block, giving every "
        "outer level the same number of cells. The four entries sum to 100%:\n",
        "| target | element mean | A conditional | X conditional | A-by-X residual |",
        "|---|---:|---:|---:|---:|",
    ])
    for prop, row in block_summary.iterrows():
        lines.append(
            f"| {prop.replace('_', ' ')} | {row.fibre_mean_mse:.1%} | "
            f"{row.a_conditional_mse:.1%} | {row.x_conditional_mse:.1%} | "
            f"{row.ax_residual_mse:.1%} |"
        )
    lines.extend([
        "\n## As/Cs weighting and boundary check\n",
        "| target | level | balanced B edges | true triple mean | margin weighting error | "
        "semi margin | triple-oracle margin |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    selected = components[components.held_level.isin(["As", "Cs"])]
    for row in selected.itertuples(index=False):
        lines.append(
            f"| {row.property.replace('_', ' ')} | {row.held_level} | "
            f"{'yes' if row.b_edge_balanced else 'NO'} ({row.b_edge_count}/{row.b_edge_expected}) | "
            f"{row.true_triple_fibre_mean:+.2e} | {row.true_margin_weighting_error:+.2e} | "
            f"{row.semi_margin_from_cube:+.4f} | {row.triple_oracle_margin_from_cube:+.4f} |"
        )
    lines.extend([
        "\nA balanced held-B direction averages uniformly over A, X and partner B. Therefore a "
        "centred held triple field cannot change that direction; any As/Cs mean-direction flip "
        "that survives the numerical checks belongs to B-main transfer, not the omitted triple "
        "field.\n",
        "## Interpretation rule\n",
        "Read the table target by target. The projection and truth-oracle "
        "columns diagnose different restrictions and must not be described as deployable "
        "models.\n",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", help="comma-separated B levels; default all 52")
    parser.add_argument("--targets", help="comma-separated targets; default locked three")
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()

    frame = load_full_grid()
    desc = descriptors(frame)
    axes = cube_axes(frame)
    levels = axes[1]
    targets = list(TARGETS)
    if args.levels:
        requested = [value.strip() for value in args.levels.split(",") if value.strip()]
        unknown = sorted(set(requested) - set(levels))
        if unknown:
            raise ValueError(f"unknown B levels: {unknown}")
        levels = requested
    if args.targets:
        targets = [value.strip() for value in args.targets.split(",") if value.strip()]
        unknown = sorted(set(targets) - set(TARGETS))
        if unknown:
            raise ValueError(f"targets outside locked set: {unknown}")

    VALIDATION.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for prop in targets:
        truth_cube = values_to_cube(frame, frame[prop].to_numpy(float), axes)
        _, true_triple = project_cube_to_second_order(truth_cube)
        target_results = Parallel(n_jobs=args.n_jobs, prefer="threads", verbose=5)(
            delayed(run_outer)(frame, desc, prop, level, true_triple, axes)
            for level in levels
        )
        results.extend(target_results)
        pd.DataFrame([row for result in results for row in result[0]]).to_csv(
            VALIDATION, index=False)
        pd.DataFrame([result[1] for result in results]).to_csv(COMPONENTS, index=False)
        print(f"checkpointed {prop}", flush=True)

    validation = pd.DataFrame([row for result in results for row in result[0]])
    components = pd.DataFrame([result[1] for result in results])
    if set(levels) == set(axes[1]) and set(targets) == set(TARGETS):
        REPORT.write_text(build_report(validation, components))
        print(f"wrote {VALIDATION}, {COMPONENTS}, and {REPORT}")
    else:
        print(f"trial run only; wrote {VALIDATION} and {COMPONENTS}")


if __name__ == "__main__":
    main()
