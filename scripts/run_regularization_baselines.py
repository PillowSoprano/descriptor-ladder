"""The coverage sweep with matched regularisation.

Every arm uses the same outer designs, the same three training-only inner designs, nine candidate
penalties, and the error of substitution responses whose two compositions are both hidden.

    free_min_norm       untuned free element effects (minimum norm)
    ordinary_ridge      one tuned penalty for every free effect
    block_ridge         separate main-effect and pair-interaction penalties
    descriptor_metric   similarity prior from the physical descriptors
    shuffled_metric     the same descriptor rows reassigned among elements
    random_metric       a dimension-matched random geometry

The score on responses with at least one composition hidden is recorded as well; selection and the
comparison use the both-hidden score.

Writes ``reports/regularization_baselines.csv`` and ``reports/AUDIT_REGULARIZATION_BASELINES.md``.

Usage:
    python3 scripts/run_regularization_baselines.py
    python3 scripts/run_regularization_baselines.py --pilot     (one formation-energy design)
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import scipy.linalg
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prfield.descriptors import elemental as el  # noqa: E402
from prfield.descriptors.elemental import ElementalDescriptors  # noqa: E402
from prfield.graph.typed_graph import ComparabilityPolicy, accepted, build_edges  # noqa: E402
from prfield.models import hybrid  # noqa: E402
from prfield.schema import SITES  # noqa: E402
from prfield.splits import observation as obs  # noqa: E402


REPORT = Path("reports/AUDIT_REGULARIZATION_BASELINES.md")
CSV = Path("reports/regularization_baselines.csv")
PROPERTIES = ("heat_of_formation", "volume_per_fu", "magnetic_moment")
FRACS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90)
SEEDS = (0, 1, 2)
MAGNETIC_SEEDS = tuple(range(100, 110))
INNER_FOLDS = 3
INNER_TRAIN_FRACTION = 2.0 / 3.0
LAMBDA_GRID = tuple(float(v) for v in np.logspace(-6, 2, 9))
BLOCK_SCALES = (1e-4, 1e-1, 1e2)
BLOCK_RATIOS = (0.1, 1.0, 10.0)
NUMERICAL_RIDGE = 1e-8
CONTROL_SEED = 7919
POLICY = ComparabilityPolicy(
    max_volume_change_frac=None,
    max_rmsd_init_final=None,
    require_same_coordination=False,
)


@dataclass(frozen=True)
class Candidate:
    label: str
    main_penalty: float = 0.0
    interaction_penalty: float = 0.0
    metric_penalty: float = 0.0


def descriptor_controls(
    descriptors: Mapping[str, ElementalDescriptors],
) -> tuple[dict[str, ElementalDescriptors], dict[str, ElementalDescriptors]]:
    """Return frozen row-shuffled and dimension-matched random descriptor bundles."""
    shuffled: dict[str, ElementalDescriptors] = {}
    randomised: dict[str, ElementalDescriptors] = {}
    for offset, site in enumerate(SITES):
        frame = descriptors[site].frame.copy()
        rng = np.random.default_rng(CONTROL_SEED + offset)

        permuted = frame.to_numpy(float)[rng.permutation(len(frame))]
        shuffled[site] = ElementalDescriptors(
            site=site,
            frame=pd.DataFrame(permuted, index=frame.index, columns=frame.columns),
            kind="row-shuffled control",
        )

        coordinates = rng.normal(size=frame.shape)
        randomised[site] = ElementalDescriptors(
            site=site,
            frame=pd.DataFrame(coordinates, index=frame.index, columns=frame.columns),
            kind="dimension-matched random control",
        )
    return shuffled, randomised


def metric_matrix(
    descriptors: Mapping[str, ElementalDescriptors],
    levels: Mapping[str, list[str]],
    slices: Mapping[str, slice],
    n_columns: int,
) -> np.ndarray:
    """The exact descriptor Laplacian of the element-effect fit, separated from its scalar strength."""
    penalty = np.zeros((n_columns, n_columns), dtype=float)
    laplacians = {
        site: hybrid.level_laplacian(descriptors[site], levels[site], k=3)
        for site in SITES
    }
    for site in SITES:
        block = slices[site]
        penalty[block, block] = laplacians[site]
    for pair in hybrid.PAIRS:
        first, second = pair
        n_first, n_second = len(levels[first]), len(levels[second])
        block = slices[pair]
        penalty[block, block] = (
            np.kron(laplacians[first], np.eye(n_second))
            + np.kron(np.eye(n_first), laplacians[second])
        )
    return penalty


def diagonal_penalty(
    slices: Mapping[str, slice],
    n_columns: int,
    main_penalty: float,
    interaction_penalty: float,
) -> np.ndarray:
    penalty = np.zeros(n_columns, dtype=float)
    for site in SITES:
        penalty[slices[site]] = main_penalty
    for pair in hybrid.PAIRS:
        penalty[slices[pair]] = interaction_penalty
    return penalty


def normal_equations(
    train: pd.DataFrame,
    prop: str,
    levels: Mapping[str, list[str]],
) -> tuple[np.ndarray, np.ndarray, Mapping[str, slice]]:
    work = train.dropna(subset=[prop]).reset_index(drop=True)
    design, slices = hybrid._design_and_slices(work, dict(levels))
    constraints = hybrid._constraints(dict(levels), slices, design.shape[1])
    response = work[prop].to_numpy(float)
    weight = np.sqrt(max(len(work), 1)) * 10.0
    gram = (design.T @ design).toarray() + (weight**2) * (constraints.T @ constraints).toarray()
    rhs = np.asarray(design.T @ response, float).ravel()
    return gram, rhs, slices


def solve(
    gram: np.ndarray,
    rhs: np.ndarray,
    diagonal: np.ndarray,
    metric: np.ndarray | None = None,
    metric_strength: float = 0.0,
) -> np.ndarray:
    system = gram.copy()
    index = np.diag_indices_from(system)
    system[index] += diagonal + NUMERICAL_RIDGE
    if metric is not None and metric_strength:
        system += metric_strength * metric
    try:
        factor = scipy.linalg.cho_factor(system, lower=True, check_finite=False)
        return scipy.linalg.cho_solve(factor, rhs, check_finite=False)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(system, rhs, rcond=None)[0]


def edge_subset(edges: pd.DataFrame, hidden_ids: Iterable[str], both_hidden: bool) -> pd.DataFrame:
    hidden = frozenset(hidden_ids)
    src = edges["src"].isin(hidden)
    dst = edges["dst"].isin(hidden)
    return edges[src & dst] if both_hidden else edges[src | dst]


def edge_mae(
    beta: np.ndarray,
    levels: Mapping[str, list[str]],
    node_index: pd.DataFrame,
    edges: pd.DataFrame,
) -> float:
    if edges.empty:
        return float("nan")
    source = node_index.loc[edges["src"]].reset_index()
    target = node_index.loc[edges["dst"]].reset_index()
    source_design, _ = hybrid._design_and_slices(source, dict(levels))
    target_design, _ = hybrid._design_and_slices(target, dict(levels))
    prediction = np.asarray((target_design - source_design) @ beta, float).ravel()
    truth = edges["delta"].to_numpy(float)
    return float(np.mean(np.abs(truth - prediction)))


def candidates(model: str) -> tuple[Candidate, ...]:
    if model in {"ordinary_ridge", "descriptor_metric", "shuffled_metric", "random_metric"}:
        return tuple(Candidate(f"lambda={value:g}", main_penalty=value,
                               interaction_penalty=value, metric_penalty=value)
                     for value in LAMBDA_GRID)
    if model == "block_ridge":
        output = []
        for scale in BLOCK_SCALES:
            for ratio in BLOCK_RATIOS:
                root = np.sqrt(ratio)
                output.append(Candidate(
                    f"scale={scale:g};ratio={ratio:g}",
                    main_penalty=scale / root,
                    interaction_penalty=scale * root,
                ))
        return tuple(output)
    if model == "free_min_norm":
        return (Candidate("minimum-norm"),)
    raise KeyError(model)


def fit_candidate(
    train: pd.DataFrame,
    prop: str,
    levels: Mapping[str, list[str]],
    model: str,
    candidate: Candidate,
    metric_matrices: Mapping[str, np.ndarray],
) -> np.ndarray:
    gram, rhs, slices = normal_equations(train, prop, levels)
    n_columns = gram.shape[0]
    if model == "free_min_norm":
        diagonal = diagonal_penalty(slices, n_columns, 0.0, 0.0)
        return solve(gram, rhs, diagonal)
    if model in {"ordinary_ridge", "block_ridge"}:
        diagonal = diagonal_penalty(
            slices, n_columns, candidate.main_penalty, candidate.interaction_penalty
        )
        return solve(gram, rhs, diagonal)
    diagonal = diagonal_penalty(slices, n_columns, 0.0, 0.0)
    return solve(
        gram,
        rhs,
        diagonal,
        metric=metric_matrices[model],
        metric_strength=candidate.metric_penalty,
    )


def select_candidate(
    outer_train: pd.DataFrame,
    prop: str,
    levels: Mapping[str, list[str]],
    all_edges: pd.DataFrame,
    node_index: pd.DataFrame,
    model: str,
    metric_matrices: Mapping[str, np.ndarray],
    outer_seed: int,
) -> tuple[Candidate, float, int]:
    options = candidates(model)
    if len(options) == 1:
        return options[0], float("nan"), 0

    folds = []
    for fold in range(INNER_FOLDS):
        split = obs.random_nodes(
            outer_train,
            INNER_TRAIN_FRACTION,
            seed=100_000 + 100 * outer_seed + fold,
        )
        inner_train = split.train_nodes(outer_train)
        validation = edge_subset(all_edges, split.held_out_ids, both_hidden=True)
        validation = validation[
            validation["src"].isin(set(outer_train.sample_id))
            & validation["dst"].isin(set(outer_train.sample_id))
        ]
        if not validation.empty:
            folds.append((inner_train, validation))
    if not folds:
        raise RuntimeError("no two-hidden-endpoint inner validation edges")

    losses = []
    for option in options:
        fold_losses = []
        for inner_train, validation in folds:
            beta = fit_candidate(
                inner_train, prop, levels, model, option, metric_matrices
            )
            fold_losses.append(edge_mae(beta, levels, node_index, validation))
        losses.append(float(np.nanmean(fold_losses)))
    best = min(range(len(options)), key=lambda index: (losses[index], index))
    return options[best], losses[best], len(folds)


def evaluate_outer(
    sub: pd.DataFrame,
    prop: str,
    frac: float,
    seed: int,
    levels: Mapping[str, list[str]],
    all_edges: pd.DataFrame,
    node_index: pd.DataFrame,
    metric_matrices: Mapping[str, np.ndarray],
    model_names: tuple[str, ...] = (
        "free_min_norm",
        "ordinary_ridge",
        "block_ridge",
        "descriptor_metric",
        "shuffled_metric",
        "random_metric",
    ),
) -> list[dict]:
    split = obs.random_nodes(sub, frac, seed=seed)
    train = split.train_nodes(sub).dropna(subset=[prop]).reset_index(drop=True)
    primary_edges = edge_subset(all_edges, split.held_out_ids, both_hidden=True)
    historical_edges = edge_subset(all_edges, split.held_out_ids, both_hidden=False)
    rows = []
    for model in model_names:
        option, inner_mae, n_inner = select_candidate(
            train, prop, levels, all_edges, node_index, model, metric_matrices, seed
        )
        beta = fit_candidate(train, prop, levels, model, option, metric_matrices)
        rows.append({
            "property": prop,
            "coverage": frac,
            "n_over_p": len(train) / 358.0,
            "seed": seed,
            "model": model,
            "selected": option.label,
            "inner_mae": inner_mae,
            "n_inner_folds": n_inner,
            "n_train": len(train),
            "n_test_both_hidden": len(primary_edges),
            "mae_both_hidden": edge_mae(beta, levels, node_index, primary_edges),
            "n_test_touching": len(historical_edges),
            "mae_touching": edge_mae(beta, levels, node_index, historical_edges),
        })
    return rows


def spread_levels(labels: list[str], count: int) -> list[str]:
    """Deterministic farthest-point panel in standardised elemental-descriptor space."""
    bundle = el.physical_descriptors(labels, "element")
    matrix = np.nan_to_num(np.asarray(bundle.matrix(labels, allow_missing=True), float))
    z = (matrix - matrix.mean(0)) / np.where(matrix.std(0) > 0, matrix.std(0), 1.0)
    chosen = [int(np.argmin(((z - z.mean(0)) ** 2).sum(1)))]
    while len(chosen) < count:
        distance = np.min(((z[:, None, :] - z[None, chosen, :]) ** 2).sum(-1), axis=1)
        distance[chosen] = -1.0
        chosen.append(int(np.argmax(distance)))
    return sorted(labels[index] for index in chosen)


def panel_definitions(frame: pd.DataFrame) -> list[tuple[str, list[str], list[str]]]:
    all_a = sorted(frame.site_A.unique().tolist())
    all_b = sorted(frame.site_B.unique().tolist())
    panels = [("alphabetical", all_a[:14], all_b[:14])]
    for seed in range(5):
        rng = np.random.default_rng(seed)
        panels.append((
            f"random-{seed}",
            sorted(rng.choice(all_a, 14, replace=False).tolist()),
            sorted(rng.choice(all_b, 14, replace=False).tolist()),
        ))
    panels.append(("descriptor-maximin", spread_levels(all_a, 14), spread_levels(all_b, 14)))
    return panels


def run_panel_curves(frame: pd.DataFrame, n_jobs: int) -> None:
    """Seven-panel replication of the chemically structured gain curve."""
    output_rows = []
    start = time.time()
    for panel, a_levels, b_levels in panel_definitions(frame):
        sub = frame[
            frame.site_A.isin(a_levels) & frame.site_B.isin(b_levels)
        ].reset_index(drop=True)
        levels = {site: sorted(sub[f"site_{site}"].unique().tolist()) for site in SITES}
        descriptors = {
            site: el.physical_descriptors(
                levels[site], "element" if site in ("A", "B") else "auto"
            )
            for site in SITES
        }
        template, slices = hybrid._design_and_slices(sub.iloc[:1], levels)
        metric_matrices = {
            "descriptor_metric": metric_matrix(
                descriptors, levels, slices, template.shape[1]
            )
        }
        node_index = sub.set_index("sample_id")
        edges_by_property = {
            prop: accepted(build_edges(sub, prop, POLICY)).reset_index(drop=True)
            for prop in PROPERTIES
        }
        tasks = [
            (prop, frac, seed)
            for prop in PROPERTIES
            for frac in FRACS
            for seed in SEEDS
        ]
        panel_output = Parallel(n_jobs=n_jobs, verbose=5, prefer="threads")(
            delayed(evaluate_outer)(
                sub,
                prop,
                frac,
                seed,
                levels,
                edges_by_property[prop],
                node_index,
                metric_matrices,
                ("free_min_norm", "ordinary_ridge", "descriptor_metric"),
            )
            for prop, frac, seed in tasks
        )
        for group in panel_output:
            for row in group:
                row["panel"] = panel
                output_rows.append(row)
        print(f"panel={panel} complete; elapsed={time.time() - start:.1f}s", flush=True)

    result = pd.DataFrame(output_rows)
    output_csv = Path("reports/regularization_panel_curves.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)

    pivot = result.pivot_table(
        index=["panel", "property", "coverage", "seed"],
        columns="model",
        values="mae_both_hidden",
    ).reset_index()
    pivot["descriptor_vs_ridge_pct"] = 100.0 * (
        pivot["ordinary_ridge"] - pivot["descriptor_metric"]
    ) / pivot["ordinary_ridge"]
    summary = (
        pivot.groupby(["panel", "property", "coverage"], as_index=False)
        .descriptor_vs_ridge_pct.median()
    )
    report = [
        "# Seven-panel coverage curves against ordinary ridge",
        "",
        "Each panel uses the same coverage fractions, three masks and nested penalty budget. "
        "The reported value is the median paired descriptor-metric gain over ordinary ridge.",
        "",
        "| property | coverage | panel median | panel minimum | panel maximum | positive panels |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for (prop, frac), group in summary.groupby(["property", "coverage"], sort=False):
        values = group.descriptor_vs_ridge_pct
        report.append(
            f"| {prop.replace('_', ' ')} | {frac:.0%} | {values.median():+.2f}% | "
            f"{values.min():+.2f}% | {values.max():+.2f}% | "
            f"{int((values > 0).sum())}/{len(values)} |"
        )
    report.extend(["", f"Runtime: {(time.time() - start) / 60.0:.1f} minutes."])
    Path("reports/AUDIT_REGULARIZATION_PANELS.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(f"wrote {output_csv} and reports/AUDIT_REGULARIZATION_PANELS.md")


def write_report(results: pd.DataFrame, elapsed: float) -> None:
    pivot = results.pivot_table(
        index=["property", "coverage", "seed"],
        columns="model",
        values="mae_both_hidden",
    ).reset_index()
    pivot["descriptor_vs_ridge_pct"] = 100.0 * (
        pivot["ordinary_ridge"] - pivot["descriptor_metric"]
    ) / pivot["ordinary_ridge"]
    pivot["descriptor_vs_block_pct"] = 100.0 * (
        pivot["block_ridge"] - pivot["descriptor_metric"]
    ) / pivot["block_ridge"]

    lines = ["# Matched regularisation baselines for the coverage claim", ""]
    lines.append(
        "The primary score uses only substitution edges whose two endpoint compositions were "
        "hidden. Every tuned arm receives three identical training-only inner masks and nine "
        "candidate configurations. One outer mask is the independent paired unit."
    )
    lines.append("")
    lines.append("## Median two-hidden-endpoint MAE and chemically structured gain")
    lines.append("")
    lines.append(
        "| property | coverage | free | ordinary ridge | block ridge | descriptor metric | "
        "shuffled metric | random metric | descriptor vs ordinary | descriptor vs block |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (prop, frac), group in results.groupby(["property", "coverage"], sort=False):
        med = group.groupby("model")["mae_both_hidden"].median()
        ordinary_gain = 100.0 * (med["ordinary_ridge"] - med["descriptor_metric"]) / med["ordinary_ridge"]
        block_gain = 100.0 * (med["block_ridge"] - med["descriptor_metric"]) / med["block_ridge"]
        lines.append(
            f"| {prop.replace('_', ' ')} | {frac:.0%} | {med['free_min_norm']:.4f} | "
            f"{med['ordinary_ridge']:.4f} | {med['block_ridge']:.4f} | "
            f"{med['descriptor_metric']:.4f} | {med['shuffled_metric']:.4f} | "
            f"{med['random_metric']:.4f} | {ordinary_gain:+.1f}% | {block_gain:+.1f}% |"
        )

    lines.extend(["", "## Paired summary", ""])
    lines.append(
        "| property | median descriptor-vs-ordinary gain | wins | "
        "median descriptor-vs-block gain | wins |"
    )
    lines.append("|---|---:|---:|---:|---:|")
    for prop, group in pivot.groupby("property", sort=False):
        d1 = group["descriptor_vs_ridge_pct"]
        d2 = group["descriptor_vs_block_pct"]
        lines.append(
            f"| {prop.replace('_', ' ')} | {d1.median():+.2f}% | "
            f"{int((d1 > 0).sum())}/{len(d1)} | {d2.median():+.2f}% | "
            f"{int((d2 > 0).sum())}/{len(d2)} |"
        )

    lines.extend(["", "## Endpoint-estimand audit", ""])
    joined = results.pivot_table(
        index=["property", "coverage", "seed", "model"],
        values=["mae_both_hidden", "mae_touching"],
    ).reset_index()
    joined["relative_change_pct"] = 100.0 * (
        joined["mae_both_hidden"] / joined["mae_touching"] - 1.0
    )
    lines.append(
        "The median change from the historical touching-edge score to the primary "
        "two-hidden-endpoint score is "
        f"{joined['relative_change_pct'].median():+.1f}%."
    )
    lines.append("")
    lines.append(f"Runtime: {elapsed / 60.0:.1f} minutes.")

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="one formation-energy outer mask")
    parser.add_argument("--panels", action="store_true", help="run the seven-panel coverage replication")
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()

    frame = pd.read_parquet("data/interim/castelli_raw.parquet").copy()
    if "magnetic_moment" not in frame or frame["magnetic_moment"].isna().all():
        source = "_magmom_signed" if "_magmom_signed" in frame else "_magmom"
        if source not in frame:
            raise KeyError("cached Castelli frame has no magnetic-moment source column")
        frame["magnetic_moment"] = frame[source].abs()
    if args.panels:
        run_panel_curves(frame, args.n_jobs)
        return
    a_levels = sorted(frame.site_A.unique())[:14]
    b_levels = sorted(frame.site_B.unique())[:14]
    sub = frame[
        frame.site_A.isin(a_levels) & frame.site_B.isin(b_levels)
    ].reset_index(drop=True)
    levels = {site: sorted(sub[f"site_{site}"].unique().tolist()) for site in SITES}
    descriptors = {
        site: el.physical_descriptors(
            levels[site], "element" if site in ("A", "B") else "auto"
        )
        for site in SITES
    }
    shuffled, randomised = descriptor_controls(descriptors)

    template, slices = hybrid._design_and_slices(sub.iloc[:1], levels)
    metric_matrices = {
        "descriptor_metric": metric_matrix(descriptors, levels, slices, template.shape[1]),
        "shuffled_metric": metric_matrix(shuffled, levels, slices, template.shape[1]),
        "random_metric": metric_matrix(randomised, levels, slices, template.shape[1]),
    }
    node_index = sub.set_index("sample_id")

    properties = ("heat_of_formation",) if args.pilot else PROPERTIES
    fractions = (0.20,) if args.pilot else FRACS
    seeds = (0,) if args.pilot else SEEDS
    edges_by_property = {
        prop: accepted(build_edges(sub, prop, POLICY)).reset_index(drop=True)
        for prop in properties
    }
    tasks = []
    for prop in properties:
        property_seeds = MAGNETIC_SEEDS if prop == "magnetic_moment" and not args.pilot else seeds
        tasks.extend((prop, frac, seed) for frac in fractions for seed in property_seeds)

    start = time.time()
    output = Parallel(n_jobs=args.n_jobs, verbose=10, prefer="threads")(
        delayed(evaluate_outer)(
            sub,
            prop,
            frac,
            seed,
            levels,
            edges_by_property[prop],
            node_index,
            metric_matrices,
        )
        for prop, frac, seed in tasks
    )
    results = pd.DataFrame([row for group in output for row in group])
    CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(CSV, index=False)
    write_report(results, time.time() - start)
    print(f"wrote {CSV} and {REPORT}")


if __name__ == "__main__":
    main()
