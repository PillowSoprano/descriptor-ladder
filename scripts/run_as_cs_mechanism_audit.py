"""Where the errors for left-out As and Cs come from.

Five questions:

1. Is the error localised to A-, B- or X-site substitutions, or to particular anion contexts?
2. Does the magnetic result hold when the moment magnitude is replaced by the signed value?
3. Do formation-energy and magnetic sign errors occur on the same substitution pairs?
4. Is the formation-energy displacement carried by the varied B descriptor or by the A/X context?
5. Which correlated descriptor families are needed for the As/Cs contrast?

The signed moment is a diagnostic control, not a physical target: reversing the global collinear
spin axis changes its sign without changing the magnetic state.

Writes eight CSVs and reports/AUDIT_AS_CS_MECHANISM.md.

Usage: PYTHONPATH=src python3 scripts/run_as_cs_mechanism_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prfield.descriptors import elemental as el  # noqa: E402
from prfield.graph.typed_graph import LOOSE, build_edges  # noqa: E402
from prfield.models import hybrid  # noqa: E402
from prfield.models.baselines import DifferenceFromNode, DirectResponseModel, NodeModel  # noqa: E402
from prfield.report import write_report  # noqa: E402
from prfield.splits import observation as obs  # noqa: E402

REPORT = Path("reports/AUDIT_AS_CS_MECHANISM.md")
SUMMARY = Path("reports/as_cs_edge_decomposition.csv")
CONTEXT = Path("reports/as_cs_context_decomposition.csv")
CROSS = Path("reports/as_cs_cross_target.csv")
MAGNETIC = Path("reports/as_cs_magnetic_folding.csv")
PARTNER = Path("reports/as_cs_partner_decomposition.csv")
MODEL_CLASS = Path("reports/as_cs_model_class_control.csv")
COMPONENTS = Path("reports/as_cs_direct_components.csv")
ABLATION = Path("reports/as_cs_descriptor_ablation.csv")

TARGETS = ("As", "Cs")
PROPS = (
    "heat_of_formation",
    "volume_per_fu",
    "magnetic_moment",
    "magnetic_moment_signed",
)
SMOOTH = 0.1
MOMENT_ZERO = 0.05

FEATURE_FAMILIES = {
    "periodic identity": {"atomic_number", "atomic_mass", "row", "group"},
    "size": {"ionic_radius", "atomic_radius"},
    "electron energetics": {"electronegativity", "ionization_energy", "electron_affinity"},
    "valence": {"valence_electrons"},
}


def sign_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    resolvable = np.abs(y) > 0
    if not resolvable.any():
        return float("nan")
    return float(np.mean(np.sign(y[resolvable]) == np.sign(pred[resolvable])))


def load_subgrid() -> pd.DataFrame:
    df = pd.read_parquet("data/interim/castelli_raw.parquet").copy()
    df["magnetic_moment_signed"] = df["_magmom"].astype(float)
    df["magnetic_moment"] = df["_magmom"].abs().astype(float)
    a = sorted(df.site_A.unique())[:14]
    b = sorted(df.site_B.unique())[:14]
    return df[df.site_A.isin(a) & df.site_B.isin(b)].reset_index(drop=True)


def descriptors(df: pd.DataFrame):
    return {
        site: el.physical_descriptors(
            sorted(df[f"site_{site}"].unique()), "element" if site in ("A", "B") else "auto"
        )
        for site in ("A", "B", "X")
    }


def descriptor_features(df: pd.DataFrame, desc, scalers=None):
    blocks, fitted = [], {} if scalers is None else scalers
    for site in ("A", "B", "X"):
        matrix = np.nan_to_num(np.asarray(desc[site].matrix(
            df[f"site_{site}"].tolist(), allow_missing=True), float))
        if scalers is None:
            mean, scale = matrix.mean(axis=0), matrix.std(axis=0)
            fitted[site] = (mean, np.where(scale > 0, scale, 1.0))
        mean, scale = fitted[site]
        blocks.append((matrix - mean) / scale)
    return np.hstack(blocks), fitted


def context_label(row: pd.Series) -> str:
    if row["site"] in ("A", "B"):
        return str(row["site_X"])
    return f"{row['src_species']}->{row['dst_species']}"


def fit_predictions(df: pd.DataFrame, desc, prop: str, level: str) -> pd.DataFrame:
    edges = build_edges(df, prop, LOOSE)
    design = obs.held_out_fiber(df, "B", level)
    induced = design.induced_edges(edges)
    test = induced.test.copy().reset_index(drop=True)
    train = design.train_nodes(df).dropna(subset=[prop]).reset_index(drop=True)

    direct = DirectResponseModel(desc, prop).fit(induced.train)
    test["direct"] = direct.predict_delta(test)

    effects = hybrid.fit_stage1(train, prop, smooth=SMOOTH, desc=desc)
    maps, normalisation = hybrid.fit_stage2(effects, desc)
    metric = hybrid.HybridModel(effects, maps, normalisation, desc)
    index = df.set_index("sample_id")
    src = index.loc[test.src].reset_index()
    dst = index.loc[test.dst].reset_index()
    test["metric"] = metric.predict(dst) - metric.predict(src)
    test["context"] = test.apply(context_label, axis=1)
    test["partner"] = pd.NA
    is_b = test.site.eq("B")
    test.loc[is_b, "partner"] = np.where(
        test.loc[is_b, "src_species"].eq(level),
        test.loc[is_b, "dst_species"],
        test.loc[is_b, "src_species"],
    )

    raw_src = index.loc[test.src, "magnetic_moment_signed"].to_numpy(float)
    raw_dst = index.loc[test.dst, "magnetic_moment_signed"].to_numpy(float)
    near = (np.abs(raw_src) <= MOMENT_ZERO) | (np.abs(raw_dst) <= MOMENT_ZERO)
    cross = (np.sign(raw_src) != np.sign(raw_dst)) & ~near
    test["magnetic_edge_class"] = np.where(near, "near-zero endpoint",
                                            np.where(cross, "opposite spin axes", "same spin axis"))
    test["property"] = prop
    test["level"] = level
    return test


def summarise_edges(predictions: pd.DataFrame) -> pd.DataFrame:
    nonzero = predictions.loc[predictions.delta.abs() > 0, "delta"].abs()
    threshold = float(nonzero.quantile(0.10)) if len(nonzero) else 0.0
    rows = []
    for (prop, level, site), group in predictions.groupby(["property", "level", "site"]):
        for model in ("direct", "metric"):
            y = group.delta.to_numpy(float)
            pred = group[model].to_numpy(float)
            error = pred - y
            rows.append({
                "property": prop,
                "level": level,
                "site": site,
                "model": model,
                "n_edges": len(group),
                "n_signed": int(np.sum(np.abs(y) > 0)),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error ** 2))),
                "sign_accuracy": sign_accuracy(y, pred),
                "true_positive_fraction": float(np.mean(y > 0)),
                "pred_positive_fraction": float(np.mean(pred > 0)),
                "error_mean": float(np.mean(error)),
                "error_sd": float(np.std(error)),
                "near_zero_threshold": threshold,
                "near_zero_fraction": float(np.mean(np.abs(y) <= threshold)),
            })
    return pd.DataFrame(rows)


def summarise_context(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["property", "level", "site", "context"]
    for key, group in predictions.groupby(keys, dropna=False):
        if len(group) < 14:
            continue
        for model in ("direct", "metric"):
            y = group.delta.to_numpy(float)
            pred = group[model].to_numpy(float)
            rows.append(dict(zip(keys, key)) | {
                "model": model,
                "n_edges": len(group),
                "n_signed": int(np.sum(np.abs(y) > 0)),
                "mae": float(np.mean(np.abs(pred - y))),
                "sign_accuracy": sign_accuracy(y, pred),
                "median_abs_response": float(np.median(np.abs(y))),
            })
    return pd.DataFrame(rows)


def cross_target(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pairs = (
        ("heat_of_formation", "magnetic_moment"),
        ("heat_of_formation", "magnetic_moment_signed"),
        ("magnetic_moment", "magnetic_moment_signed"),
    )
    for level in TARGETS:
        for model in ("direct", "metric"):
            d = predictions[predictions.level == level]
            for left, right in pairs:
                a = d[d.property == left][["edge_id", "site", "delta", model]].copy()
                b = d[d.property == right][["edge_id", "delta", model]].copy()
                m = a.merge(b, on="edge_id", suffixes=("_left", "_right"))
                m = m[(m.delta_left.abs() > 0) & (m.delta_right.abs() > 0)].reset_index(drop=True)
                fail_left = np.sign(m.delta_left) != np.sign(m[f"{model}_left"])
                fail_right = np.sign(m.delta_right) != np.sign(m[f"{model}_right"])
                union = fail_left | fail_right
                jaccard = float(np.sum(fail_left & fail_right) / np.sum(union)) if union.any() else np.nan
                error_left = np.abs(m[f"{model}_left"] - m.delta_left)
                error_right = np.abs(m[f"{model}_right"] - m.delta_right)
                rho, p = stats.spearmanr(error_left, error_right)
                rows.append({
                    "level": level,
                    "model": model,
                    "left_target": left,
                    "right_target": right,
                    "n_edges": len(m),
                    "left_failure_fraction": float(fail_left.mean()),
                    "right_failure_fraction": float(fail_right.mean()),
                    "failure_jaccard": jaccard,
                    "absolute_error_spearman": float(rho),
                    "absolute_error_p": float(p),
                })
    return pd.DataFrame(rows)


def magnetic_folding(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    d = predictions[predictions.property.isin(["magnetic_moment", "magnetic_moment_signed"])]
    for key, group in d.groupby(["property", "level", "site", "magnetic_edge_class"]):
        for model in ("direct", "metric"):
            y = group.delta.to_numpy(float)
            pred = group[model].to_numpy(float)
            rows.append({
                "property": key[0],
                "level": key[1],
                "site": key[2],
                "edge_class": key[3],
                "model": model,
                "n_edges": len(group),
                "n_signed": int(np.sum(np.abs(y) > 0)),
                "mae": float(np.mean(np.abs(pred - y))),
                "sign_accuracy": sign_accuracy(y, pred),
            })
    return pd.DataFrame(rows)


def partner_decomposition(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    d = predictions[predictions.site == "B"]
    for key, group in d.groupby(["property", "level", "partner"]):
        for model in ("direct", "metric"):
            y = group.delta.to_numpy(float)
            pred = group[model].to_numpy(float)
            rows.append({
                "property": key[0],
                "level": key[1],
                "partner": key[2],
                "model": model,
                "n_edges": len(group),
                "n_signed": int(np.sum(np.abs(y) > 0)),
                "mae": float(np.mean(np.abs(pred - y))),
                "sign_accuracy": sign_accuracy(y, pred),
                "true_delta_mean": float(np.mean(y)),
                "pred_delta_mean": float(np.mean(pred)),
                "true_positive_fraction": float(np.mean(y > 0)),
                "pred_positive_fraction": float(np.mean(pred > 0)),
            })
    return pd.DataFrame(rows)


def model_class_control(df: pd.DataFrame, desc, predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    index = df.set_index("sample_id")
    for prop in PROPS:
        edges = build_edges(df, prop, LOOSE)
        for level in TARGETS:
            design = obs.held_out_fiber(df, "B", level)
            train = design.train_nodes(df).dropna(subset=[prop]).reset_index(drop=True)
            test = design.induced_edges(edges).test.reset_index(drop=True)
            src = index.loc[test.src].reset_index()
            dst = index.loc[test.dst].reset_index()
            y = test.delta.to_numpy(float)

            x_train, fitted = descriptor_features(train, desc)
            x_src, _ = descriptor_features(src, desc, fitted)
            x_dst, _ = descriptor_features(dst, desc, fitted)
            y_train = train[prop].to_numpy(float)
            rf = RandomForestRegressor(n_estimators=400, random_state=0, n_jobs=-1).fit(
                x_train, y_train)
            hgb = HistGradientBoostingRegressor(random_state=0).fit(x_train, y_train)
            linear = DifferenceFromNode(NodeModel(desc, prop, interactions=True).fit(train))
            candidates = {
                "node linear": linear.predict_delta(test),
                "node random forest": rf.predict(x_dst) - rf.predict(x_src),
                "node gradient boosting": hgb.predict(x_dst) - hgb.predict(x_src),
            }
            existing = predictions[(predictions.property == prop) &
                                   (predictions.level == level)].set_index("edge_id")
            candidates["direct linear response"] = existing.reindex(test.edge_id).direct.to_numpy(float)
            candidates["free plus metric"] = existing.reindex(test.edge_id).metric.to_numpy(float)
            for model, pred in candidates.items():
                rows.append({
                    "property": prop,
                    "level": level,
                    "model": model,
                    "n_edges": len(test),
                    "mae": float(np.mean(np.abs(pred - y))),
                    "sign_accuracy": sign_accuracy(y, pred),
                })
    return pd.DataFrame(rows)


def _base_feature(column: str) -> str:
    for prefix in ("mean_", "sum_"):
        if column.startswith(prefix):
            return column[len(prefix):]
    return column


def drop_descriptor_family(desc, family: set[str], sites: set[str]):
    """Return descriptor bundles with one correlated feature family removed at chosen sites."""
    out = {}
    for site, bundle in desc.items():
        remove = [c for c in bundle.frame.columns
                  if site in sites and _base_feature(c) in family]
        out[site] = el.ElementalDescriptors(
            site=bundle.site,
            frame=bundle.frame.drop(columns=remove),
            kind=bundle.kind,
            notes=list(bundle.notes) + ([f"ablation removed: {', '.join(remove)}"] if remove else []),
        )
    return out


def direct_component_audit(df: pd.DataFrame, desc) -> pd.DataFrame:
    """Attribute held-minus-partner B predictions to fitted blocks and B features.

    Feature-level rows are descriptive sensitivity attributions. The descriptors are
    correlated, so their coefficients must not be read as independent chemical causes.
    """
    rows = []
    edges = build_edges(df, "heat_of_formation", LOOSE)
    for level in TARGETS:
        design = obs.held_out_fiber(df, "B", level)
        induced = design.induced_edges(edges)
        test = induced.test[induced.test.site == "B"].reset_index(drop=True)
        model = DirectResponseModel(desc, "heat_of_formation").fit(induced.train)
        X = model._design_for_site("B", test)
        beta = model.coef_["B"]
        orientation = np.where(test.dst_species.eq(level), 1.0, -1.0)
        prediction = X @ beta
        true_held_minus_partner = orientation * test.delta.to_numpy(float)

        for name, sl in model.slices_["B"].items():
            label = {"M_A": "context A", "M_X": "context X"}.get(name, name)
            contribution = X[:, sl] @ beta[sl]
            rows.append({
                "level": level,
                "resolution": "block",
                "component": label,
                "mean_held_minus_partner": float(np.mean(orientation * contribution)),
                "mean_abs_contribution": float(np.mean(np.abs(contribution))),
            })

        # Sum each B feature's main and context-coupled terms. Feature rows plus the
        # intercept row therefore reproduce the fitted prediction exactly.
        diff = X[:, model.slices_["B"]["theta"]]
        theta = model.blocks_["B"].theta
        contexts = {
            context_site: model.scalers_[context_site].transform(
                desc[context_site].matrix(test[f"site_{context_site}"].tolist())
            )
            for context_site in model.blocks_["B"].context
        }
        for j, feature in enumerate(desc["B"].feature_names):
            contribution = diff[:, j] * theta[j]
            for context_site, matrix in model.blocks_["B"].context.items():
                contribution = contribution + diff[:, j] * (contexts[context_site] @ matrix[:, j])
            rows.append({
                "level": level,
                "resolution": "B feature",
                "component": feature,
                "mean_held_minus_partner": float(np.mean(orientation * contribution)),
                "mean_abs_contribution": float(np.mean(np.abs(contribution))),
            })

        rows.extend([
            {
                "level": level,
                "resolution": "total",
                "component": "true response",
                "mean_held_minus_partner": float(np.mean(true_held_minus_partner)),
                "mean_abs_contribution": float(np.mean(np.abs(test.delta.to_numpy(float)))),
            },
            {
                "level": level,
                "resolution": "total",
                "component": "predicted response",
                "mean_held_minus_partner": float(np.mean(orientation * prediction)),
                "mean_abs_contribution": float(np.mean(np.abs(prediction))),
            },
        ])
    return pd.DataFrame(rows)


def descriptor_ablation_audit(df: pd.DataFrame, desc) -> pd.DataFrame:
    """Remove descriptor families from the varied site or contexts and refit from scratch."""
    rows = []
    edges = build_edges(df, "heat_of_formation", LOOSE)
    configs = [("full", desc, True, True),
               ("no context", desc, False, True),
               ("no intercept", desc, True, False)]
    for family_name, family in FEATURE_FAMILIES.items():
        configs.append((f"drop {family_name} from B", drop_descriptor_family(desc, family, {"B"}),
                        True, True))
        configs.append((f"drop {family_name} from A/X context",
                        drop_descriptor_family(desc, family, {"A", "X"}), True, True))

    for level in TARGETS:
        design = obs.held_out_fiber(df, "B", level)
        induced = design.induced_edges(edges)
        test = induced.test[induced.test.site == "B"].reset_index(drop=True)
        y = test.delta.to_numpy(float)
        orientation = np.where(test.dst_species.eq(level), 1.0, -1.0)
        for name, candidate_desc, use_context, fit_intercept in configs:
            model = DirectResponseModel(
                candidate_desc, "heat_of_formation", use_context=use_context,
                fit_intercept=fit_intercept,
            ).fit(induced.train)
            pred = model.predict_delta(test)
            rows.append({
                "level": level,
                "configuration": name,
                "n_edges": len(test),
                "mae": float(np.mean(np.abs(pred - y))),
                "sign_accuracy": sign_accuracy(y, pred),
                "mean_true_held_minus_partner": float(np.mean(orientation * y)),
                "mean_pred_held_minus_partner": float(np.mean(orientation * pred)),
                "mean_bias": float(np.mean(orientation * (pred - y))),
            })
    return pd.DataFrame(rows)


def main() -> None:
    df = load_subgrid()
    desc = descriptors(df)
    frames = []
    for prop in PROPS:
        for level in TARGETS:
            frames.append(fit_predictions(df, desc, prop, level))
            print(f"{prop:24s} B={level} complete", flush=True)
    predictions = pd.concat(frames, ignore_index=True)

    summary = summarise_edges(predictions)
    context = summarise_context(predictions)
    cross = cross_target(predictions)
    magnetic = magnetic_folding(predictions)
    partner = partner_decomposition(predictions)
    classes = model_class_control(df, desc, predictions)
    components = direct_component_audit(df, desc)
    ablation = descriptor_ablation_audit(df, desc)
    for path, frame in ((SUMMARY, summary), (CONTEXT, context), (CROSS, cross),
                        (MAGNETIC, magnetic), (PARTNER, partner), (MODEL_CLASS, classes),
                        (COMPONENTS, components), (ABLATION, ablation)):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    lines = ["# As/Cs mechanism audit\n"]
    lines.append(
        "This audit separates localisation, magnetic-target construction and cross-target "
        "co-location. It is diagnostic: none of the comparisons assigns an elemental mechanism.\n"
    )
    lines.append("## Site localisation\n")
    lines.append("| target | level | model | A sign | B sign | X sign | worst site |")
    lines.append("|---|---|---|---:|---:|---:|---|")
    for prop in PROPS:
        for level in TARGETS:
            for model in ("direct", "metric"):
                g = summary[(summary.property == prop) & (summary.level == level) &
                            (summary.model == model)].set_index("site")
                if len(g) != 3:
                    continue
                worst = g.sign_accuracy.idxmin()
                lines.append(
                    f"| {prop.replace('_', ' ')} | {level} | {model} | "
                    f"{g.loc['A','sign_accuracy']:.3f} | {g.loc['B','sign_accuracy']:.3f} | "
                    f"{g.loc['X','sign_accuracy']:.3f} | {worst} |"
                )

    lines.append("\n## Magnetic folding control\n")
    lines.append("| level | model | magnitude sign | signed-source sign | change |")
    lines.append("|---|---|---:|---:|---:|")
    for level in TARGETS:
        for model in ("direct", "metric"):
            g = summary[(summary.level == level) & (summary.model == model)]
            mag_group = g[g.property == "magnetic_moment"]
            signed_group = g[g.property == "magnetic_moment_signed"]
            mag = np.average(mag_group.sign_accuracy, weights=mag_group.n_signed)
            signed = np.average(signed_group.sign_accuracy, weights=signed_group.n_signed)
            lines.append(f"| {level} | {model} | {mag:.3f} | {signed:.3f} | {signed-mag:+.3f} |")
    lines.append(
        "\nThe signed-source control is not a physical replacement for moment magnitude. A "
        "large improvement would implicate the absolute-value fold; no improvement would rule "
        "out that simple explanation.\n"
    )

    lines.append("## Cross-target co-location\n")
    lines.append("| level | model | targets | failure Jaccard | error-rank rho |")
    lines.append("|---|---|---|---:|---:|")
    for r in cross.itertuples(index=False):
        lines.append(
            f"| {r.level} | {r.model} | {r.left_target} / {r.right_target} | "
            f"{r.failure_jaccard:.3f} | {r.absolute_error_spearman:+.3f} |"
        )

    lines.append("\n## Most fragile contexts\n")
    lines.append("Direct-model groups with at least 14 edges, ranked within each target and level.\n")
    lines.append("| target | level | site | context | n | sign | median |delta| |")
    lines.append("|---|---|---|---|---:|---:|---:|")
    direct = context[context.model == "direct"]
    for prop in PROPS:
        for level in TARGETS:
            g = direct[(direct.property == prop) & (direct.level == level)].nsmallest(3, "sign_accuracy")
            for r in g.itertuples(index=False):
                lines.append(
                    f"| {prop.replace('_',' ')} | {level} | {r.site} | {r.context} | "
                    f"{r.n_edges} | {r.sign_accuracy:.3f} | {r.median_abs_response:.3f} |"
                )

    lines.append("\n## B-substitution partner decomposition\n")
    lines.append("Direct-model B edges, averaged over all 14 A levels and seven anions.\n")
    lines.append("| target | held level | partner | n | sign | mean true delta | mean predicted delta |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    direct_partner = partner[partner.model == "direct"]
    for prop in PROPS:
        for level in TARGETS:
            g = direct_partner[(direct_partner.property == prop) &
                               (direct_partner.level == level)].sort_values("sign_accuracy")
            for r in g.itertuples(index=False):
                lines.append(
                    f"| {prop.replace('_',' ')} | {level} | {r.partner} | {r.n_edges} | "
                    f"{r.sign_accuracy:.3f} | {r.true_delta_mean:+.3f} | {r.pred_delta_mean:+.3f} |"
                )

    lines.append("\n## Model-class control\n")
    lines.append("| target | level | model | MAE | sign |")
    lines.append("|---|---|---|---:|---:|")
    for r in classes.itertuples(index=False):
        lines.append(
            f"| {r.property.replace('_',' ')} | {r.level} | {r.model} | "
            f"{r.mae:.3f} | {r.sign_accuracy:.3f} |"
        )

    lines.append("\n## Direct-response block attribution\n")
    lines.append("Formation-energy B substitutions, oriented as held level minus partner.\n")
    lines.append("| level | component | mean contribution | mean absolute contribution |")
    lines.append("|---|---|---:|---:|")
    selected = components[components.resolution.isin(["block", "total"])]
    for r in selected.itertuples(index=False):
        lines.append(
            f"| {r.level} | {r.component} | {r.mean_held_minus_partner:+.3f} | "
            f"{r.mean_abs_contribution:.3f} |"
        )

    lines.append("\n## Descriptor-family ablation\n")
    lines.append("Formation-energy B substitutions only; every row is refitted without the held level.\n")
    lines.append("| level | configuration | MAE | sign | predicted held-partner | bias |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for r in ablation.itertuples(index=False):
        lines.append(
            f"| {r.level} | {r.configuration} | {r.mae:.3f} | {r.sign_accuracy:.3f} | "
            f"{r.mean_pred_held_minus_partner:+.3f} | {r.mean_bias:+.3f} |"
        )
    lines.append(
        "\nBecause the elemental descriptors are strongly correlated, these refitted family "
        "ablations are more defensible than interpreting an individual ridge coefficient. "
        "They identify model dependence, not an elemental cause.\n"
    )

    lines.append("\n## Metadata control\n")
    lines.append(
        "Every As and Cs calculation in this grid uses the same recorded stack: RPBE, GPAW, "
        "PAW, cubic Pm-3m ABX3. The available metadata therefore contain no element-specific "
        "code, functional, pseudopotential family or prototype branch that can explain the gap."
    )
    write_report(REPORT, "\n".join(lines))
    print(f"wrote {REPORT} and eight CSV tables")


if __name__ == "__main__":
    main()
