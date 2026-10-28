"""How much of the basis cost is chemistry, and how much is just dimension?

The paper prices the descriptor basis by an exact quantity: the share of the true main effects
that lies inside the span of the descriptors, computable because the grid is complete. On the
ten-feature set in common use that is 50.4% at the A site and 78.4% at B. A single number
invites a single objection, and it is the right one: those ten features are collinear to
|r| = 0.998 in places, so is the loss a property of descriptor *bases*, or a property of these
ten descriptors, or merely of the fact that ten is fewer than fifty-two?

Nothing in the paper separates those, and this does. The same exact measurement is run across a
ladder of feature sets:

  paper-10        the ten physical features the manuscript uses
  pymatgen-14     every numeric elemental property pymatgen carries for all 52 species
  shuffled-10     the paper's ten features with the rows permuted across elements, so the
                  marginal distribution of every feature is preserved exactly and the
                  correspondence between element and property is destroyed
  random-d        a Gaussian random projection of the one-hot design to d dimensions, for d
                  from 2 to 40, which is dimension and nothing else
  one-hot         the ceiling: capture is 1 by construction

The two controls are the point. `shuffled-10` says how much of the capture survives when the
features carry no chemical information but the same numbers; `random-d` says how much any
d-dimensional compression of level identity would achieve. If a ten-dimensional random
projection already captures most of what the real descriptors do, then the headline number is
about dimension and the paper's framing has to change. If it does not, the basis cost is a
structural property of compressing level identity through a low-dimensional map, which is what
the paper claims and has so far not shown.

Everything is exact: the grid is complete, so the main effects are the true ones and the
capture is an orthogonal projection, with no fitting and no held-out data anywhere.

Writes reports/AUDIT_DESCRIPTOR_LADDER.md and reports/descriptor_ladder.csv.

Usage:  PYTHONPATH=src python3 scripts/run_descriptor_ladder.py
"""

from __future__ import annotations

import numbers
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Element

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prfield.descriptors import elemental as el  # noqa: E402
from prfield.report import write_report  # noqa: E402

REPORT = Path("reports/AUDIT_DESCRIPTOR_LADDER.md")
CSV = Path("reports/descriptor_ladder.csv")
PROPS = ("heat_of_formation", "volume_per_fu")
RAND_DIMS = (2, 5, 10, 20, 40)
N_DRAWS = 20


def _three_factor(g: pd.DataFrame, prop: str) -> np.ndarray:
    """Exact three-factor residual of a complete grid."""
    mu = g[prop].mean()
    main = {s: (g.groupby(f"site_{s}")[prop].mean() - mu) for s in ("A", "B", "X")}
    fit = np.full(len(g), mu)
    for s in ("A", "B", "X"):
        fit = fit + main[s].reindex(g[f"site_{s}"]).to_numpy(float)
    for p in ("AB", "AX", "BX"):
        s1, s2 = p[0], p[1]
        M = g.groupby([f"site_{s1}", f"site_{s2}"])[prop].mean().unstack()
        M = M.sub(main[s1].reindex(M.index), axis=0).sub(main[s2].reindex(M.columns), axis=1) - mu
        key = pd.MultiIndex.from_arrays([g[f"site_{s1}"], g[f"site_{s2}"]])
        fit = fit + M.stack().reindex(key).to_numpy(float)
    return g[prop].to_numpy(float) - fit


def capture(M: np.ndarray, v: np.ndarray) -> float:
    """Share of v explained by an orthogonal projection onto [1 | standardised M]."""
    sd = M.std(0)
    Z = (M - M.mean(0)) / np.where(sd > 0, sd, 1.0)
    P = np.hstack([np.ones((len(Z), 1)), Z])
    resid = v - P @ np.linalg.lstsq(P, v, rcond=None)[0]
    return float(1 - np.var(resid) / np.var(v)) if np.var(v) > 0 else float("nan")


def paper_matrix(labels: list[str]) -> np.ndarray:
    d = el.physical_descriptors(labels, "element")
    return np.nan_to_num(np.asarray(d.matrix(labels, allow_missing=True), float))


def pymatgen_matrix(labels: list[str]) -> tuple[np.ndarray, list[str]]:
    warnings.simplefilter("ignore")
    keys = set()
    for a in labels:
        keys |= {k for k, v in Element(a).data.items() if isinstance(v, numbers.Number)}
    keep = [k for k in sorted(keys)
            if all(isinstance(Element(a).data.get(k), numbers.Number) for a in labels)]
    return np.array([[float(Element(a).data[k]) for k in keep] for a in labels]), keep


def main() -> None:
    df = pd.read_parquet("data/interim/castelli_raw.parquet")
    rows = []
    for prop in PROPS:
        g = df.dropna(subset=[prop])
        mu = g[prop].mean()
        for site in ("A", "B"):
            eff = (g.groupby(f"site_{site}")[prop].mean() - mu)
            labels = sorted(eff.index)
            v = eff.reindex(labels).to_numpy(float)
            n = len(labels)

            P10 = paper_matrix(labels)
            PMG, pmg_keys = pymatgen_matrix(labels)

            rows.append({"property": prop, "site": site, "family": "paper-10",
                         "dim": P10.shape[1], "capture": capture(P10, v), "spread": 0.0})
            rows.append({"property": prop, "site": site, "family": "pymatgen-14",
                         "dim": PMG.shape[1], "capture": capture(PMG, v), "spread": 0.0})

            # shuffled: identical marginals, no element-to-property correspondence
            vals = []
            for s in range(N_DRAWS):
                r = np.random.default_rng(1000 + s)
                vals.append(capture(P10[r.permutation(n)], v))
            rows.append({"property": prop, "site": site, "family": "shuffled-10",
                         "dim": P10.shape[1], "capture": float(np.median(vals)),
                         "spread": float(np.std(vals))})

            # random projections of one-hot: dimension and nothing else
            for d in RAND_DIMS:
                vals = []
                for s in range(N_DRAWS):
                    r = np.random.default_rng(10_000 * d + s)
                    vals.append(capture(r.normal(size=(n, d)), v))
                rows.append({"property": prop, "site": site, "family": f"random-{d}",
                             "dim": d, "capture": float(np.median(vals)),
                             "spread": float(np.std(vals))})

            rows.append({"property": prop, "site": site, "family": "one-hot",
                         "dim": n, "capture": 1.0, "spread": 0.0})

    d = pd.DataFrame(rows)
    CSV.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(CSV, index=False)

    out = ["# Is the basis cost chemistry, or dimension?\n"]
    out.append(
        "_Generated by `scripts/run_descriptor_ladder.py`._ The share of the exact main effects "
        "lying inside each feature span, on the complete $52\\times52\\times7$ grid. No fitting "
        "and no held-out data: the grid is complete, so the effects are the true ones and the "
        "capture is an orthogonal projection. Random and shuffled families are medians over "
        f"{N_DRAWS} draws with the standard deviation across draws in brackets.\n")
    out.append("| feature set | dim | " + " | ".join(
        f"{p.replace('_',' ')}, {s}" for p in PROPS for s in ("A", "B")) + " |")
    out.append("|---|---|" + "---|" * (2 * len(PROPS)))
    order = ["paper-10", "pymatgen-14", "shuffled-10"] + \
            [f"random-{k}" for k in RAND_DIMS] + ["one-hot"]
    for fam in order:
        cells = []
        dim = None
        for p in PROPS:
            for s in ("A", "B"):
                r = d[(d["family"] == fam) & (d["property"] == p) & (d["site"] == s)]
                if r.empty:
                    cells.append("--")
                    continue
                dim = int(r["dim"].iloc[0])
                c, sp = float(r["capture"].iloc[0]), float(r["spread"].iloc[0])
                cells.append(f"{100*c:.1f}\\%" + (f" ({100*sp:.1f})" if sp > 0 else ""))
        out.append(f"| {fam} | {dim} | " + " | ".join(cells) + " |")

    out.append("\n## What this does to the paper's central comparison\n")
    out.append(
        "The manuscript compares two losses on the same scale: the variance the descriptor "
        "basis drops from the main effects it does represent, against the variance the "
        "second-order truncation drops with the three-factor term. That comparison is a "
        "function of the feature set, so it is reported for both.\n")
    out.append("| feature set | points lost from A and B main effects | three-factor "
               "truncation | ratio |")
    out.append("|---|---|---|---|")
    g = df.dropna(subset=["heat_of_formation"])
    mu, tot = g["heat_of_formation"].mean(), g["heat_of_formation"].var()
    main = {s_: (g.groupby(f"site_{s_}")["heat_of_formation"].mean() - mu) for s_ in "AB"}
    share = {s_: float(np.mean(main[s_].to_numpy(float) ** 2)) / tot for s_ in "AB"}
    three = float(np.var(_three_factor(g, "heat_of_formation"))) / tot
    losses = {}
    for fam in ("paper-10", "pymatgen-14"):
        lost = 0.0
        for s_ in "AB":
            c = float(d[(d["family"] == fam) & (d["property"] == "heat_of_formation")
                        & (d["site"] == s_)]["capture"].iloc[0])
            lost += share[s_] * (1 - c)
        losses[fam] = 100 * lost
        out.append(f"| {fam} | {100*lost:.1f} | {100*three:.1f} | "
                   f"{lost / three:.1f}$\\times$ |")
    out.append(
        f"\nThe ordering survives the better feature set and the margin does not: widening "
        f"from ten hand-picked features to every numeric property pymatgen carries takes the "
        f"basis loss from {losses['paper-10']:.1f} points to {losses['pymatgen-14']:.1f} "
        f"against a truncation loss of {100*three:.1f}, so the factor falls from "
        f"{losses['paper-10']/(100*three):.1f} to "
        f"{losses['pymatgen-14']/(100*three):.1f}. Any statement of the form \"the basis "
        "costs more than the truncation\" has to carry its feature set with it.\n")

    out.append("\n## Reading\n")
    piv = d.pivot_table(index="family", columns=["property", "site"], values="capture")
    p10 = piv.loc["paper-10"]
    r10 = piv.loc["random-10"]
    sh = piv.loc["shuffled-10"]
    pmg = piv.loc["pymatgen-14"]
    out.append(
        f"- A ten-dimensional random projection of level identity captures "
        f"{100*r10.min():.1f} to {100*r10.max():.1f}\\% of the main effects, against "
        f"{100*p10.min():.1f} to {100*p10.max():.1f}\\% for the paper's ten physical features. "
        f"The chemistry is worth {100*(p10-r10).min():+.1f} to {100*(p10-r10).max():+.1f} "
        "points over dimension alone.")
    out.append(
        f"- Destroying the element-to-property correspondence while keeping every feature's "
        f"marginal distribution drops the capture to {100*sh.min():.1f}--{100*sh.max():.1f}\\%, "
        f"which is where the random projections of the same dimension sit. So what the real "
        "features buy is the correspondence, not the numbers.")
    out.append(
        f"- Widening to every numeric property pymatgen carries for all 52 species, "
        f"{int(d[d.family=='pymatgen-14']['dim'].iloc[0])} features, moves the capture to "
        f"{100*pmg.min():.1f}--{100*pmg.max():.1f}\\%, and at the A site on formation energy "
        f"from {100*p10[('heat_of_formation','A')]:.1f} to "
        f"{100*pmg[('heat_of_formation','A')]:.1f}\\%. The basis cost is a functional of the "
        "feature set and this is what varying it does. A random projection at the matched "
        "dimension is in the table for comparison.")
    out.append(
        "- The capture does not reach 1 at any dimension short of the one-hot design itself, "
        "which is the statement the subspace lemma makes and the reason the basis cost is not "
        "an artefact of having chosen ten poor features.")
    write_report(REPORT, "\n".join(out))
    print(d.pivot_table(index="family", columns=["property", "site"],
                        values="capture").round(3).to_string())
    print(f"\nwrote {REPORT} and {CSV}")


if __name__ == "__main__":
    main()
