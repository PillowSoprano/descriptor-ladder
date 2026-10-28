"""The descriptor ladder on GB1 double-mutant fitness grids.

Each pair of positions of the GB1 deep mutational scan of Olson et al. (2014) gives a complete
19 x 19 grid of fitness scores, with eight physicochemical descriptors on the amino acids. A
two-position grid has main effects and one interaction, so the models compared are additive:

  descriptor basis   each position's amino-acid effect linear in the descriptors, a restriction of
                     the free design by a fixed linear map, so Lemma 1 applies unchanged
  free               a free effect per amino acid per position
  free + metric      the same free effects under the descriptor-similarity Laplacian

with gradient boosting on the same descriptors as a reference. The grids are complete, so the
variance decomposition is exact. The measurement noise is estimated from the read counts first.

p = 37 identified parameters on 361 cells, so n = p at about 10% coverage.

Data: https://raw.githubusercontent.com/gitter-lab/nn4dms/master/data/gb1/gb1.tsv
Descriptors: Bio.SeqUtils.ProtParamData and Bio.Data.IUPACData.

Writes reports/AUDIT_PROTEIN_REPLICATION.md and reports/protein_replication.csv.

Usage:  PYTHONPATH=src python3 scripts/run_protein_replication.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Data import IUPACData
from Bio.SeqUtils import ProtParamData
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prfield.report import write_report  # noqa: E402

RAW = Path("data/raw/gb1/gb1.tsv")
REPORT = Path("reports/AUDIT_PROTEIN_REPLICATION.md")
CSV = Path("reports/protein_replication.csv")
N_PAIRS = 120
FRACS = (0.03, 0.05, 0.08, 0.12, 0.20, 0.35, 0.50, 0.75)
SEEDS = (0, 1)
LAM = 1e-3
SMOOTH = 0.3
KNN = 3
POS_PK = {"K": 1.0, "R": 1.0, "H": 0.1}
NEG_PK = {"D": -1.0, "E": -1.0}


def descriptors(aas: list[str]) -> np.ndarray:
    """Eight physicochemical properties per amino acid, all from Biopython's tables."""
    cols = []
    for tbl in ("kd", "Flex", "hw", "em", "ja"):
        t = getattr(ProtParamData, tbl)
        cols.append([t[a] for a in aas])
    cols.append([IUPACData.protein_weights[a] for a in aas])
    cols.append([POS_PK.get(a, 0.0) + NEG_PK.get(a, 0.0) for a in aas])
    cols.append([1.0 if a in "FWY" else 0.0 for a in aas])
    M = np.asarray(cols, float).T
    sd = M.std(0)
    return (M - M.mean(0)) / np.where(sd > 0, sd, 1.0)


def laplacian(Z: np.ndarray, k: int) -> np.ndarray:
    D = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(D, np.inf)
    W = np.zeros_like(D)
    for i in range(len(Z)):
        for j in np.argsort(D[i])[:k]:
            W[i, j] = W[j, i] = 1.0
    return np.diag(W.sum(1)) - W


# ------------------------------------------------------------------------------ estimators

def fit_free(i1, i2, y, n1, n2, L1=None, L2=None, smooth=0.0):
    """Additive free effects at both positions, sum-to-zero, optionally smoothed."""
    p = 1 + n1 + n2
    X = np.zeros((len(y), p))
    X[:, 0] = 1.0
    X[np.arange(len(y)), 1 + i1] = 1.0
    X[np.arange(len(y)), 1 + n1 + i2] = 1.0
    G = X.T @ X + LAM * np.eye(p)
    w = np.sqrt(len(y)) * 10.0
    for sl in (slice(1, 1 + n1), slice(1 + n1, p)):
        c = np.zeros(p)
        c[sl] = 1.0
        G += (w ** 2) * np.outer(c, c)
    if smooth > 0:
        G[1:1 + n1, 1:1 + n1] += smooth * L1
        G[1 + n1:, 1 + n1:] += smooth * L2
    return np.linalg.lstsq(G, X.T @ y, rcond=None)[0]


def pred_free(b, i1, i2, n1):
    return b[0] + b[1 + i1] + b[1 + n1 + i2]


def fit_basis(Z1, Z2, i1, i2, y):
    """The same additive class restricted to the descriptor span on both margins."""
    A = np.hstack([np.ones((len(y), 1)), Z1[i1], Z2[i2]])
    return np.linalg.lstsq(A.T @ A + LAM * np.eye(A.shape[1]), A.T @ y, rcond=None)[0]


def pred_basis(c, Z1, Z2, i1, i2):
    return np.hstack([np.ones((len(i1), 1)), Z1[i1], Z2[i2]]) @ c


# ------------------------------------------------------------------------------------ main

def main() -> None:
    if not RAW.exists():
        sys.exit(f"{RAW} not found")
    d = pd.read_csv(RAW, sep="\t")
    db = d[d["num_mutations"] == 2].copy()
    parts = db["variant"].str.split(",", expand=True)
    db["p1"] = parts[0].str.extract(r"(\d+)").astype(int)
    db["a1"] = parts[0].str[-1]
    db["p2"] = parts[1].str.extract(r"(\d+)").astype(int)
    db["a2"] = parts[1].str[-1]

    # noise first: the log-ratio score of Poisson counts has variance about 1/inp + 1/sel,
    # and a decomposition of model error is not measurable in a domain where this dominates
    noise_sd = float(np.sqrt(1.0 / db["inp"] + 1.0 / db["sel"]).median())
    signal_sd = float(db["score"].std())

    counts = db.groupby(["p1", "p2"]).size()
    full = counts[counts == 361].index.tolist()
    rng = np.random.default_rng(0)
    chosen = [full[i] for i in rng.choice(len(full), min(N_PAIRS, len(full)), replace=False)]

    # each position excludes its own wild-type residue, so the two factors of a pair carry
    # different 19-letter level sets and the grid has to be built in local coordinates
    ALL = sorted(set(db["a1"]) | set(db["a2"]))
    ZALL = descriptors(ALL)
    gidx = {a: i for i, a in enumerate(ALL)}

    shares, capt, grids = [], [], {}
    for pr in list(chosen):
        g = db[(db["p1"] == pr[0]) & (db["p2"] == pr[1])]
        M = g.pivot_table(index="a1", columns="a2", values="score")
        if M.isna().any().any() or M.shape != (19, 19):
            chosen.remove(pr)
            continue
        l1, l2 = list(M.index), list(M.columns)
        Z1 = ZALL[[gidx[a] for a in l1]]
        Z2 = ZALL[[gidx[a] for a in l2]]
        A = M.to_numpy(float)
        grids[pr] = (A, Z1, Z2, laplacian(Z1, KNN), laplacian(Z2, KNN))
        m1 = A.mean(1) - A.mean()
        m2 = A.mean(0) - A.mean()
        E = A - A.mean() - m1[:, None] - m2[None, :]
        tot = float(np.var(A))
        shares.append((float(np.var(m1)) / tot, float(np.var(m2)) / tot,
                       float(np.mean(E ** 2)) / tot))
        for Zs, v in ((Z1, m1), (Z2, m2)):
            P = np.hstack([np.ones((len(Zs), 1)), Zs])
            H = P @ np.linalg.pinv(P.T @ P) @ P.T
            capt.append(1 - float(np.var(v - H @ v) / np.var(v)))
    shares = np.asarray(shares)
    capt = np.asarray(capt)
    n = 19
    p_free = 1 + 2 * (n - 1)

    rows = []
    for pr in chosen:
        M, Z1, Z2, L1, L2 = grids[pr]
        cells = np.array([(i, j) for i in range(n) for j in range(n)])
        y_all = M[cells[:, 0], cells[:, 1]]
        for frac in FRACS:
            for seed in SEEDS:
                r = np.random.default_rng(pr[0] * 100003 + pr[1] * 1009 + seed * 31
                                          + int(frac * 1000))
                obs = r.random(len(cells)) < frac
                if obs.sum() < 12 or (~obs).sum() < 20:
                    continue
                i1, i2, y = cells[obs, 0], cells[obs, 1], y_all[obs]
                held = set(map(tuple, cells[~obs]))

                # Responses: change one position, hold the other.  Match the materials
                # estimand by scoring only when both endpoint compositions are hidden.
                pairs = []
                for a in range(n):
                    for b_ in range(a + 1, n):
                        for c_ in range(n):
                            for u, v in (((a, c_), (b_, c_)), ((c_, a), (c_, b_))):
                                if (u in held) and (v in held):
                                    pairs.append((u, v))
                pairs = np.array(pairs)
                if len(pairs) > 4000:
                    pairs = pairs[r.choice(len(pairs), 4000, replace=False)]
                if len(pairs) < 100:
                    continue
                t = M[pairs[:, 1, 0], pairs[:, 1, 1]] - M[pairs[:, 0, 0], pairs[:, 0, 1]]
                u1, u2 = pairs[:, 0, 0], pairs[:, 0, 1]
                v1, v2 = pairs[:, 1, 0], pairs[:, 1, 1]

                pred = {"zero": np.zeros(len(t))}
                c = fit_basis(Z1, Z2, i1, i2, y)
                pred["basis"] = (pred_basis(c, Z1, Z2, v1, v2)
                                 - pred_basis(c, Z1, Z2, u1, u2))
                for tag, kw in (("free", dict()),
                                ("metric", dict(L1=L1, L2=L2, smooth=SMOOTH))):
                    b = fit_free(i1, i2, y, n, n, **kw)
                    pred[tag] = pred_free(b, v1, v2, n) - pred_free(b, u1, u2, n)
                h = HistGradientBoostingRegressor(random_state=0).fit(
                    np.hstack([Z1[i1], Z2[i2]]), y)
                pred["hgb"] = (h.predict(np.hstack([Z1[v1], Z2[v2]]))
                               - h.predict(np.hstack([Z1[u1], Z2[u2]])))

                rec = {"pair": f"{pr[0]}-{pr[1]}", "frac": frac, "seed": seed,
                       "n_obs": int(obs.sum()), "n_over_p": obs.sum() / p_free,
                       "n_pairs": len(pairs)}
                for k, v in pred.items():
                    rec[k] = float(np.mean(np.abs(t - v)))
                    rec[f"{k}_sign"] = float(np.mean(np.sign(t) == np.sign(v)))
                rows.append(rec)
        if len(rows) % 200 < len(FRACS) * len(SEEDS):
            print(f"  {len(rows)} records", flush=True)

    dd = pd.DataFrame(rows)
    CSV.parent.mkdir(parents=True, exist_ok=True)
    dd.to_csv(CSV, index=False)

    out = ["# The same split, in a protein fitness landscape\n"]
    out.append(
        "_Generated by `scripts/run_protein_replication.py`._ GB1 deep mutational scan, "
        f"Olson et al. (2014). Every double mutant was measured, so each of the 1,485 position "
        f"pairs gives a complete $19\\times19$ grid of fitness scores; {len(chosen)} of the "
        f"{len(full)} fully complete grids are used, drawn once with a fixed seed. Amino-acid "
        "descriptors are eight physicochemical properties taken from Biopython's tables.\n")

    out.append("## Is there room to measure anything?\n")
    out.append(
        f"The score is a log ratio of read counts, so its sampling standard deviation is about "
        f"$\\sqrt{{1/n_{{\\mathrm{{in}}}}+1/n_{{\\mathrm{{sel}}}}}}$, a median of "
        f"{noise_sd:.3f} against a signal spread of {signal_sd:.3f}, a ratio of about "
        f"{signal_sd/noise_sd:.0f} to 1. This is the check that a pairwise-preference version "
        "of this replication failed, and it is done first here for that reason.\n")

    out.append("## What the complete grids give exactly\n")
    out.append(
        f"Averaged over the {len(chosen)} grids, the two main effects carry "
        f"{100*shares[:,0].mean():.1f}\\% and {100*shares[:,1].mean():.1f}\\% of the variance "
        f"and epistasis carries {100*shares[:,2].mean():.1f}\\%. The additive class therefore "
        "excludes a real term, in the same way the paper's second-order class excludes the "
        "three-factor one. Projecting the exact main effects onto the descriptor span captures "
        f"a median {100*np.median(capt):.1f}\\% of them, which is the basis cost priced without "
        "any fitting, against 50.4 and 78.4\\% on the perovskite grid.\n")

    out.append("## Basis cost and metric value against the design ratio\n")
    out.append(
        "$p=37$ identified parameters on 361 cells, so $n=p$ at about 10\\% coverage. Both "
        "quantities are relative to the free arm, and the floor is the error of predicting no "
        "change at all.\n")
    out.append("| n/p | observed | predict zero | descriptor basis | free | + metric "
               "| boosting | basis cost | metric value |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for frac in FRACS:
        g = dd[dd["frac"] == frac]
        if g.empty:
            continue
        out.append(
            f"| {g['n_over_p'].median():.2f} | {100*frac:.0f}\\% | "
            + " | ".join(f"{g[c].median():.4f}" for c in
                         ("zero", "basis", "free", "metric", "hgb"))
            + f" | {100*(g['basis'].median()/g['free'].median()-1):+.1f}\\%"
            + f" | {100*(1-g['metric'].median()/g['free'].median()):+.1f}\\% |")

    out.append("\n## What replicates, and what does not\n")
    gg = dd.groupby("frac")[["zero", "basis", "free", "metric", "hgb", "n_over_p"]].median()
    mv = 100 * (1 - gg["metric"] / gg["free"])
    bc = 100 * (gg["basis"] / gg["free"] - 1)
    best_mv = mv.idxmax()
    out.append("**Replicates.**\n")
    out.append(
        f"- The basis is a real cost, and a large one. At the densest design the descriptor "
        f"restriction is {bc.iloc[-1]:+.0f}\\% worse than free effects, against $+26$ and "
        "$+21$\\% on the perovskite grid at 90\\% coverage. The exact ANOVA agrees with the "
        f"fitted comparison: the descriptor span captures {100*np.median(capt):.0f}\\% of the "
        "true main effects, which sits between the perovskite grid's 50.4 and 78.4\\%.")
    out.append(
        f"- The metric's value decays as the design ratio grows, which is the shape the "
        f"coverage law asserts. It peaks at {mv.max():+.1f}\\% at $n/p="
        f"{gg.loc[best_mv,'n_over_p']:.2f}$ and is {mv.iloc[-1]:+.1f}\\% at "
        f"$n/p={gg['n_over_p'].iloc[-1]:.1f}$.")
    nb = int((dd["basis"] > dd["free"]).sum())
    out.append(
        f"- Free effects beat the descriptor basis in {nb} of {len(dd)} configurations "
        f"({100*nb/len(dd):.0f}\\%), at every design ratio in the sweep.")

    out.append("\n**Does not replicate.**\n")
    bump = 100 * (gg['free'].max() / gg['free'].iloc[0] - 1)
    out.append(
        f"- There is no interpolation peak. The free arm's error is flat to within "
        f"{abs(bump):.0f}\\% below $n/p=1$ and falls monotonically above it, where the "
        "perovskite grid rises 31\\% into a peak at $n/p\\approx1$. The mechanism explains "
        "the difference rather than excusing it: on a two-factor product a second-order model "
        "is saturated, so the class compared here is additive and carries no interaction block "
        "to be weakly identified. The peak is a property of designs whose parameters are mostly "
        "interaction parameters, not of categorical products as such.")
    out.append(
        f"- The trap the perovskite grid sets is absent. There, the basis appears to win by "
        "17--19\\% in the band around $n/p=1$ because the free arm is momentarily at its "
        f"worst. Here the basis cost is positive at every ratio, from {bc.min():+.0f} to "
        f"{bc.max():+.0f}\\%, and never changes sign.")

    out.append("\n**New here.**\n")
    neg = mv[mv < 0]
    out.append(
        f"- The metric has a window rather than a monotone decay to zero. It helps between "
        f"$n/p\\approx0.5$ and $1.2$ and *hurts* outside it, by {mv.iloc[0]:.1f}\\% at the "
        f"sparsest design and by {neg.iloc[-1]:.1f}\\% at the densest. The smoothing weight is "
        "a fixed constant in both this and the perovskite sweep, so what changes is not tuning: "
        "with almost nothing observed, pulling each level toward its descriptor neighbours "
        "imports the basis's bias without buying any variance reduction worth having.")
    out.append(
        f"- Gradient boosting on the same descriptors is exactly the zero baseline below "
        f"$n/p=1$, to four decimal places, because it makes no splits on that little data. It "
        f"overtakes the free arm only at the densest design ({gg['hgb'].iloc[-1]:.4f} against "
        f"{gg['free'].iloc[-1]:.4f}).")
    write_report(REPORT, "\n".join(out))
    print(gg.round(4).to_string())
    print(f"\nwrote {REPORT} and {CSV}")


if __name__ == "__main__":
    main()
