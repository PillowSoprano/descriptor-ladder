"""The interpolation peaks split into design and omitted-part contributions.

Uses the 13 coverages, five designs and two properties of run_peak_vs_interaction_share.py, with no
penalty search and no new response data. Coordinates are orthonormal in the constrained one-hot
coefficient space, which keeps the minimum-norm convention of the original fits. Outputs are the
exact error components, the expected errors for an isotropic omitted part, and the errors with the
omitted part removed along the weakest or strongest determined directions.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.sparse as sp
from scipy.special import erf
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import run_peak_vs_interaction_share as previous
from prfield.graph.typed_graph import accepted, build_edges
from prfield.splits import observation as obs

OUT = ROOT / 'reports/peak_mode_coupling'


def basis(frame, levels, pairs):
    indices = {s: frame[f'site_{s}'].map({v: i for i, v in enumerate(levels[s])}).to_numpy()
               for s in ('A', 'B', 'X')}
    contrasts = {s: la.helmert(len(levels[s]), full=False).T[indices[s]] for s in indices}
    blocks = [np.ones((len(frame), 1))] + [contrasts[s] for s in indices]
    for pair in pairs:
        blocks.append(np.einsum('ni,nj->nij', contrasts[pair[0]], contrasts[pair[1]])
                      .reshape(len(frame), -1))
    return np.column_stack(blocks), indices


def bipartite_rank(left, right, a, b):
    rows = np.r_[left, a + right]
    cols = np.r_[a + right, left]
    graph = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(a+b, a+b))
    c = connected_components(graph, directed=False, return_labels=False)
    return a + b - c


def graph_rank(model, indices, train, sizes):
    idx = {s: values[train] for s, values in indices.items()}
    pairs = previous.MODELS[model]
    if len(pairs) == 1:
        x, y = pairs[0]
        other = next(s for s in sizes if s not in pairs[0])
        return bipartite_rank(idx[x] * sizes[y] + idx[y], idx[other],
                              sizes[x] * sizes[y], sizes[other])
    if len(pairs) == 2:
        common = next(iter(set(pairs[0]) & set(pairs[1])))
        x, y = [s for s in sizes if s != common]
        return sum(bipartite_rank(idx[x][idx[common] == k], idx[y][idx[common] == k],
                                  sizes[x], sizes[y]) for k in range(sizes[common]))
    return np.nan


def run():
    start = time.time()
    OUT.mkdir(exist_ok=True, parents=True)
    frame = pd.read_parquet(ROOT / 'data/interim/castelli_raw.parquet')
    _, aa, bb = previous.mh.panel_definitions(frame)[0]
    frame = frame[frame.site_A.isin(aa) & frame.site_B.isin(bb)].reset_index(drop=True)
    levels = {s: sorted(frame[f'site_{s}'].unique()) for s in ('A', 'B', 'X')}
    sizes = {s: len(v) for s, v in levels.items()}
    N = len(frame)
    lookup = pd.Series(np.arange(N), index=frame.sample_id)
    edges = {}
    for prop in previous.PROPS:
        e = accepted(build_edges(frame, prop, previous.mh.POLICY))
        edges[prop] = (lookup.loc[e.src].to_numpy(), lookup.loc[e.dst].to_numpy())
    old = pd.read_csv(ROOT / 'reports/peak_vs_interaction_share.csv')
    rows, modes = [], []
    for model, pairs in previous.MODELS.items():
        Z, indices = basis(frame, levels, pairs)
        p = Z.shape[1]
        full_diag = (Z * Z).sum(axis=0)
        full_inv = 1 / full_diag
        Y = frame[list(previous.PROPS)].to_numpy(float)
        beta = (Z.T @ Y) / full_diag[:, None]
        residual = Y - Z @ beta
        for target in previous.NP_TARGETS:
            frac = min(.95, target * p / N)
            for seed in previous.SEEDS:
                split = obs.random_nodes(frame, frac, seed=seed)
                train = frame.sample_id.isin(split.observed_ids).to_numpy()
                U, sv, Vt = la.svd(Z[train], full_matrices=False, check_finite=False)
                positive = sv > sv[0] * 1e-10
                sv, U, V = sv[positive], U[:, positive], Vt[positive].T
                rank = len(sv)
                # Ascending singular values: a fixed 10% of positive modes.
                order = np.argsort(sv)
                sv, U, V = sv[order], U[:, order], V[:, order]
                k = max(1, int(np.ceil(.1 * rank)))
                g = sv**2
                Pr = V @ V.T
                Gplus = (V / g) @ V.T
                # Covariance of coefficients induced by unit-variance omitted field.
                K = Gplus - (Pr * full_inv) @ Pr
                rg = graph_rank(model, indices, train, sizes)
                if np.isfinite(rg) and rg != rank:
                    raise ValueError(f'Graph/SVD rank disagree: {model} {rg} {rank}')
                projected = U.T @ residual[train]
                alias_beta = V @ (projected / sv[:, None])
                weak_beta = V[:, :k] @ (projected[:k] / sv[:k, None])
                strong_beta = V[:, -k:] @ (projected[-k:] / sv[-k:, None])
                missing_beta = (Pr - np.eye(p)) @ beta
                missing_nodes = Z @ missing_beta
                alias_nodes = Z @ alias_beta
                weak_nodes = Z @ weak_beta
                strong_nodes = Z @ strong_beta
                # Exact Gaussian residual risk on hidden endpoints, including
                # the residual correlation induced by complete-grid orthogonality.
                noise_B = K - np.diag(full_inv) + Pr * full_inv + full_inv[:, None] * Pr
                ZB = Z @ noise_B
                noise_node_diag = np.sum(ZB * Z, axis=1)
                for j, prop in enumerate(previous.PROPS):
                    src, dst = edges[prop]
                    held = ~train[src] & ~train[dst]
                    src, dst = src[held], dst[held]
                    ne = len(src)
                    D = sp.csr_matrix((np.r_[-np.ones(ne), np.ones(ne)],
                                       (np.r_[np.arange(ne), np.arange(ne)], np.r_[src, dst])),
                                      shape=(ne, N))
                    E = np.asarray(Z.T @ (D.T @ (D @ Z))) / ne
                    unident = missing_nodes[dst, j] - missing_nodes[src, j]
                    alias = alias_nodes[dst, j] - alias_nodes[src, j]
                    weak = weak_nodes[dst, j] - weak_nodes[src, j]
                    strong = strong_nodes[dst, j] - strong_nodes[src, j]
                    omitted = -(residual[dst, j] - residual[src, j])
                    error = unident + alias + omitted
                    tau2 = float(residual[:, j] @ residual[:, j] / (N-p))
                    geometry = float(np.sum(E * K.T))
                    floor_unit = 2 - float(np.diag(E) @ full_inv)
                    cross_unit = float(np.sum(E * (Pr * full_inv).T))
                    iso_risk = float(np.mean(unident**2) + tau2 *
                                     (geometry + floor_unit + 2 * cross_unit))
                    edge_var = tau2 * (2 + noise_node_diag[dst] + noise_node_diag[src]
                                       - 2 * np.sum(ZB[dst] * Z[src], axis=1))
                    assert np.min(edge_var) > -1e-8
                    std = np.sqrt(np.maximum(edge_var, 1e-30))
                    iso_mae = np.mean(std * np.sqrt(2/np.pi) * np.exp(-.5*(unident/std)**2)
                                      + unident * erf(unident/(std*np.sqrt(2))))
                    assert np.isclose(np.mean(unident**2+edge_var), iso_risk, rtol=1e-8)
                    alias_mse = float(np.mean(alias**2))
                    same = old[(old.model == model) & (old.property == prop) &
                               (old.seed == seed) & np.isclose(old.coverage, frac)]
                    assert len(same) == 1
                    reference = float(same.untuned.iloc[0])
                    row = dict(model=model, property=prop, target=target, coverage=frac,
                               seed=seed, n=int(train.sum()), p=p, rank=rank,
                               graph_rank=rg, nullity=p-rank, n_over_p=train.sum()/p,
                               min_positive_g=g[0], median_positive_g=np.median(g),
                               weak_count=k, residual_variance=tau2,
                               residual_node_mse=np.mean(residual[:, j]**2),
                               geometry=geometry, alias_mse=alias_mse,
                               isotropic_alias_mse=tau2*geometry,
                               orientation_lift=alias_mse/(tau2*geometry),
                               mse=np.mean(error**2), isotropic_mse=iso_risk, isotropic_mae=iso_mae,
                               unident_mse=np.mean(unident**2), omitted_mse=np.mean(omitted**2),
                               cross_unident_alias=2*np.mean(unident*alias),
                               cross_unident_omitted=2*np.mean(unident*omitted),
                               cross_alias_omitted=2*np.mean(alias*omitted),
                               weak_alias_mse=np.mean(weak**2),
                               mae=np.mean(np.abs(error)), old_mae=reference,
                               mae_no_weak=np.mean(np.abs(error-weak)),
                               mae_no_strong=np.mean(np.abs(error-strong)),
                               mae_no_alias=np.mean(np.abs(error-alias)),
                               mse_no_weak=np.mean((error-weak)**2),
                               mse_no_alias=np.mean((error-alias)**2),
                               max_reconstruct_error=np.max(np.abs(error -
                                  (D @ (Z @ (V @ ((U.T @ Y[train, j]) / sv)) - Y[:, j])))))
                    rows.append(row)
                    sensitivity = np.sum(V * (E @ V), axis=0)
                    expected_projection = 1 - g * np.sum(V**2 * full_inv[:, None], axis=0)
                    for q in range(rank):
                        modes.append(dict(model=model, property=prop, target=target, seed=seed,
                                          mode=q, rank=rank, g=g[q], weak=q<k,
                                          residual_projection_sq=projected[q, j]**2,
                                          isotropic_projection_sq=tau2*expected_projection[q],
                                          edge_sensitivity=sensitivity[q],
                                          diagonal_alias_energy=sensitivity[q]*projected[q,j]**2/g[q]))
        print(f'{model}: {time.time()-start:.1f} seconds', flush=True)
        pd.DataFrame(rows).to_csv(OUT / 'decomposition.csv', index=False)
    result = pd.DataFrame(rows)
    pd.DataFrame(modes).to_csv(OUT / 'modes.csv', index=False)
    curves = result.groupby(['property','model','target'], as_index=False).median(numeric_only=True)
    curves.to_csv(OUT / 'curves.csv', index=False)
    peaks = []
    for (prop, model), group in curves.groupby(['property','model']):
        group = group.sort_values('target').reset_index(drop=True)
        out = dict(property=prop, model=model)
        for metric in ['mae','old_mae','mae_no_weak','mae_no_strong','mae_no_alias',
                       'mse','isotropic_mse','isotropic_mae']:
            a = group[metric].to_numpy()
            rise = a / np.minimum.accumulate(np.r_[np.inf,a[:-1]]) - 1
            ix = int(np.argmax(rise))
            out[metric+'_rise_pct'] = max(0,100*rise[ix])
            out[metric+'_peak_at'] = group.target.iloc[ix]
            if metric == 'mae':
                for key in ['geometry','alias_mse','isotropic_alias_mse','orientation_lift',
                            'unident_mse','omitted_mse','residual_variance','rank','nullity',
                            'min_positive_g','weak_alias_mse','mae_no_weak','mae','mse',
                            'cross_unident_alias','cross_unident_omitted','cross_alias_omitted']:
                    out['at_peak_'+key] = group[key].iloc[ix]
        peaks.append(out)
    peaks = pd.DataFrame(peaks)
    peaks.to_csv(OUT / 'peaks.csv', index=False)
    summary = dict(seconds=time.time()-start, designs=len(result),
                   max_mae_difference=float(np.max(np.abs(result.mae-result.old_mae))),
                   max_relative_mae_difference=float(np.max(np.abs(result.mae/result.old_mae-1))),
                   max_reconstruction_error=float(result.max_reconstruct_error.max()),
                   graph_rank_matches=bool(np.all(result.dropna(subset=['graph_rank']).graph_rank ==
                                                  result.dropna(subset=['graph_rank'])['rank'])),
                   weakest_fraction=.1, properties=list(previous.PROPS), sizes=sizes)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(peaks[['property','model','mae_rise_pct','mae_no_weak_rise_pct',
                 'mae_no_alias_rise_pct','isotropic_mse_rise_pct','at_peak_orientation_lift']]
          .to_string(index=False), flush=True)


if __name__ == '__main__':
    run()
