"""Semiparametric kernel transfer for a left-out level of a complete grid.

The B-site effect is split into a descriptor component and a categorical remainder,

    eta_B = g_B(phi_B) + u_B,

with covariance ``K_g + rho I``. An element seen in training uses both terms; a left-out element
has no covariance through ``u_B`` and transfers through ``g_B`` alone. On a complete A x B x X grid
the B-involving second-order effects can be extracted element by element, so the AB and BX blocks
are the product kernels ``C_B (x) I_A`` and ``C_B (x) I_X`` without forming a large Kronecker matrix.

The module handles left-out B-site elements on the complete grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..descriptors.elemental import ElementalDescriptors

GAMMAS = (0.01, 0.03, 0.1, 0.3, 1.0)
RHOS = (0.01, 0.1, 1.0, 10.0, 100.0)
UNCERTAINTY_MULTIPLIERS = (0.25, 1.0, 4.0)
NOISE_FLOOR = 1e-6


def _descriptor_matrix(desc: ElementalDescriptors, labels: Sequence) -> np.ndarray:
    matrix = np.asarray(desc.matrix(list(labels), allow_missing=True), dtype=float)
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def _standardise(matrix: np.ndarray, mean=None, scale=None):
    if mean is None:
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        scale = np.where(scale > 0, scale, 1.0)
    return (matrix - mean) / scale, mean, scale


def descriptor_kernel(a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
    """Combined constant, linear, and radial descriptor kernel."""
    squared_distance = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1)
    return 1.0 + a @ b.T + np.exp(-gamma * squared_distance)


def _complete_cube(df: pd.DataFrame, prop: str):
    work = df.dropna(subset=[prop]).copy()
    a_levels = sorted(work.site_A.unique().tolist())
    b_levels = sorted(work.site_B.unique().tolist())
    x_levels = sorted(work.site_X.unique().tolist())
    expected = len(a_levels) * len(b_levels) * len(x_levels)
    if len(work) != expected or work.duplicated(["site_A", "site_B", "site_X"]).any():
        raise ValueError("semiparametric surface fit requires one complete A x B x X grid")
    ai = {level: i for i, level in enumerate(a_levels)}
    bi = {level: i for i, level in enumerate(b_levels)}
    xi = {level: i for i, level in enumerate(x_levels)}
    cube = np.empty((len(b_levels), len(a_levels), len(x_levels)), dtype=float)
    for row in work.itertuples(index=False):
        cube[bi[row.site_B], ai[row.site_A], xi[row.site_X]] = float(getattr(row, prop))
    return cube, a_levels, b_levels, x_levels


def decompose_two_way_field(field: np.ndarray):
    """Orthogonally decompose a balanced A-by-X field.

    Returns the scalar mean, centred A and X conditional effects, and the centred A-by-X
    residual.  The four broadcast components are mutually orthogonal under uniform cell
    weighting.
    """
    field = np.asarray(field, dtype=float)
    if field.ndim != 2 or 0 in field.shape:
        raise ValueError("field must be a non-empty two-dimensional array")
    mean = float(field.mean())
    a_component = field.mean(axis=1) - mean
    x_component = field.mean(axis=0) - mean
    ax_residual = field - mean - a_component[:, None] - x_component[None, :]
    return mean, a_component, x_component, ax_residual


def project_cube_to_second_order(cube: np.ndarray):
    """Project a balanced B-by-A-by-X cube onto the full second-order ANOVA space."""
    cube = np.asarray(cube, dtype=float)
    if cube.ndim != 3 or 0 in cube.shape:
        raise ValueError("cube must be a non-empty B-by-A-by-X array")
    b_main = cube.mean(axis=(1, 2))
    ab = cube.mean(axis=2) - b_main[:, None]
    bx = cube.mean(axis=1) - b_main[:, None]
    b_additive = b_main[:, None, None] + ab[:, :, None] + bx[:, None, :]
    residual = cube - b_additive
    ax = residual.mean(axis=0)
    triple = residual - ax[None, :, :]
    return cube - triple, triple


def decompose_b_surfaces(df: pd.DataFrame, prop: str):
    """Extract B main, AB, BX, shared AX, and unresolved three-factor fields.

    Each B target is defined within the compositions of its own element, so holding out another B level cannot alter
    its target.  Under the second-order projection, the returned columns are exactly the B
    main effect and the AB/BX effects plus their observed A/X main contributions.
    """
    cube, a_levels, b_levels, x_levels = _complete_cube(df, prop)
    projection, triple = project_cube_to_second_order(cube)
    main = projection.mean(axis=(1, 2))
    a_component = projection.mean(axis=2) - main[:, None]
    x_component = projection.mean(axis=1) - main[:, None]
    ax_surface = (projection - main[:, None, None] - a_component[:, :, None]
                  - x_component[:, None, :]).mean(axis=0)
    targets = np.concatenate([main[:, None], a_component, x_component], axis=1)
    return targets, ax_surface, triple, a_levels, b_levels, x_levels


def _pairwise_mean_absolute(values: np.ndarray) -> float:
    """Mean absolute pairwise difference within every row."""
    if values.shape[1] < 2:
        return 0.0
    diff = np.abs(values[:, :, None] - values[:, None, :])
    upper = np.triu_indices(values.shape[1], k=1)
    return float(diff[:, upper[0], upper[1]].mean())


@dataclass(frozen=True)
class Hyperparameters:
    gamma: float
    rho: float
    uncertainty_multiplier: float
    inner_score: float


def _eigendecomposition(z: np.ndarray, gamma: float, rho: float):
    covariance = descriptor_kernel(z, z, gamma) + rho * np.eye(len(z))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return covariance, np.clip(eigenvalues, 1e-12, None), eigenvectors


def _posterior_coefficients(
    y_standardised: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    noise: np.ndarray,
):
    projected = eigenvectors.T @ y_standardised
    denominator = eigenvalues[:, None] + noise[None, :]
    alpha = eigenvectors @ (projected / denominator)
    inverse_diagonal = (eigenvectors ** 2) @ (1.0 / denominator)
    return alpha, inverse_diagonal


def _inner_score(
    targets: np.ndarray,
    scale: np.ndarray,
    alpha: np.ndarray,
    inverse_diagonal: np.ndarray,
    n_a: int,
) -> float:
    # Kernel-ridge analytic leave-one-level-out residual: alpha_i / A^{-1}_{ii}.
    residual = alpha / np.maximum(inverse_diagonal, 1e-14)
    residual = residual * scale[None, :]
    main_error = float(np.mean(np.abs(residual[:, 0])))
    a_error = _pairwise_mean_absolute(residual[:, 1:1 + n_a])
    x_error = _pairwise_mean_absolute(residual[:, 1 + n_a:])
    return (main_error + a_error + x_error) / 3.0


def select_hyperparameters(
    targets: np.ndarray,
    z: np.ndarray,
    stage1_variance: np.ndarray,
    n_a: int,
) -> tuple[Hyperparameters, pd.DataFrame]:
    """Training-only analytic leave-one-B-level-out selection."""
    location = targets.mean(axis=0)
    scale = np.where(targets.std(axis=0) > 0, targets.std(axis=0), 1.0)
    y_standardised = (targets - location) / scale
    variance_standardised = stage1_variance / np.maximum(scale ** 2, 1e-14)
    rows = []
    best = None
    for gamma in GAMMAS:
        for rho in RHOS:
            _, eigenvalues, eigenvectors = _eigendecomposition(z, gamma, rho)
            for multiplier in UNCERTAINTY_MULTIPLIERS:
                noise = np.maximum(multiplier * variance_standardised, NOISE_FLOOR)
                alpha, inverse_diagonal = _posterior_coefficients(
                    y_standardised, eigenvalues, eigenvectors, noise)
                score = _inner_score(targets, scale, alpha, inverse_diagonal, n_a)
                row = {
                    "gamma": gamma,
                    "rho": rho,
                    "uncertainty_multiplier": multiplier,
                    "inner_score": score,
                }
                rows.append(row)
                key = (score, gamma, rho, multiplier)
                if best is None or key < best[0]:
                    best = (key, Hyperparameters(gamma, rho, multiplier, score))
    return best[1], pd.DataFrame(rows)


@dataclass
class SemiparametricBModel:
    prop: str
    desc_b: ElementalDescriptors
    a_levels: list
    b_levels: list
    x_levels: list
    ax_surface: np.ndarray
    triple_variance: float
    target_location: np.ndarray
    target_scale: np.ndarray
    stage1_variance: np.ndarray
    z_train: np.ndarray
    descriptor_mean: np.ndarray
    descriptor_scale: np.ndarray
    hyperparameters: Hyperparameters
    covariance: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    noise: np.ndarray
    alpha: np.ndarray
    inner_grid: pd.DataFrame

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        prop: str,
        desc_b: ElementalDescriptors,
    ) -> "SemiparametricBModel":
        targets, ax_surface, triple, a_levels, b_levels, x_levels = \
            decompose_b_surfaces(df, prop)
        raw_desc = _descriptor_matrix(desc_b, b_levels)
        z_train, descriptor_mean, descriptor_scale = _standardise(raw_desc)
        triple_variance = float(np.mean(triple ** 2))
        n_a, n_x = len(a_levels), len(x_levels)
        stage1_variance = np.concatenate([
            [triple_variance / (n_a * n_x)],
            np.repeat(triple_variance / n_x, n_a),
            np.repeat(triple_variance / n_a, n_x),
        ])
        hyperparameters, inner_grid = select_hyperparameters(
            targets, z_train, stage1_variance, n_a)
        target_location = targets.mean(axis=0)
        target_scale = np.where(targets.std(axis=0) > 0, targets.std(axis=0), 1.0)
        y_standardised = (targets - target_location) / target_scale
        covariance, eigenvalues, eigenvectors = _eigendecomposition(
            z_train, hyperparameters.gamma, hyperparameters.rho)
        variance_standardised = stage1_variance / np.maximum(target_scale ** 2, 1e-14)
        noise = np.maximum(
            hyperparameters.uncertainty_multiplier * variance_standardised, NOISE_FLOOR)
        alpha, _ = _posterior_coefficients(
            y_standardised, eigenvalues, eigenvectors, noise)
        return cls(
            prop=prop,
            desc_b=desc_b,
            a_levels=a_levels,
            b_levels=b_levels,
            x_levels=x_levels,
            ax_surface=ax_surface,
            triple_variance=triple_variance,
            target_location=target_location,
            target_scale=target_scale,
            stage1_variance=stage1_variance,
            z_train=z_train,
            descriptor_mean=descriptor_mean,
            descriptor_scale=descriptor_scale,
            hyperparameters=hyperparameters,
            covariance=covariance,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            noise=noise,
            alpha=alpha,
            inner_grid=inner_grid,
        )

    def _query(self, labels: Sequence):
        labels = list(labels)
        raw = _descriptor_matrix(self.desc_b, labels)
        z_query, _, _ = _standardise(
            raw, self.descriptor_mean, self.descriptor_scale)
        gamma = self.hyperparameters.gamma
        rho = self.hyperparameters.rho
        cross = descriptor_kernel(z_query, self.z_train, gamma)
        train_index = {level: i for i, level in enumerate(self.b_levels)}
        for qi, level in enumerate(labels):
            if level in train_index:
                cross[qi, train_index[level]] += rho
        self_covariance = descriptor_kernel(z_query, z_query, gamma) + rho * np.eye(len(labels))
        mean_standardised = cross @ self.alpha
        mean = self.target_location + mean_standardised * self.target_scale
        covariances = []
        projected_cross = self.eigenvectors.T @ cross.T
        for output, noise in enumerate(self.noise):
            solved = self.eigenvectors @ (
                projected_cross / (self.eigenvalues[:, None] + noise))
            covariance = self_covariance - cross @ solved
            covariance = 0.5 * (covariance + covariance.T)
            covariance *= self.target_scale[output] ** 2
            covariances.append(covariance)
        return mean, covariances, z_query

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        labels = sorted(df.site_B.unique().tolist())
        means, _, _ = self._query(labels)
        bi = {level: i for i, level in enumerate(labels)}
        ai = {level: i for i, level in enumerate(self.a_levels)}
        xi = {level: i for i, level in enumerate(self.x_levels)}
        n_a = len(self.a_levels)
        out = np.empty(len(df), dtype=float)
        for k, row in enumerate(df.itertuples(index=False)):
            b = bi[row.site_B]
            a = ai[row.site_A]
            x = xi[row.site_X]
            out[k] = (means[b, 0] + means[b, 1 + a]
                      + means[b, 1 + n_a + x] + self.ax_surface[a, x])
        return out

    def predict_edges(self, src: pd.DataFrame, dst: pd.DataFrame):
        labels = sorted(set(src.site_B).union(set(dst.site_B)))
        means, covariances, _ = self._query(labels)
        bi = {level: i for i, level in enumerate(labels)}
        ai = {level: i for i, level in enumerate(self.a_levels)}
        xi = {level: i for i, level in enumerate(self.x_levels)}
        n_a = len(self.a_levels)

        def node_mean(row):
            b, a, x = bi[row.site_B], ai[row.site_A], xi[row.site_X]
            return (means[b, 0] + means[b, 1 + a]
                    + means[b, 1 + n_a + x] + self.ax_surface[a, x])

        delta_mean = np.empty(len(src), dtype=float)
        delta_variance = np.empty(len(src), dtype=float)
        for k, (s, d) in enumerate(zip(src.itertuples(index=False),
                                       dst.itertuples(index=False))):
            bs, bd = bi[s.site_B], bi[d.site_B]
            a_s, a_d = ai[s.site_A], ai[d.site_A]
            x_s, x_d = xi[s.site_X], xi[d.site_X]
            delta_mean[k] = node_mean(d) - node_mean(s)

            src_outputs = (0, 1 + a_s, 1 + n_a + x_s)
            dst_outputs = (0, 1 + a_d, 1 + n_a + x_d)
            var_s = sum(covariances[j][bs, bs] for j in src_outputs)
            var_d = sum(covariances[j][bd, bd] for j in dst_outputs)
            cross = covariances[0][bs, bd]
            if a_s == a_d:
                cross += covariances[1 + a_s][bs, bd]
            if x_s == x_d:
                cross += covariances[1 + n_a + x_s][bs, bd]
            # The second-order surface does not resolve the three-factor field.  Treat its
            # training variance as response-level projection uncertainty for each endpoint.
            delta_variance[k] = max(
                var_s + var_d - 2.0 * cross + 2.0 * self.triple_variance, 1e-12)
        return delta_mean, delta_variance

    def effective_degrees_of_freedom(self) -> float:
        values = []
        for noise in self.noise:
            values.append(float(np.sum(self.eigenvalues / (self.eigenvalues + noise))))
        return float(np.mean(values))

    def support(self, labels: Sequence) -> pd.DataFrame:
        labels = list(labels)
        _, posterior_covariances, z_query = self._query(labels)
        gamma = self.hyperparameters.gamma
        descriptor_covariance = descriptor_kernel(self.z_train, self.z_train, gamma)
        descriptor_inverse = np.linalg.pinv(descriptor_covariance + 1e-8 * np.eye(len(self.z_train)))
        rows = []
        for query_index, (level, z) in enumerate(zip(labels, z_query)):
            distance = np.sqrt(np.sum((self.z_train - z) ** 2, axis=1))
            k = descriptor_kernel(z[None, :], self.z_train, gamma).ravel()
            k_self = float(descriptor_kernel(z[None, :], z[None, :], gamma)[0, 0])
            leverage = float(np.clip(k @ descriptor_inverse @ k / max(k_self, 1e-12), 0.0, 1.0))
            prior_variance = k_self + self.hyperparameters.rho
            posterior_ratio = float(np.mean([
                covariance[query_index, query_index]
                / max(prior_variance * self.target_scale[output] ** 2, 1e-14)
                for output, covariance in enumerate(posterior_covariances)
            ]))
            rows.append({
                "held_level": level,
                "knn_distance": float(distance.min()),
                "kernel_leverage": leverage,
                "descriptor_instability": 1.0 - leverage,
                "posterior_variance_ratio": posterior_ratio,
                "extrapolation_instability": posterior_ratio,
            })
        return pd.DataFrame(rows)


@dataclass
class LinearHybridBModel:
    """The current hard switch expressed in the same B-surface coordinates."""

    prop: str
    desc_b: ElementalDescriptors
    a_levels: list
    b_levels: list
    x_levels: list
    targets: np.ndarray
    ax_surface: np.ndarray
    descriptor_mean: np.ndarray
    descriptor_scale: np.ndarray
    coefficients: np.ndarray
    lam: float = 1e-2

    @classmethod
    def fit(cls, df: pd.DataFrame, prop: str, desc_b: ElementalDescriptors, lam: float = 1e-2):
        targets, ax_surface, _, a_levels, b_levels, x_levels = decompose_b_surfaces(df, prop)
        raw = _descriptor_matrix(desc_b, b_levels)
        z, mean, scale = _standardise(raw)
        design = np.hstack([np.ones((len(z), 1)), z])
        coefficients = np.linalg.lstsq(
            design.T @ design + lam * np.eye(design.shape[1]),
            design.T @ targets,
            rcond=None,
        )[0]
        return cls(prop, desc_b, a_levels, b_levels, x_levels, targets, ax_surface,
                   mean, scale, coefficients, lam)

    def _components(self, labels: Sequence):
        labels = list(labels)
        raw = _descriptor_matrix(self.desc_b, labels)
        z, _, _ = _standardise(raw, self.descriptor_mean, self.descriptor_scale)
        mapped = np.hstack([np.ones((len(z), 1)), z]) @ self.coefficients
        train_index = {level: i for i, level in enumerate(self.b_levels)}
        for i, level in enumerate(labels):
            if level in train_index:
                mapped[i] = self.targets[train_index[level]]
        return mapped

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        labels = sorted(df.site_B.unique().tolist())
        components = self._components(labels)
        bi = {level: i for i, level in enumerate(labels)}
        ai = {level: i for i, level in enumerate(self.a_levels)}
        xi = {level: i for i, level in enumerate(self.x_levels)}
        n_a = len(self.a_levels)
        out = np.empty(len(df), dtype=float)
        for k, row in enumerate(df.itertuples(index=False)):
            b, a, x = bi[row.site_B], ai[row.site_A], xi[row.site_X]
            out[k] = (components[b, 0] + components[b, 1 + a]
                      + components[b, 1 + n_a + x] + self.ax_surface[a, x])
        return out
