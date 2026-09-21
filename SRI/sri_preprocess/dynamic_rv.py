# -*- coding: utf-8 -*-
"""Measurement-based radial-velocity evidence for the SRI pipeline.

This module deliberately contains no import from another SRI branch.  It does
not alter an observed RV and it never performs an effective-RV
shrinkage.  The source-level route uses positive member RV errors; the
entity-level fallback uses the node error already present in the node table.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RVEvidence:
    """Scalar RV information for one scoring entity."""

    information: float = 0.0
    n_eff: float = 0.0
    n_valid: int = 0
    center_median: float = np.nan
    observed_scatter: float = np.nan
    expected_scatter: float = np.nan
    intrinsic_scatter: float = np.nan
    sigma_sys: float = np.nan
    precision: float = 0.0
    source_level: bool = False

    def as_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


def _as_float(values) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)


def _robust_scatter(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    centre = float(np.median(values))
    scatter = float(1.4826 * np.median(np.abs(values - centre)))
    return centre, scatter


def compute_rv_evidence(
    rv_values,
    rv_errors,
    *,
    fallback_error: float | None = None,
) -> RVEvidence:
    """Compute measurement-based RV evidence without empirical ranks.

    Source-level errors are used whenever at least one pair (finite RV,
    positive error) exists.  The point estimate is not changed: the median is
    used only as a robust centre for the scatter diagnostic.  If no source
    errors are available, a finite positive entity/node error supplies the
    precision fallback; this fallback does not invent a member-level scatter.
    """
    rv = _as_float(rv_values)
    err = _as_float(rv_errors)
    from .uncertainty_mode import ENABLED
    if not ENABLED.get():
        values=rv[np.isfinite(rv)]
        centre,scatter=_robust_scatter(values)
        return RVEvidence(n_valid=len(values),center_median=centre,observed_scatter=scatter,source_level=True)
    valid_source = np.isfinite(rv) & np.isfinite(err) & (err > 0)
    if np.any(valid_source):
        rv_v = rv[valid_source]
        err_v = err[valid_source]
        weights = 1.0 / np.square(err_v)
        information = float(np.sum(weights))
        weight_sum = float(np.sum(weights))
        n_eff = (
            float(weight_sum**2 / np.sum(np.square(weights)))
            if np.sum(np.square(weights)) > 0
            else 0.0
        )
        centre, observed = _robust_scatter(rv_v)
        expected = float(np.sqrt(np.median(np.square(err_v))))
        intrinsic2 = max(0.0, observed**2 - expected**2)
        intrinsic = float(np.sqrt(intrinsic2))
        # Separate the two uncertainty sources.  ``information`` carries the
        # heteroscedastic measurement-error contribution, while the robust
        # intrinsic scatter is averaged over the actual number of independent
        # valid RV sources.  ``n_eff`` remains an auditable diagnostic, but is
        # not used as a second precision multiplier here: doing so would let a
        # single very precise source dominate both terms.
        n_valid = int(len(rv_v))
        sigma2 = (1.0 / information) + (
            intrinsic2 / n_valid if n_valid > 0 else 0.0
        )
        sigma = float(np.sqrt(sigma2)) if sigma2 >= 0 else np.nan
        precision = float(1.0 / sigma2) if sigma2 > 0 else 0.0
        return RVEvidence(
            information=information,
            n_eff=n_eff,
            n_valid=n_valid,
            center_median=centre,
            observed_scatter=observed,
            expected_scatter=expected,
            intrinsic_scatter=intrinsic,
            sigma_sys=sigma,
            precision=precision,
            source_level=True,
        )

    valid_rv = np.isfinite(rv)
    fallback = float(fallback_error) if fallback_error is not None else np.nan
    if np.isfinite(fallback) and fallback > 0 and np.any(valid_rv):
        centre, observed = _robust_scatter(rv[valid_rv])
        precision = float(1.0 / fallback**2)
        return RVEvidence(
            information=precision,
            n_eff=1.0,
            n_valid=int(valid_rv.sum()),
            center_median=centre,
            observed_scatter=observed,
            expected_scatter=np.nan,
            intrinsic_scatter=np.nan,
            sigma_sys=fallback,
            precision=precision,
            source_level=False,
        )
    return RVEvidence(n_valid=int(valid_rv.sum()))


def pair_precision(precision_i: float, precision_j: float) -> float:
    """Precision of a difference between two independent scalar estimates."""
    pi = float(precision_i)
    pj = float(precision_j)
    if not (np.isfinite(pi) and np.isfinite(pj) and pi > 0 and pj > 0):
        return 0.0
    return float(1.0 / (1.0 / pi + 1.0 / pj))


def _weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.nan
    return float(np.sqrt(np.average(np.square(values[valid]), weights=weights[valid])))


def _mean_unit_direction(ra_deg, dec_deg) -> np.ndarray | None:
    ra = _as_float(ra_deg)
    dec = _as_float(dec_deg)
    valid = np.isfinite(ra) & np.isfinite(dec)
    if not np.any(valid):
        return None
    ra_r = np.radians(ra[valid])
    dec_r = np.radians(dec[valid])
    unit = np.column_stack([
        np.cos(dec_r) * np.cos(ra_r),
        np.cos(dec_r) * np.sin(ra_r),
        np.sin(dec_r),
    ])
    direction = np.mean(unit, axis=0)
    norm = float(np.linalg.norm(direction))
    return direction / norm if norm > 0 else None


def entity_rv_covariances(
    node_rows: pd.DataFrame,
    entity_members: dict[str, pd.DataFrame],
    components: list[list[str]],
    icrs_to_galactic: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Return entity UVW centre-covariance matrices and RV diagnostics.

    The scoring entities are fixed *before* this uncertainty calculation.
    For every base node we construct a 3-D centre covariance from the node's
    source-level/systemic RV, RA-tangent and Dec-tangent centre errors at that
    node's own sightline.  When an entity contains multiple nodes, the node
    centre covariances are propagated through the *same member-count weighted
    centre estimator used by :func:`sri_preprocess.entities.aggregate_group`*::

        Sigma_entity = sum_j alpha_j**2 Sigma_j,
        alpha_j = N_j / sum_k N_k.

    Thus contraction does not reinterpret between-node velocity offsets as
    measurement uncertainty.  A singleton entity is unchanged by construction.
    The entity-level pooled RV evidence is retained only as an auditable
    diagnostic; it does not replace the propagated full 3-D covariance.
    """
    transform = np.asarray(icrs_to_galactic, float)
    if transform.shape != (3, 3):
        raise ValueError("icrs_to_galactic must be a 3x3 matrix")
    covariances: list[np.ndarray] = []
    diagnostics: list[dict[str, float | int | bool | str]] = []

    def numeric_column(frame: pd.DataFrame, name: str) -> np.ndarray:
        if name not in frame:
            return np.full(len(frame), np.nan, dtype=float)
        return pd.to_numeric(frame[name], errors="coerce").to_numpy(float)

    def local_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y, z = direction
        dec = np.arcsin(np.clip(z, -1.0, 1.0))
        ra = np.arctan2(y, x)
        radial = direction
        ra_hat = np.array([-np.sin(ra), np.cos(ra), 0.0])
        dec_hat = np.array([
            -np.sin(dec) * np.cos(ra),
            -np.sin(dec) * np.sin(ra),
            np.cos(dec),
        ])
        return radial, ra_hat, dec_hat

    def node_covariance(row: pd.Series, members: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
        direction = None
        if not members.empty:
            direction = _mean_unit_direction(
                members.get("ra", pd.Series(dtype=float)),
                members.get("dec", pd.Series(dtype=float)),
            )
        if direction is None:
            direction = _mean_unit_direction(
                pd.Series([row.get("RA", np.nan)]),
                pd.Series([row.get("DEC", np.nan)]),
            )
        e_rv = float(pd.to_numeric(pd.Series([row.get("e_RV", np.nan)]), errors="coerce").iloc[0])
        e_vra = float(pd.to_numeric(pd.Series([row.get("e_Vra", np.nan)]), errors="coerce").iloc[0])
        e_vdec = float(pd.to_numeric(pd.Series([row.get("e_Vdec", np.nan)]), errors="coerce").iloc[0])
        if direction is None or not (
            np.isfinite(e_rv) and e_rv > 0
            and np.isfinite(e_vra) and e_vra > 0
            and np.isfinite(e_vdec) and e_vdec > 0
        ):
            return None
        radial, ra_hat, dec_hat = local_basis(direction)
        cov_tangent_icrs = (
            e_vra**2 * np.outer(ra_hat, ra_hat)
            + e_vdec**2 * np.outer(dec_hat, dec_hat)
        )
        cov_icrs = e_rv**2 * np.outer(radial, radial) + cov_tangent_icrs
        cov_gal = transform @ cov_icrs @ transform.T
        tangent_gal = transform @ cov_tangent_icrs @ transform.T
        return (cov_gal + cov_gal.T) / 2.0, (tangent_gal + tangent_gal.T) / 2.0

    node = node_rows.copy()
    node["id_node"] = node["id_node"].astype(str)

    for component in components:
        entity_id = "+".join(component)
        members = entity_members.get(entity_id, pd.DataFrame())
        node_part = node[node["id_node"].isin([str(x) for x in component])].copy()
        node_n = pd.to_numeric(node_part.get("N"), errors="coerce").to_numpy(float)
        node_n = np.where(np.isfinite(node_n) & (node_n > 0), node_n, 1.0)

        # Keep the pooled entity RV evidence for diagnostics/backward-readable
        # output only.  The nominal 3-D covariance below is propagated from the
        # already-estimated node centre covariances after contraction is fixed.
        node_rv_err = pd.to_numeric(node_part.get("e_RV"), errors="coerce").to_numpy(float)
        fallback_error = _weighted_rms(node_rv_err, node_n)
        if not members.empty:
            rv_values = members.get("radial_velocity", pd.Series(dtype=float))
            rv_errors = members.get("radial_velocity_error", pd.Series(dtype=float))
            evidence = compute_rv_evidence(rv_values, rv_errors, fallback_error=fallback_error)
            entity_direction = _mean_unit_direction(
                members.get("ra", pd.Series(dtype=float)),
                members.get("dec", pd.Series(dtype=float)),
            )
        else:
            evidence = compute_rv_evidence([], [], fallback_error=fallback_error)
            entity_direction = None
        if entity_direction is None:
            entity_direction = _mean_unit_direction(
                node_part.get("RA", pd.Series(dtype=float)),
                node_part.get("DEC", pd.Series(dtype=float)),
            )

        node_covs: list[np.ndarray] = []
        node_tangent_covs: list[np.ndarray] = []
        node_weights: list[float] = []
        for (_, row), weight in zip(node_part.iterrows(), node_n):
            node_id = str(row["id_node"])
            if not members.empty and "id_node" in members:
                node_members = members[members["id_node"].astype(str).eq(node_id)]
            else:
                node_members = pd.DataFrame()
            cov_pair = node_covariance(row, node_members)
            if cov_pair is None:
                continue
            cov_node, tangent_node = cov_pair
            node_covs.append(cov_node)
            node_tangent_covs.append(tangent_node)
            node_weights.append(float(weight))

        if len(node_covs) != len(node_part) or not node_covs:
            covariances.append(np.full((3, 3), np.nan))
            diagnostics.append({
                "entity": entity_id,
                **evidence.as_dict(),
                "e_vra_eff": np.nan,
                "e_vdec_eff": np.nan,
                "covariance_available": False,
                "covariance_mode": "post_contraction_node_covariance_propagation",
                "covariance_node_count": int(len(node_covs)),
            })
            continue

        weights = np.asarray(node_weights, float)
        weights = weights / np.sum(weights)
        cov_gal = np.zeros((3, 3), dtype=float)
        tangent_gal = np.zeros((3, 3), dtype=float)
        for alpha, cov_node, tangent_node in zip(weights, node_covs, node_tangent_covs):
            factor = float(alpha**2)
            cov_gal += factor * cov_node
            tangent_gal += factor * tangent_node
        cov_gal = (cov_gal + cov_gal.T) / 2.0
        tangent_gal = (tangent_gal + tangent_gal.T) / 2.0

        # Report local-direction effective standard deviations derived from the
        # propagated covariance.  They are diagnostics only; the scorer uses
        # the full UVW covariance matrix including off-diagonal terms.
        e_vra_eff = np.nan
        e_vdec_eff = np.nan
        e_rv_eff = np.nan
        if entity_direction is not None and np.isfinite(cov_gal).all():
            radial, ra_hat, dec_hat = local_basis(entity_direction)
            cov_icrs_eff = transform.T @ cov_gal @ transform
            e_rv_eff = float(np.sqrt(max(0.0, radial @ cov_icrs_eff @ radial)))
            e_vra_eff = float(np.sqrt(max(0.0, ra_hat @ cov_icrs_eff @ ra_hat)))
            e_vdec_eff = float(np.sqrt(max(0.0, dec_hat @ cov_icrs_eff @ dec_hat)))

        covariances.append(cov_gal)
        diagnostics.append({
            "entity": entity_id,
            **evidence.as_dict(),
            "e_vra_eff": e_vra_eff,
            "e_vdec_eff": e_vdec_eff,
            "e_rv_propagated": e_rv_eff,
            "covariance_available": True,
            "tangent_covariance_gal": tangent_gal,
            "covariance_mode": "post_contraction_node_covariance_propagation",
            "covariance_node_count": int(len(node_covs)),
        })
    return np.asarray(covariances, float), diagnostics

def standardized_pairwise_distance(
    points: np.ndarray,
    covariances: np.ndarray,
) -> np.ndarray:
    """Mahalanobis pairwise distances using independent entity covariances."""
    values = np.asarray(points, float)
    cov = np.asarray(covariances, float)
    if values.ndim != 2 or cov.shape != (len(values), values.shape[1], values.shape[1]):
        raise ValueError("points/covariances shape mismatch")
    n = len(values)
    distance = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        if not np.isfinite(values[i]).all() or not np.isfinite(cov[i]).all():
            continue
        distance[i, i] = 0.0
        for j in range(i + 1, n):
            if not np.isfinite(values[j]).all() or not np.isfinite(cov[j]).all():
                continue
            # For independent estimates, Cov(v_i-v_j) = Sigma_i + Sigma_j.
            # Dividing by two would artificially inflate standardized gaps.
            sigma = cov[i] + cov[j]
            sigma = (sigma + sigma.T) / 2.0
            delta = values[i] - values[j]
            try:
                chi2 = float(delta @ np.linalg.pinv(sigma, hermitian=True) @ delta)
            except TypeError:  # older NumPy without hermitian keyword
                chi2 = float(delta @ np.linalg.pinv(sigma) @ delta)
            distance[i, j] = distance[j, i] = np.sqrt(max(chi2, 0.0))
    return distance
