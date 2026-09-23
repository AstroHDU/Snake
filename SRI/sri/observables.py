from __future__ import annotations

"""Derive SRI working coordinates directly from basic astrometric observables.

The public raw-input path intentionally does *not* trust pre-computed X/Y/Z or
Vra/Vdec columns.  Those quantities are reconstructed from the member-level
observables so that the package can be run from catalogue-like inputs.

Conventions
-----------
* ra, dec: degrees (ICRS)
* parallax: milliarcseconds
* pmra, pmdec: milliarcseconds / year; pmra is Gaia's cos(dec)-included value
* radial_velocity, radial_velocity_error: km/s
* distance: 1000/parallax pc, therefore non-positive parallaxes are unusable
* X,Y,Z: heliocentric Galactic Cartesian pc
* Vra,Vdec: heliocentric tangential km/s in the local ICRS tangent basis

No solar-motion correction is applied here, matching the SRI coordinate layer.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd

from sri_preprocess.coordinates import K_TAN, ICRS_TO_GALACTIC
from sri_preprocess.dynamic_rv import compute_rv_evidence


BASIC_MEMBER_COLUMNS = (
    "Snake", "id_part", "id_node", "source_id",
    "ra", "dec", "parallax", "pmra", "pmdec",
    "radial_velocity", "radial_velocity_error",
)

OPTIONAL_ASTROMETRIC_ERROR_COLUMNS = (
    "parallax_error", "pmra_error", "pmdec_error",
    "parallax_pmra_corr", "parallax_pmdec_corr",
)

DERIVED_MEMBER_COLUMNS = (
    "distance_pc", "l", "b", "X", "Y", "Z", "Vra", "Vdec",
    "e_Vra_source", "e_Vdec_source",
)


def _num(frame: pd.DataFrame, name: str) -> np.ndarray:
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(float)


def _icrs_unit(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.radians(np.asarray(ra_deg, float))
    dec = np.radians(np.asarray(dec_deg, float))
    return np.column_stack([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec),
    ])


def _galactic_lonlat(gal_unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(gal_unit, float)
    l = np.degrees(np.arctan2(g[:, 1], g[:, 0])) % 360.0
    b = np.degrees(np.arcsin(np.clip(g[:, 2], -1.0, 1.0)))
    return l, b


def _source_tangent_error(
    pm: np.ndarray,
    parallax: np.ndarray,
    pm_error: np.ndarray,
    parallax_error: np.ndarray,
    corr_pm_parallax: np.ndarray,
) -> np.ndarray:
    """First-order uncertainty of K*pm/parallax.

    Correlation is optional.  Missing correlation is treated as zero; missing
    measurement errors yield NaN rather than an invented floor.
    """
    pm = np.asarray(pm, float)
    plx = np.asarray(parallax, float)
    epm = np.asarray(pm_error, float)
    eplx = np.asarray(parallax_error, float)
    corr = np.asarray(corr_pm_parallax, float)
    out = np.full(np.broadcast(pm, plx, epm, eplx, corr).shape, np.nan, float)
    valid = (
        np.isfinite(pm) & np.isfinite(plx) & (plx > 0)
        & np.isfinite(epm) & (epm >= 0)
        & np.isfinite(eplx) & (eplx >= 0)
    )
    if not np.any(valid):
        return out
    c = np.where(np.isfinite(corr), np.clip(corr, -1.0, 1.0), 0.0)
    d_pm = K_TAN / plx[valid]
    d_plx = -K_TAN * pm[valid] / np.square(plx[valid])
    cov = c[valid] * epm[valid] * eplx[valid]
    var = (
        np.square(d_pm * epm[valid])
        + np.square(d_plx * eplx[valid])
        + 2.0 * d_pm * d_plx * cov
    )
    out[valid] = np.sqrt(np.maximum(var, 0.0))
    return out


def derive_member_observables(members: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with all SRI derived member coordinates recomputed.

    Any pre-existing X/Y/Z/Vra/Vdec columns are overwritten deliberately.  This
    prevents a stale or differently-defined derived table from silently changing
    the public release results.
    """
    missing = [c for c in BASIC_MEMBER_COLUMNS if c not in members.columns]
    if missing:
        raise ValueError(f"member table missing basic input columns: {missing}")

    m = members.copy()
    # Identifiers are kept as text to avoid source-id precision loss.
    m["source_id"] = m["source_id"].astype(str)
    m["id_node"] = m["id_node"].where(m["id_node"].notna(), pd.NA).astype("string")
    m["id_part"] = m["id_part"].where(m["id_part"].notna(), pd.NA).astype("string")
    m["Snake"] = pd.to_numeric(m["Snake"], errors="raise").astype(int)

    ra = _num(m, "ra")
    dec = _num(m, "dec")
    plx = _num(m, "parallax")
    pmra = _num(m, "pmra")
    pmdec = _num(m, "pmdec")

    # K*mu/parallax is valid when parallax is positive and finite.
    valid_plx = np.isfinite(plx) & (plx > 0)
    distance = np.full(len(m), np.nan, float)
    distance[valid_plx] = 1000.0 / plx[valid_plx]
    vra = np.full(len(m), np.nan, float)
    vdec = np.full(len(m), np.nan, float)
    vra[valid_plx & np.isfinite(pmra)] = K_TAN * pmra[valid_plx & np.isfinite(pmra)] / plx[valid_plx & np.isfinite(pmra)]
    vdec[valid_plx & np.isfinite(pmdec)] = K_TAN * pmdec[valid_plx & np.isfinite(pmdec)] / plx[valid_plx & np.isfinite(pmdec)]

    icrs = _icrs_unit(ra, dec)
    gal = icrs @ ICRS_TO_GALACTIC.T
    xyz = gal * distance[:, None]
    l, b = _galactic_lonlat(gal)

    m["distance_pc"] = distance
    m["l"] = l
    m["b"] = b
    m[["X", "Y", "Z"]] = xyz
    m["Vra"] = vra
    m["Vdec"] = vdec

    # Source-level tangent errors are required by strict uncertainty mode.
    # Nominal mode can omit them; it is selected explicitly by the runner.
    if {"parallax_error", "pmra_error"}.issubset(m.columns):
        corr = _num(m, "parallax_pmra_corr") if "parallax_pmra_corr" in m else np.zeros(len(m))
        m["e_Vra_source"] = _source_tangent_error(
            pmra, plx, _num(m, "pmra_error"), _num(m, "parallax_error"), corr
        )
    else:
        m["e_Vra_source"] = np.nan
    if {"parallax_error", "pmdec_error"}.issubset(m.columns):
        corr = _num(m, "parallax_pmdec_corr") if "parallax_pmdec_corr" in m else np.zeros(len(m))
        m["e_Vdec_source"] = _source_tangent_error(
            pmdec, plx, _num(m, "pmdec_error"), _num(m, "parallax_error"), corr
        )
    else:
        m["e_Vdec_source"] = np.nan
    return m


def _robust_scatter(values: np.ndarray) -> tuple[float, float]:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    med = float(np.median(x))
    mad = float(1.4826 * np.median(np.abs(x - med)))
    return med, mad


def _systemic_error(values: np.ndarray, source_errors: np.ndarray | None = None) -> float:
    """Approximate uncertainty of a robust node centre.

    When source uncertainties are available, use the same transparent
    measurement+intrinsic decomposition used for the RV evidence layer.  When
    they are unavailable, use robust scatter/sqrt(N) as a fallback *diagnostic*
    centre error.  This quantity does not alter the node centre point estimate.
    """
    x = np.asarray(values, float)
    finite_x = np.isfinite(x)
    if not np.any(finite_x):
        return np.nan
    x = x[finite_x]
    med, obs = _robust_scatter(x)
    if source_errors is not None:
        e = np.asarray(source_errors, float)[finite_x]
        good = np.isfinite(e) & (e > 0)
        if np.any(good):
            info = float(np.sum(1.0 / np.square(e[good])))
            expected = float(np.sqrt(np.median(np.square(e[good]))))
            intrinsic2 = max(0.0, obs * obs - expected * expected)
            n = int(len(x))
            return float(np.sqrt((1.0 / info) + intrinsic2 / max(n, 1)))
    if len(x) == 1:
        return np.nan
    return float(obs / np.sqrt(len(x)))


def derive_node_table(members: pd.DataFrame) -> pd.DataFrame:
    """Build the node summaries required by the SRI from member observations.

    Point estimates are transparent componentwise robust summaries of the same
    member-derived quantities used by the pipeline.  No pre-computed Vra/Vdec,
    X/Y/Z, or node RV is required from the user.
    """
    if not set(DERIVED_MEMBER_COLUMNS[:8]).issubset(members.columns):
        raise ValueError("derive_member_observables must be called first")
    rows: list[dict[str, object]] = []
    node_members = members.loc[members["id_node"].notna()].copy()
    if node_members.empty:
        raise ValueError("member table contains no rows assigned to id_node")

    for (snake, node_id), g in node_members.groupby(["Snake", "id_node"], sort=True):
        # Keep only finite rows for each coordinate separately; medians are the
        # Node-centre convention used by the SRI calculation.
        row: dict[str, object] = {
            "Snake": int(snake),
            "id_node": str(node_id),
            "N": int(len(g)),
        }
        for src, dst in [
            ("ra", "RA"), ("dec", "DEC"), ("parallax", "PLX"),
            ("pmra", "PMRA"), ("pmdec", "PMDE"),
            ("X", "X"), ("Y", "Y"), ("Z", "Z"),
            ("Vra", "Vra"), ("Vdec", "Vdec"),
        ]:
            vals = pd.to_numeric(g[src], errors="coerce").to_numpy(float)
            row[dst] = float(np.nanmedian(vals)) if np.isfinite(vals).any() else np.nan

        good_rv = (
            pd.to_numeric(g["radial_velocity"], errors="coerce").notna()
            & pd.to_numeric(g["radial_velocity_error"], errors="coerce").notna()
            & (pd.to_numeric(g["radial_velocity_error"], errors="coerce") > 0)
        )
        from sri_preprocess.uncertainty_mode import ENABLED
        if not ENABLED.get():good_rv=np.isfinite(pd.to_numeric(g['radial_velocity'],errors='coerce'))
        rv_g = g.loc[good_rv]
        evidence = compute_rv_evidence(
            rv_g["radial_velocity"].to_numpy(float),
            rv_g["radial_velocity_error"].to_numpy(float),
        )
        row["RV"] = evidence.center_median
        row["e_RV"] = evidence.sigma_sys
        row["N_RV"] = int(rv_g["source_id"].astype(str).nunique())
        row["f_RV"] = float(row["N_RV"] / row["N"]) if row["N"] else np.nan

        row["e_Vra"] = _systemic_error(
            pd.to_numeric(g["Vra"], errors="coerce").to_numpy(float),
            pd.to_numeric(g["e_Vra_source"], errors="coerce").to_numpy(float),
        )
        row["e_Vdec"] = _systemic_error(
            pd.to_numeric(g["Vdec"], errors="coerce").to_numpy(float),
            pd.to_numeric(g["e_Vdec_source"], errors="coerce").to_numpy(float),
        )
        parts = sorted(set(g["id_part"].dropna().astype(str)))
        row["id_part_list"] = ",".join(parts)
        rows.append(row)

    nodes = pd.DataFrame(rows)
    if nodes["id_node"].duplicated().any():
        raise ValueError("id_node must map to exactly one Snake in the raw input")
    nodes = nodes.sort_values(["Snake", "id_node"], kind="stable").reset_index(drop=True)
    nodes["rng_id"] = nodes["Snake"].astype(int)
    nodes["rv_rng_index"] = np.arange(len(nodes), dtype=int)
    return nodes


def audit_existing_derived(raw_members: pd.DataFrame, derived_members: pd.DataFrame) -> dict[str, float]:
    """Compare optional pre-existing derived columns with release recomputation."""
    result: dict[str, float] = {}
    for col in ("X", "Y", "Z", "Vra", "Vdec"):
        if col not in raw_members.columns:
            continue
        old = pd.to_numeric(raw_members[col], errors="coerce").to_numpy(float)
        new = pd.to_numeric(derived_members[col], errors="coerce").to_numpy(float)
        valid = np.isfinite(old) & np.isfinite(new)
        result[f"max_abs_{col}"] = float(np.max(np.abs(old[valid] - new[valid]))) if np.any(valid) else np.nan
    return result
