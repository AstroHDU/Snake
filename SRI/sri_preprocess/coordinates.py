from __future__ import annotations
import numpy as np
import pandas as pd
K_TAN = 4.74047
ICRS_TO_GALACTIC = np.asarray([
    [-0.0548755604, -0.8734370902, -0.4838350155],
    [0.4941094279, -0.4448296300, 0.7469822445],
    [-0.8676661490, -0.1980763734, 0.4559837762],
], dtype=float)
MEMBER_RV_TRANSPORT_MODE="projected"
REFERENCE_WEIGHT_MODE="node_equal"
VELOCITY_MODE="cartesian"
INCOMPLETE_RV_MODE="valid_rv_centroid"
from . import entities as ENT_MOD
MEMBER_SCORE_VELOCITY_COLUMNS=["Vtan1_member_transport","Vtan2_member_transport"]
def add_velocity(members, frame="raw"):
    if frame!="raw": raise ValueError("Only raw heliocentric input is supported")
    m=members.copy(); p=m.parallax.to_numpy(float)
    m["Vra"]=np.where(p>0,K_TAN*m.pmra.to_numpy(float)/p,np.nan)
    m["Vdec"]=np.where(p>0,K_TAN*m.pmdec.to_numpy(float)/p,np.nan)
    return m

def icrs_velocity_to_galactic(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    vra: np.ndarray,
    vdec: np.ndarray,
    rv: np.ndarray,
) -> np.ndarray:
    """Transform local ICRS velocity components to Galactic (U,V,W).

    ``vra`` is the cos(dec)-included tangential component associated with Gaia
    ``pmra``. Inputs are broadcast to one-dimensional arrays and invalid rows
    remain NaN. No solar-motion correction is applied: both transport modes remain
    heliocentric and differ only in their coordinate basis.
    """
    ra_deg, dec_deg, vra, vdec, rv = np.broadcast_arrays(
        np.asarray(ra_deg, float),
        np.asarray(dec_deg, float),
        np.asarray(vra, float),
        np.asarray(vdec, float),
        np.asarray(rv, float),
    )
    ra = np.radians(ra_deg.ravel())
    dec = np.radians(dec_deg.ravel())
    vra = vra.ravel(); vdec = vdec.ravel(); rv = rv.ravel()
    valid = (
        np.isfinite(ra) & np.isfinite(dec) & np.isfinite(vra)
        & np.isfinite(vdec) & np.isfinite(rv)
    )
    result = np.full((len(ra), 3), np.nan, dtype=float)
    if not np.any(valid):
        return result
    av = ra[valid]; dv = dec[valid]
    radial_hat = np.column_stack([
        np.cos(dv) * np.cos(av),
        np.cos(dv) * np.sin(av),
        np.sin(dv),
    ])
    ra_hat = np.column_stack([
        -np.sin(av),
        np.cos(av),
        np.zeros(len(av)),
    ])
    dec_hat = np.column_stack([
        -np.sin(dv) * np.cos(av),
        -np.sin(dv) * np.sin(av),
        np.cos(dv),
    ])
    icrs = (
        rv[valid, None] * radial_hat
        + vra[valid, None] * ra_hat
        + vdec[valid, None] * dec_hat
    )
    result[valid] = icrs @ ICRS_TO_GALACTIC.T
    return result

def rv_valid_mask(node_rows: pd.DataFrame) -> np.ndarray:
    """Production-valid RV mask: finite RV/e_RV with e_RV strictly positive."""
    rv = pd.to_numeric(node_rows["RV"], errors="coerce").to_numpy(float)
    from .uncertainty_mode import ENABLED
    if not ENABLED.get():return np.isfinite(rv)
    error = pd.to_numeric(node_rows["e_RV"], errors="coerce").to_numpy(float)
    return np.isfinite(rv) & np.isfinite(error) & (error > 0)

def rv_complete_rows(node_rows: pd.DataFrame) -> bool:
    """Production-valid RV completeness: every node has finite RV/e_RV, e_RV>0."""
    return bool(len(node_rows) > 0 and np.all(rv_valid_mask(node_rows)))

def rv_valid_entity_mask(
    node_rows: pd.DataFrame,
    components: list[list[str]],
) -> np.ndarray:
    """Return one validity flag per scoring entity using real node RV evidence."""
    valid_nodes = set(
        node_rows.loc[rv_valid_mask(node_rows), "id_node"].astype(str)
    )
    return np.asarray(
        [any(str(node) in valid_nodes for node in component) for component in components],
        dtype=bool,
    )

def icrs_radial_unit(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """Return ICRS line-of-sight unit vectors for finite sky coordinates."""
    ra_deg, dec_deg = np.broadcast_arrays(
        np.asarray(ra_deg, float),
        np.asarray(dec_deg, float),
    )
    ra = np.radians(ra_deg.ravel())
    dec = np.radians(dec_deg.ravel())
    return np.column_stack([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec),
    ])

def common_tangent_basis(
    node_rows: pd.DataFrame,
    members: pd.DataFrame,
    weight_mode: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one common Galactic tangent basis from base-node sightlines.

    ``node_equal`` gives every base node one vote. ``N_weighted`` uses the
    production node membership count and is retained as the explicit control.
    """
    mode = REFERENCE_WEIGHT_MODE if weight_mode is None else str(weight_mode)
    if mode not in {"node_equal", "N_weighted"}:
        raise ValueError(f"Unknown reference weight mode: {mode}")
    directions = []
    weights = []
    member_node = members[members["id_node"].notna()].copy()
    member_node["id_node"] = member_node["id_node"].astype(str)
    indexed_nodes = node_rows.copy()
    indexed_nodes["id_node"] = indexed_nodes["id_node"].astype(str)
    indexed_nodes = indexed_nodes.set_index("id_node", drop=False)
    for node_id in sorted(indexed_nodes.index):
        sky = (
            member_node.loc[member_node["id_node"].eq(node_id), ["ra", "dec"]]
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
            .to_numpy(float)
        )
        if len(sky) == 0:
            raise ValueError(f"Node {node_id}: perspective correction requires finite sky positions")
        ra = np.radians(sky[:, 0]); dec = np.radians(sky[:, 1])
        unit = np.column_stack([
            np.cos(dec) * np.cos(ra),
            np.cos(dec) * np.sin(ra),
            np.sin(dec),
        ])
        direction = np.mean(unit, axis=0)
        directions.append(direction / np.linalg.norm(direction))
        if mode == "node_equal":
            weights.append(1.0)
        else:
            weight = float(pd.to_numeric(indexed_nodes.loc[node_id, "N"], errors="coerce"))
            if not np.isfinite(weight) or weight <= 0:
                raise ValueError(f"Node {node_id}: N_weighted reference requires finite N>0")
            weights.append(weight)
    reference_icrs = np.average(
        np.asarray(directions),
        axis=0,
        weights=np.asarray(weights, float),
    )
    reference_icrs /= np.linalg.norm(reference_icrs)
    reference_galactic = reference_icrs @ ICRS_TO_GALACTIC.T
    axes = np.eye(3)
    anchor = axes[np.argmin(np.abs(axes @ reference_galactic))]
    tangent_1 = np.cross(reference_galactic, anchor)
    tangent_1 /= np.linalg.norm(tangent_1)
    tangent_2 = np.cross(reference_galactic, tangent_1)
    tangent_2 /= np.linalg.norm(tangent_2)
    return tangent_1, tangent_2

def add_perspective_velocity_columns(
    node_rows: pd.DataFrame,
    members: pd.DataFrame,
    transport_mode: str | None = None,
    reference_weight_mode: str | None = None,
) -> pd.DataFrame:
    """Transport member tangential velocities to a Snake-wide tangent plane.

    The measured base-node RV is used only in the coordinate transport. In
    ``projected`` mode, each member receives the radial projection of its
    node's systemic 3D vector along that member's own sightline. ``constant``
    assigns the measured node RV unchanged and is the small-angle control. Neither
    mode claims individual stellar RV measurements.
    """
    mode = MEMBER_RV_TRANSPORT_MODE if transport_mode is None else str(transport_mode)
    reference_mode = (
        REFERENCE_WEIGHT_MODE
        if reference_weight_mode is None
        else str(reference_weight_mode)
    )
    if mode not in {"constant", "projected"}:
        raise ValueError(f"Unknown member RV transport mode: {mode}")
    if reference_mode not in {"node_equal", "N_weighted"}:
        raise ValueError(f"Unknown reference weight mode: {reference_mode}")
    required = {"Vtan1_transport", "Vtan2_transport", "U_transport", "V_transport", "W_transport"}
    metadata = {"member_rv_transport_transport", "reference_weight_transport"}
    if required.issubset(members.columns) and metadata.issubset(members.columns):
        cached_transport = set(members["member_rv_transport_transport"].dropna().astype(str))
        cached_reference = set(members["reference_weight_transport"].dropna().astype(str))
        if cached_transport == {mode} and cached_reference == {reference_mode}:
            return members
    if required.issubset(members.columns):
        members = members.drop(columns=list(required | metadata), errors="ignore")
    if not rv_complete_rows(node_rows):
        raise ValueError("Perspective correction requires complete base-node RVs")
    result = members.copy()
    result["member_rv_transport_transport"] = mode
    result["reference_weight_transport"] = reference_mode
    for column in sorted(required):
        result[column] = np.nan
    rv_measured = pd.to_numeric(node_rows["RV"], errors="coerce").to_numpy(float)
    node_ids_all = node_rows["id_node"].astype(str).to_numpy()
    rv_map = {
        str(node): float(value)
        for node, value in zip(node_ids_all, np.asarray(rv_measured, float))
    }
    node_mask = result["id_node"].notna()
    node_ids = result.loc[node_mask, "id_node"].astype(str)
    values = result.loc[node_mask, ["ra", "dec", "Vra", "Vdec"]].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(float)
    if mode == "constant":
        transported_rv = node_ids.map(rv_map).to_numpy(float)
    else:
        node_numeric = node_rows[["RA", "DEC", "Vra", "Vdec"]].apply(
            pd.to_numeric,
            errors="coerce",
        ).to_numpy(float)
        node_uvw = icrs_velocity_to_galactic(
            node_numeric[:, 0],
            node_numeric[:, 1],
            node_numeric[:, 2],
            node_numeric[:, 3],
            np.asarray(rv_measured, float),
        )
        node_icrs = node_uvw @ ICRS_TO_GALACTIC
        if not np.isfinite(node_icrs).all():
            raise ValueError("Projected member transport requires finite node RA/DEC/Vra/Vdec")
        vector_map = {
            str(node): vector
            for node, vector in zip(node_ids_all, node_icrs)
        }
        systemic_vectors = np.asarray([vector_map.get(node, np.full(3, np.nan)) for node in node_ids])
        member_sightlines = icrs_radial_unit(values[:, 0], values[:, 1])
        transported_rv = np.einsum("ij,ij->i", systemic_vectors, member_sightlines)
    uvw = icrs_velocity_to_galactic(
        values[:, 0], values[:, 1], values[:, 2], values[:, 3], transported_rv
    )
    tangent_1, tangent_2 = common_tangent_basis(
        node_rows,
        result,
        weight_mode=reference_mode,
    )
    corrected = np.column_stack([uvw @ tangent_1, uvw @ tangent_2])
    result.loc[node_mask, ["U_transport", "V_transport", "W_transport"]] = uvw
    result.loc[node_mask, ["Vtan1_transport", "Vtan2_transport"]] = corrected
    return result

def add_common_tangent_2d_velocity_columns(
    node_rows: pd.DataFrame,
    members: pd.DataFrame,
    reference_weight_mode: str | None = None,
) -> pd.DataFrame:
    """Project RV-free tangential velocities onto one Snake-wide tangent plane.

    E6 uses this representation for the *whole* Snake whenever any base node
    lacks a production-valid RV.  Every node/member is assigned the same
    neutral radial component (zero) before the coordinate transport; no mean
    RV is estimated and no 2D/3D pairwise distance is mixed.  Cached columns
    intentionally survive the full-to-core subset operation so both scores use
    the reference plane fixed by the original base-node set.
    """
    reference_mode = (
        REFERENCE_WEIGHT_MODE
        if reference_weight_mode is None
        else str(reference_weight_mode)
    )
    if reference_mode not in {"node_equal", "N_weighted"}:
        raise ValueError(f"Unknown reference weight mode: {reference_mode}")
    required = {"Vtan1_transport", "Vtan2_transport"}
    metadata = {"velocity_basis_transport", "reference_weight_transport"}
    if required.issubset(members.columns) and metadata.issubset(members.columns):
        cached_basis = set(members["velocity_basis_transport"].dropna().astype(str))
        cached_reference = set(members["reference_weight_transport"].dropna().astype(str))
        if (
            cached_basis == {"common_tangent_2d_rv_zero"}
            and cached_reference == {reference_mode}
        ):
            return members
    if required.issubset(members.columns):
        members = members.drop(columns=list(required | metadata), errors="ignore")
    result = members.copy()
    result["velocity_basis_transport"] = "common_tangent_2d_rv_zero"
    result["reference_weight_transport"] = reference_mode
    result["Vtan1_transport"] = np.nan
    result["Vtan2_transport"] = np.nan
    node_mask = result["id_node"].notna()
    values = result.loc[node_mask, ["ra", "dec", "Vra", "Vdec"]].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(float)
    uvw_rv_zero = icrs_velocity_to_galactic(
        values[:, 0],
        values[:, 1],
        values[:, 2],
        values[:, 3],
        np.zeros(len(values), dtype=float),
    )
    tangent_1, tangent_2 = common_tangent_basis(
        node_rows,
        result,
        weight_mode=reference_mode,
    )
    projected = np.column_stack([
        uvw_rv_zero @ tangent_1,
        uvw_rv_zero @ tangent_2,
    ])
    result.loc[node_mask, ["Vtan1_transport", "Vtan2_transport"]] = projected
    return result

def add_member_score_columns(
    node_rows: pd.DataFrame,
    members: pd.DataFrame,
    active_velocity_mode: str,
) -> pd.DataFrame:
    """Use the same explicitly selected transport in the member consumer.

    Complete-RV Cartesian mode uses projected common-tangent columns;
    incomplete-RV mode keeps all members in the RV-free tangent plane.
    Source RVs are not imputed or treated as independent member measurements.
    This changes coordinate routing only, not the member kernel or weights.
    """
    mode = str(active_velocity_mode)
    if mode in {"cartesian", "cartesian_centroid"}:
        transported = add_perspective_velocity_columns(node_rows, members)
        basis = "node_systemic_projected_common_tangent_2d"
    elif mode == "valid_rv_centroid":
        transported = add_valid_rv_centroid_velocity_columns(node_rows, members)
        basis = "all_nodes_rv_free_common_tangent_2d"
    elif mode == "common_tangent_2d":
        transported = add_common_tangent_2d_velocity_columns(node_rows, members)
        basis = "all_nodes_rv_free_common_tangent_2d"
    elif mode == "raw":
        result = members.copy()
        result[MEMBER_SCORE_VELOCITY_COLUMNS[0]] = result["Vra"]
        result[MEMBER_SCORE_VELOCITY_COLUMNS[1]] = result["Vdec"]
        result["member_velocity_basis_transport"] = "raw_equatorial_control"
        result["member_velocity_reference_weight_transport"] = "not_applicable"
        return result
    else:
        raise ValueError(f"Unknown member coordinate mode: {mode}")
    result = transported.copy()
    for target, source in zip(MEMBER_SCORE_VELOCITY_COLUMNS, ["Vtan1_transport", "Vtan2_transport"]):
        result[target] = pd.to_numeric(transported[source], errors="coerce").to_numpy(float)
    result["member_velocity_basis_transport"] = basis
    result["member_velocity_reference_weight_transport"] = REFERENCE_WEIGHT_MODE
    return result

def add_valid_rv_centroid_velocity_columns(
    node_rows: pd.DataFrame,
    members: pd.DataFrame,
    reference_weight_mode: str | None = None,
) -> pd.DataFrame:
    """Prepare the E6 valid-RV-centroid branch without imputing missing RVs.

    All base nodes first receive the same RV-free common-tangent representation;
    this is the only basis used for entity contraction and member-cloud
    continuity.  Separate 3D columns are then populated only for members whose
    base node has a production-valid RV.  Invalid nodes remain NaN in those
    columns and therefore cannot enter the centroid or cross terms.
    """
    reference_mode = (
        REFERENCE_WEIGHT_MODE
        if reference_weight_mode is None
        else str(reference_weight_mode)
    )
    result = add_common_tangent_2d_velocity_columns(
        node_rows,
        members,
        reference_weight_mode=reference_mode,
    )
    velocity_columns = {"U_RVvalid_transport", "V_RVvalid_transport", "W_RVvalid_transport"}
    metadata = {"rv_centroid_transport_transport"}
    required = velocity_columns | {"Vtan1_transport", "Vtan2_transport"}
    if required.issubset(result.columns) and metadata.issubset(result.columns):
        cached_transport = set(
            result["rv_centroid_transport_transport"].dropna().astype(str)
        )
        cached_basis = set(result["velocity_basis_transport"].dropna().astype(str))
        cached_reference = set(result["reference_weight_transport"].dropna().astype(str))
        if (
            cached_transport == {"valid_nodes_only_projected"}
            and cached_basis == {"valid_rv_centroid"}
            and cached_reference == {reference_mode}
        ):
            return result
    result = result.drop(columns=list(velocity_columns | metadata), errors="ignore")
    result["velocity_basis_transport"] = "valid_rv_centroid"
    result["rv_centroid_transport_transport"] = "valid_nodes_only_projected"
    for column in sorted(velocity_columns):
        result[column] = np.nan

    valid_nodes = node_rows.loc[rv_valid_mask(node_rows)].copy()
    if len(valid_nodes) == 0:
        return result
    rv_measured = pd.to_numeric(valid_nodes["RV"], errors="coerce").to_numpy(float)
    node_ids_all = valid_nodes["id_node"].astype(str).to_numpy()
    node_numeric = valid_nodes[["RA", "DEC", "Vra", "Vdec"]].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(float)
    node_uvw = icrs_velocity_to_galactic(
        node_numeric[:, 0],
        node_numeric[:, 1],
        node_numeric[:, 2],
        node_numeric[:, 3],
        np.asarray(rv_measured, float),
    )
    node_icrs = node_uvw @ ICRS_TO_GALACTIC
    if not np.isfinite(node_icrs).all():
        raise ValueError("Valid-RV centroid transport requires finite node sky/velocity data")
    vector_map = {
        str(node): vector
        for node, vector in zip(node_ids_all, node_icrs)
    }
    member_node_ids = result["id_node"].astype(str)
    valid_member_mask = result["id_node"].notna() & member_node_ids.isin(vector_map)
    values = result.loc[
        valid_member_mask,
        ["ra", "dec", "Vra", "Vdec"],
    ].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    member_ids = member_node_ids.loc[valid_member_mask]
    systemic_vectors = np.asarray([vector_map[node] for node in member_ids])
    member_sightlines = icrs_radial_unit(values[:, 0], values[:, 1])
    transported_rv = np.einsum("ij,ij->i", systemic_vectors, member_sightlines)
    uvw = icrs_velocity_to_galactic(
        values[:, 0],
        values[:, 1],
        values[:, 2],
        values[:, 3],
        transported_rv,
    )
    result.loc[
        valid_member_mask,
        ["U_RVvalid_transport", "V_RVvalid_transport", "W_RVvalid_transport"],
    ] = uvw
    return result

def configure_entity_velocity_basis(active_velocity_mode: str | None = None) -> None:
    """Set the vendored contraction basis for the selected transport mode."""
    mode = VELOCITY_MODE if active_velocity_mode is None else str(active_velocity_mode)
    velocity_columns = (
        ["Vtan1_transport", "Vtan2_transport"]
        if mode in {"cartesian", "common_tangent_2d", "valid_rv_centroid"}
        else ["Vra", "Vdec"]
    )
    ENT_MOD.PRIMARY_SPACES = {
        "xyz": ["X", "Y", "Z"],
        "velocity_tan": velocity_columns,
    }
