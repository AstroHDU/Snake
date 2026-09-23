from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
RANDOM_SEED=20260609
N_BACKGROUND_ROTATIONS=100
BACKGROUND_NULL_MODE="los"
def midrank_percentile(observed: float, null: np.ndarray) -> float:
    finite = null[np.isfinite(null)]
    if not np.isfinite(observed) or len(finite) == 0:
        return np.nan
    tied = np.isclose(finite, observed)
    less = np.count_nonzero((finite < observed) & ~tied)
    equal = np.count_nonzero(tied)
    return float((less + 0.5 * equal) / len(finite))

def support_only_corrected_continuity(
    node_continuity: float,
    observed_gain: float,
    signed_alignment: float,
) -> float:
    """Add companion support without subtracting node-only evidence."""
    values = np.asarray(
        [node_continuity, observed_gain, signed_alignment],
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        return np.nan
    return float(
        np.clip(
            node_continuity
            + max(observed_gain, 0.0) * max(signed_alignment, 0.0),
            0.0,
            1.0,
        )
    )

def random_rotation(rng: np.random.Generator) -> np.ndarray:
    matrix = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(matrix)
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q

def rotate_offsets_about_axis(
    offsets: np.ndarray,
    axis: np.ndarray,
    angle: float,
) -> np.ndarray:
    """Rotate row-vector offsets around ``axis`` using Rodrigues' formula."""
    offsets = np.asarray(offsets, dtype=float)
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Line-of-sight rotation requires a finite non-zero axis")
    unit = axis / norm
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    parallel = np.outer(offsets @ unit, unit)
    perpendicular = offsets - parallel
    return parallel + cosine * perpendicular + sine * np.cross(unit, offsets)

def xyz_values(data: pd.DataFrame) -> np.ndarray:
    return (
        data[["X", "Y", "Z"]]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
        .to_numpy(float)
    )

def rotate_background_by_part(
    background: pd.DataFrame,
    part_centers: dict[str, np.ndarray],
    rng: np.random.Generator,
    mode: str | None = None,
) -> pd.DataFrame:
    selected_mode = BACKGROUND_NULL_MODE if mode is None else mode
    if selected_mode not in {"isotropic", "los"}:
        raise ValueError(f"Unknown background null mode: {selected_mode}")
    blocks = []
    for part, data in background.groupby("id_part"):
        block = data.copy()
        numeric = block[["X", "Y", "Z"]].apply(
            pd.to_numeric,
            errors="coerce",
        )
        valid = numeric.notna().all(axis=1)
        values = numeric.loc[valid].to_numpy(float)
        center = part_centers[str(part)]
        offsets = values - center
        if selected_mode == "isotropic":
            # Keep this expression and RNG call identical to the established
            # isotropic baseline so that the null remains reproducible.
            rotated = offsets @ random_rotation(rng).T + center
        else:
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            rotated = rotate_offsets_about_axis(offsets, center, angle) + center
        block.loc[valid, ["X", "Y", "Z"]] = rotated
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)

def assign_background_xyz(
    entity_members: dict[str, pd.DataFrame],
    entity_to_parts: dict[str, set[str]],
    background: pd.DataFrame,
) -> dict[str, np.ndarray]:
    node_xyz = {
        entity: xyz_values(data)
        for entity, data in entity_members.items()
    }
    trees = {
        entity: cKDTree(values)
        for entity, values in node_xyz.items()
    }
    assigned = {entity: [] for entity in entity_members}
    for part, data in background.groupby("id_part"):
        candidates = [
            entity
            for entity, parts in entity_to_parts.items()
            if str(part) in parts
        ]
        if not candidates:
            continue
        values = xyz_values(data)
        if len(values) == 0:
            continue
        distances = np.column_stack(
            [trees[entity].query(values, k=1)[0] for entity in candidates]
        )
        owners = np.argmin(distances, axis=1)
        for index, entity in enumerate(candidates):
            selected = values[owners == index]
            if len(selected):
                assigned[entity].append(selected)
    return {
        entity: (
            np.vstack([values, *assigned[entity]])
            if assigned[entity]
            else values
        )
        for entity, values in node_xyz.items()
    }
