"""Entity construction: local spatial contact + projected velocity-body compatibility.

Entity construction does not assign quality grades.
Input member velocities must ALREADY use the common-tangent Vtan*_transport frame.
The angular maximum is a finite-grid approximation, not an exact continuum supremum.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

MERGE_RULE_ID = "local-contact-velocity-body-grid-v1"
SCORE_RULE_ID = "native-spatial-mem"
VELOCITY_DIRECTIONS = 720
SPATIAL_MEM_ETA = 1.0


@dataclass(frozen=True)
class NodeSummary:
    xyz: np.ndarray
    spatial_scale: float
    velocity_median: np.ndarray
    velocity_mad: np.ndarray
    n_xyz: int
    n_velocity: int
    n_directions: int


def _finite_cloud(values: np.ndarray, dimension: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != dimension:
        raise ValueError(f"{name}: expected an (N,{dimension}) array")
    arr = arr[np.isfinite(arr).all(axis=1)]
    if len(arr) < 2:
        raise ValueError(f"{name}: fewer than two finite members")
    return arr


def internal_mst_lengths(xyz: np.ndarray) -> np.ndarray:
    """Euclidean Prim MST; includes genuine zero edges; O(N^2) time, O(N) memory.

    No sparse zero-as-absence encoding and no physical floor are used.
    This implements the mathematical MST definition; check old/new edge lengths
    explicitly on degenerate or exactly coincident input coordinates.
    """
    points = _finite_cloud(xyz, 3, "xyz")
    # Canonical row order makes the chosen representative of exact ties repeatable.
    points = points[np.lexsort((points[:, 2], points[:, 1], points[:, 0]))]
    n = len(points)
    chosen = np.zeros(n, dtype=bool)
    best = np.full(n, np.inf)
    best[0] = 0.0
    lengths = np.empty(n - 1)
    for step in range(n):
        i = int(np.argmin(np.where(chosen, np.inf, best)))
        if step:
            lengths[step - 1] = best[i]
        chosen[i] = True
        distance = np.linalg.norm(points - points[i], axis=1)
        best = np.minimum(best, distance)
    return lengths


def summarize_node(
    xyz: np.ndarray, velocity: np.ndarray,
    n_directions: int = VELOCITY_DIRECTIONS,
) -> NodeSummary:
    """Summaries of ONE original node. Never summarize a pooled group for the gate."""
    if isinstance(n_directions, bool) or int(n_directions) != n_directions or n_directions < 4:
        raise ValueError("n_directions must be an integer >= 4")
    n_directions = int(n_directions)
    x = _finite_cloud(xyz, 3, "xyz")
    v = _finite_cloud(velocity, 2, "common-tangent velocity")
    scale = float(np.median(internal_mst_lengths(x)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("undefined positive internal spatial scale")
    theta = np.arange(n_directions) * np.pi / n_directions
    directions = np.vstack((np.cos(theta), np.sin(theta)))
    median = np.empty(n_directions)
    mad = np.empty(n_directions)
    for start in range(0, n_directions, 128):
        stop = min(start + 128, n_directions)
        projected = v @ directions[:, start:stop]
        median[start:stop] = np.median(projected, axis=0)
        mad[start:stop] = np.median(np.abs(projected - median[start:stop]), axis=0)
    return NodeSummary(x, scale, median, mad, len(x), len(v), n_directions)


def pair_profile(a: NodeSummary, b: NodeSummary) -> dict:
    if a.n_directions != b.n_directions:
        raise ValueError("All nodes must use the same angular grid")
    gap = float(np.min(cKDTree(b.xyz).query(a.xyz, k=1)[0]))
    rx = gap / (a.spatial_scale + b.spatial_scale)
    delta = np.abs(a.velocity_median - b.velocity_median)
    width = a.velocity_mad + b.velocity_mad
    ratio = np.full_like(delta, np.inf)
    np.divide(delta, width, out=ratio, where=width > 0)
    ratio[(width == 0) & (delta == 0)] = 0.0
    k = int(np.argmax(ratio))
    rv = float(ratio[k])
    return dict(
        R_X=rx, R_V=rv, joint_cost=max(rx, rv),
        spatial_gap_pc=gap,
        spatial_scale_left_pc=a.spatial_scale,
        spatial_scale_right_pc=b.spatial_scale,
        velocity_gap_at_max=float(delta[k]),
        velocity_mad_left_at_max=float(a.velocity_mad[k]),
        velocity_mad_right_at_max=float(b.velocity_mad[k]),
        velocity_argmax_rad=k * np.pi / a.n_directions,
        velocity_directions=a.n_directions,
        pair_pass=bool(rx <= 1.0 and rv <= 1.0),
        pair_status="computed", merge_rule_id=MERGE_RULE_ID,
    )


def partition(
    node_ids: Sequence[str], profiles: Mapping[tuple[str, str], Mapping],
) -> tuple[list[list[str]], list[dict]]:
    """Spatial min-link + velocity max-link on frozen original-node pair values.

    Deterministic constrained agglomeration, NOT pair matching / plain union-find.
    No group-size cap. The output is one operational partition, not a uniquely
    established physical partition. Exact cost ties use canonical node labels.
    """
    ids = list(node_ids)
    if not ids or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError("node_ids must contain nonempty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate node IDs")
    ids = sorted(ids)
    for key in combinations(ids, 2):
        if key not in profiles:
            raise ValueError(f"missing original pair profile: {key}")
        for name in ("R_X", "R_V"):
            value = float(profiles[key][name])
            if np.isnan(value) or value < 0:
                raise ValueError(f"invalid {name} in {key}; unavailable must be explicit +inf")
    groups = [(x,) for x in ids]
    traces = []
    while True:
        options = []
        for i, left in enumerate(groups):
            for j in range(i + 1, len(groups)):
                right = groups[j]
                cross = [profiles[tuple(sorted((a, b)))] for a in left for b in right]
                rx = min(float(p["R_X"]) for p in cross)
                rv = max(float(p["R_V"]) for p in cross)
                if rx <= 1.0 and rv <= 1.0:
                    options.append((max(rx, rv), rv, rx, left, right, i, j))
        if not options:
            break
        cost, rv, rx, left, right, i, j = min(options)
        merged = tuple(sorted(left + right))
        traces.append(dict(step=len(traces) + 1, left="+".join(left),
            right="+".join(right), merged="+".join(merged),
            R_X=rx, R_V=rv, joint_cost=cost, merge_rule_id=MERGE_RULE_ID))
        groups[i] = merged
        del groups[j]
        groups.sort()
    return [list(g) for g in groups], traces


def resolve_components(
    node_ids: Sequence[str], node_members: Mapping[str, pd.DataFrame],
    n_directions: int = VELOCITY_DIRECTIONS,
) -> tuple[list[list[str]], list[list[str]], pd.DataFrame, list[dict]]:
    """Adapter for entities.prepare_structure after common-tangent preparation.

    Malformed/missing tables raise; unavailable finite-member scales are logged
    as non-merging pairs, retaining the original node (never silently dropping it).
    The first and second outputs are identical: no singleton fallback.
    """
    if isinstance(n_directions, bool) or int(n_directions) != n_directions or n_directions < 4:
        raise ValueError("n_directions must be an integer >= 4")
    if any(not isinstance(x, str) or not x for x in node_ids):
        raise ValueError("node_ids must contain nonempty strings")
    ids = sorted(node_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("empty or duplicate node IDs")
    summaries, issues = {}, {}
    needed = ["X", "Y", "Z", "Vtan1_transport", "Vtan2_transport"]
    for name in ids:
        if name not in node_members:
            raise ValueError(f"missing member table for {name}")
        table = node_members[name]
        if not set(needed) <= set(table.columns):
            raise ValueError(f"missing coordinate columns for {name}")
        if "source_id" in table and table.source_id.duplicated().any():
            raise ValueError(f"duplicate source_id within {name}")
        xyz = table[["X", "Y", "Z"]].to_numpy(float)
        vel = table[["Vtan1_transport", "Vtan2_transport"]].to_numpy(float)
        try:
            summaries[name] = summarize_node(xyz, vel, n_directions)
        except ValueError as error:
            issues[name] = str(error)
    lookup, rows = {}, []
    for a, b in combinations(ids, 2):
        if a in issues or b in issues:
            p = dict(R_X=np.inf, R_V=np.inf, joint_cost=np.inf, pair_pass=False,
                     pair_status="unavailable", merge_rule_id=MERGE_RULE_ID,
                     issue_left=issues.get(a, ""), issue_right=issues.get(b, ""))
        else:
            p = pair_profile(summaries[a], summaries[b])
        lookup[a, b] = p
        rows.append(dict(left=a, right=b, **p))
    groups, traces = partition(ids, lookup)
    frame = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "left", "right", "R_X", "R_V", "joint_cost", "pair_pass", "pair_status", "merge_rule_id"])
    return groups, [g.copy() for g in groups], frame, traces


def spatial_mem_response(sm_native: float, eta: float = SPATIAL_MEM_ETA) -> float:
    """Return the native spatial-member score when eta=1."""
    if not np.isfinite(eta) or not 0.0 < eta <= 1.0:
        raise ValueError("eta must be finite in (0,1]")
    if not np.isfinite(sm_native) or not 0.0 <= sm_native <= 1.0:
        raise ValueError("native S_mem must be finite in [0,1]; no clipping or NaN filling")
    return float(sm_native ** eta)


def aggregate_native(
    sm_native: float, sc: float, vm: float, vc: float, cross: float,
    info: float, M: int, *, eta: float = SPATIAL_MEM_ETA,
) -> dict:
    """Replacement for engine.aggregate; first argument is NATIVE, never USED.

    Call ONLY for >=2 natural entities. No calibration, qref, or grade is produced.
    `info` is retained for current-call compatibility; it is NOT a new weight.
    """
    if isinstance(M, bool) or not np.isfinite(M) or int(M) != M or M < 0:
        raise ValueError("M must be a nonnegative integer")
    sm = spatial_mem_response(sm_native, eta)
    active = {"V_mem": vm}
    if M >= 3:
        active.update(S_cen=sc, V_cen=vc)
    if M >= 4:
        active["cross"] = cross
    for name, value in active.items():
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"invalid active component {name}")
    S = float((sm + sc) / 2) if M >= 3 else sm
    V = float((vm + vc) / 2) if M >= 3 else float(vm)
    primary = float(2 * S * V / (S + V)) if S + V > 0 else 0.0
    raw = (0.0 if min(S, V, cross) == 0 else 3 / (1 / S + 1 / V + 1 / cross)) if M >= 4 else primary
    return dict(S_mem_native=float(sm_native), S_mem_used=sm, S_mem=sm,
                spatial_mem_eta=float(eta), score_rule_id=SCORE_RULE_ID,
                spatial=S, velocity=V, primary=primary, SRI_raw=float(raw))
