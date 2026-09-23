"""Entity construction from original-node member clouds.

The default uses the local-contact plus velocity-body rule.  Original
nodes are tested with a spatial contact ratio and a finite-grid projected
velocity-body ratio, then assembled by constrained agglomeration.  Only this construction is exposed by the public pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.optimize import minimize


PRIMARY_SPACES = {
    "xyz": ["X", "Y", "Z"],
    "velocity_tan": ["Vra", "Vdec"],
}
SUPPORT_SPACES = {}

CONTRACTION_MODES = {"local_contact_velocity_body"}
CONTRACTION_MODE = "local_contact_velocity_body"
JOINT_COST_LIMIT = 1.0
STRICT_SPACE_LIMIT = 1.0
STRICT_VELOCITY_LIMIT = 1.0
STRICT_MIX_LIMIT = 0.5

def set_contraction_mode(mode):
    if mode != CONTRACTION_MODE:
        raise ValueError('Only local_contact_velocity_body is supported')


class UnionFind:
    def __init__(self, labels: list[str]) -> None:
        self.parent = {label: label for label in labels}

    def find(self, label: str) -> str:
        root = label
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[label] != label:
            next_label = self.parent[label]
            self.parent[label] = root
            label = next_label
        return root

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root

    def components(self) -> list[list[str]]:
        groups: dict[str, list[str]] = {}
        for label in self.parent:
            groups.setdefault(self.find(label), []).append(label)
        return sorted(
            [sorted(group) for group in groups.values()],
            key=lambda group: (group[0], len(group)),
        )


def numeric_values(data: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return (
        data[columns]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
        .to_numpy(float)
    )


def internal_mst_edges(values: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.asarray([], dtype=float)
    return minimum_spanning_tree(squareform(pdist(values))).data.astype(float)


def geometric_median(values: np.ndarray) -> np.ndarray:
    """Deterministic two-dimensional geometric median used by the strict rule."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values).all(axis=1)]
    if not len(values):
        raise ValueError("empty velocity cloud")
    centre = np.median(values, axis=0)
    scale = float(np.max(np.ptp(values, axis=0)))
    if scale == 0:
        return centre
    tolerance = max(scale * 1e-10, np.finfo(float).eps)
    zero_tolerance = tolerance * 1e-3
    for _ in range(10000):
        distance = np.linalg.norm(values - centre, axis=1)
        near = distance <= zero_tolerance
        if near.any():
            candidate = values[int(np.argmin(distance))]
            other = ~near
            if not other.any():
                return candidate.copy()
            directions = (values[other] - candidate) / distance[other, None]
            if np.linalg.norm(directions.sum(axis=0)) <= int(near.sum()) + 1e-12:
                return candidate.copy()
        weights = 1.0 / np.maximum(distance, zero_tolerance)
        updated = (values * weights[:, None]).sum(axis=0) / weights.sum()
        if np.linalg.norm(updated - centre) <= tolerance:
            return updated
        centre = updated
    objective = lambda point: float(np.linalg.norm(values - point, axis=1).sum())
    result = minimize(
        objective,
        centre,
        method="Nelder-Mead",
        options={"xatol": tolerance, "fatol": tolerance, "maxiter": 2000},
    )
    if result.success and np.isfinite(result.x).all():
        return np.asarray(result.x, float)
    raise ValueError("geometric median did not converge")


def directional_mixing(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    """Return normalized 1NN cross-label mixing in both directions."""
    left = np.asarray(left, float)
    right = np.asarray(right, float)
    left = left[np.isfinite(left).all(axis=1)]
    right = right[np.isfinite(right).all(axis=1)]
    n_left, n_right = len(left), len(right)
    if n_left == 0 or n_right == 0 or n_left + n_right < 2:
        return 0.0, 0.0
    pooled = np.concatenate([left, right])
    index = cKDTree(pooled).query(pooled, k=2)[1]
    nearest = np.where(index[:, 0] != np.arange(len(pooled)), index[:, 0], index[:, 1])
    factor = (n_left + n_right - 1) / (n_left * n_right)
    return (
        float(np.sum(nearest[:n_left] >= n_left) * factor),
        float(np.sum(nearest[n_left:] < n_left) * factor),
    )


def strict_space_cost(left: np.ndarray, right: np.ndarray) -> float:
    """Legacy source-star gap divided by pooled within-cloud MST scale."""
    left = np.asarray(left, float)
    right = np.asarray(right, float)
    left = left[np.isfinite(left).all(axis=1)]
    right = right[np.isfinite(right).all(axis=1)]
    if not len(left) or not len(right):
        return np.inf
    widths = np.concatenate([internal_mst_edges(left), internal_mst_edges(right)])
    denominator = float(np.median(widths)) if len(widths) else 0.0
    if denominator <= 0 or not np.isfinite(denominator):
        return np.inf
    gap = float(np.min(cKDTree(right).query(left)[0]))
    return gap / denominator


def strict_velocity_cost(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float, float]:
    """Projected median-gap/MAD ratio and its auditable ingredients."""
    left = np.asarray(left, float)
    right = np.asarray(right, float)
    left = left[np.isfinite(left).all(axis=1)]
    right = right[np.isfinite(right).all(axis=1)]
    if not len(left) or not len(right):
        return np.inf, np.nan, np.nan, np.nan
    centre_left = geometric_median(left)
    centre_right = geometric_median(right)
    direction = centre_right - centre_left
    norm = float(np.linalg.norm(direction))
    direction = direction / norm if norm > 0 else np.array([1.0, 0.0])
    projected_left = left @ direction
    projected_right = right @ direction
    median_left = float(np.median(projected_left))
    median_right = float(np.median(projected_right))
    mad_left = float(np.median(np.abs(projected_left - median_left)))
    mad_right = float(np.median(np.abs(projected_right - median_right)))
    gap = abs(median_left - median_right)
    denominator = min(mad_left, mad_right)
    cost = gap / denominator if denominator > 0 else (0.0 if gap == 0 else np.inf)
    return float(cost), float(gap), mad_left, mad_right


def strict_pair_profile(left: str, right: str, spatial: dict[str, pd.DataFrame]) -> dict[str, float | bool | str]:
    """Evaluate one base-node pair under the strict structural rule."""
    a = spatial[left]
    b = spatial[right]
    xyz_a = a[["X", "Y", "Z"]].to_numpy(float)
    xyz_b = b[["X", "Y", "Z"]].to_numpy(float)
    velocity_a = a[["Vtan1_transport", "Vtan2_transport"]].to_numpy(float)
    velocity_b = b[["Vtan1_transport", "Vtan2_transport"]].to_numpy(float)
    velocity_cost, gap, mad_a, mad_b = strict_velocity_cost(velocity_a, velocity_b)
    mix_a, mix_b = directional_mixing(velocity_a, velocity_b)
    space_cost = strict_space_cost(xyz_a, xyz_b)
    return {
        "left": left,
        "right": right,
        "space_cost": float(space_cost),
        "velocity_cost": float(velocity_cost),
        "velocity_gap": float(gap),
        "velocity_mad_left": float(mad_a),
        "velocity_mad_right": float(mad_b),
        "mix_left": float(mix_a),
        "mix_right": float(mix_b),
        "mix_min": float(min(mix_a, mix_b)),
        "pair_pass": bool(
            space_cost <= STRICT_SPACE_LIMIT
            and velocity_cost <= STRICT_VELOCITY_LIMIT
            and min(mix_a, mix_b) >= STRICT_MIX_LIMIT
        ),
    }


def relaxed_joint_cost(
    space_cost: float,
    velocity_cost: float,
    mix_min: float,
) -> float:
    """Return the unified spatial/velocity pair cost.

    Mixing is a support term: when both velocity clouds interleave, the
    larger of the two normalized separation costs is reduced continuously.
    Invalid or non-positive-denominator inputs are rejected as infinite cost.
    """
    values = np.asarray([space_cost, velocity_cost, mix_min], dtype=float)
    if not np.isfinite(values).all() or mix_min <= -1.0:
        return float(np.inf)
    return float(max(space_cost, velocity_cost) / (1.0 + mix_min))


def relaxed_joint_pair_profile(
    left: str,
    right: str,
    spatial: dict[str, pd.DataFrame],
) -> dict[str, float | bool | str]:
    """Evaluate a pair under the unified joint acceptance rule."""
    profile = strict_pair_profile(left, right, spatial)
    profile["joint_cost"] = relaxed_joint_cost(
        float(profile["space_cost"]),
        float(profile["velocity_cost"]),
        float(profile["mix_min"]),
    )
    profile["pair_pass"] = bool(profile["joint_cost"] <= JOINT_COST_LIMIT)
    return profile


def relaxed_joint_resolve_components(
    node_ids: list[str],
    node_members: dict[str, pd.DataFrame],
) -> tuple[list[list[str]], list[list[str]], pd.DataFrame, list[dict[str, object]]]:
    """Assemble entities from one-pass non-conflicting accepted pairs.

    Pair eligibility is decided only by ``joint_cost <= 1``.  Accepted pairs
    are then considered from the smallest joint cost upward; a pair is kept
    only when neither endpoint has already been used.  This prevents a
    three-node chain from becoming a larger entity and prevents overlapping
    subset proposals from producing multiple competing partitions.
    """
    spatial = dict(node_members)
    pair_rows = [
        relaxed_joint_pair_profile(left, right, spatial)
        for index, left in enumerate(node_ids)
        for right in node_ids[index + 1 :]
    ]
    pair_frame = pd.DataFrame(pair_rows)
    accepted = pair_frame[pair_frame["pair_pass"]].sort_values(
        ["joint_cost", "space_cost", "velocity_cost", "left", "right"],
        kind="mergesort",
    )

    used: set[str] = set()
    selected: list[dict[str, object]] = []
    for row in accepted.itertuples(index=False):
        if row.left in used or row.right in used:
            continue
        selected.append(row._asdict())
        used.update((row.left, row.right))

    components = [
        sorted([str(row["left"]), str(row["right"])])
        for row in selected
    ]
    components.extend([[node_id] for node_id in node_ids if node_id not in used])
    natural = sorted(components, key=lambda group: (group[0], len(group)))
    scoring = [[node_id] for node_id in node_ids] if len(natural) == 1 else natural

    traces = [
        {
            "left": str(row["left"]),
            "right": str(row["right"]),
            "joint_cost": float(row["joint_cost"]),
            "space_cost": float(row["space_cost"]),
            "velocity_cost": float(row["velocity_cost"]),
            "velocity_gap": float(row["velocity_gap"]),
            "velocity_mad_left": float(row["velocity_mad_left"]),
            "velocity_mad_right": float(row["velocity_mad_right"]),
            "mix_left": float(row["mix_left"]),
            "mix_right": float(row["mix_right"]),
            "mix_min": float(row["mix_min"]),
            "assembly": "greedy_non_conflicting",
        }
        for row in selected
    ]
    return scoring, natural, pair_frame, traces


def strict_resolve_components(
    node_ids: list[str],
    node_members: dict[str, pd.DataFrame],
) -> tuple[list[list[str]], list[list[str]], pd.DataFrame, list[dict[str, object]]]:
    """Construct strict groups with pooled recomputation and anti-chaining."""
    pair_rows = []
    spatial = dict(node_members)
    for index, left in enumerate(node_ids):
        for right in node_ids[index + 1 :]:
            pair_rows.append(strict_pair_profile(left, right, spatial))
    pair_frame = pd.DataFrame(pair_rows)
    pair_lookup = {
        tuple(sorted((row.left, row.right))): row._asdict()
        for row in pair_frame.itertuples(index=False)
    }
    groups = [tuple([node_id]) for node_id in node_ids]
    clouds: dict[tuple[str, ...], pd.DataFrame] = {}
    traces: list[dict[str, object]] = []

    def pooled(group: tuple[str, ...]) -> pd.DataFrame:
        if group not in clouds:
            clouds[group] = pd.concat([node_members[node] for node in group], ignore_index=True)
        return clouds[group]

    while True:
        options = []
        for i, left_group in enumerate(groups):
            for j in range(i + 1, len(groups)):
                right_group = groups[j]
                original = [
                    pair_lookup[tuple(sorted((left, right)))]
                    for left in left_group
                    for right in right_group
                ]
                worst = max(float(row["velocity_cost"]) for row in original)
                left_cloud = pooled(left_group)
                right_cloud = pooled(right_group)
                whole_cost, gap, mad_left, mad_right = strict_velocity_cost(
                    left_cloud[["Vtan1_transport", "Vtan2_transport"]].to_numpy(float),
                    right_cloud[["Vtan1_transport", "Vtan2_transport"]].to_numpy(float),
                )
                mix_left, mix_right = directional_mixing(
                    left_cloud[["Vtan1_transport", "Vtan2_transport"]].to_numpy(float),
                    right_cloud[["Vtan1_transport", "Vtan2_transport"]].to_numpy(float),
                )
                space_cost = strict_space_cost(
                    left_cloud[["X", "Y", "Z"]].to_numpy(float),
                    right_cloud[["X", "Y", "Z"]].to_numpy(float),
                )
                velocity_cost = max(worst, whole_cost)
                if (
                    velocity_cost <= STRICT_VELOCITY_LIMIT
                    and min(mix_left, mix_right) >= STRICT_MIX_LIMIT
                    and space_cost <= STRICT_SPACE_LIMIT
                ):
                    options.append(
                        (
                            velocity_cost,
                            space_cost,
                            left_group,
                            right_group,
                            i,
                            j,
                            worst,
                            whole_cost,
                            gap,
                            mad_left,
                            mad_right,
                            mix_left,
                            mix_right,
                        )
                    )
        if not options:
            break
        (
            velocity_cost,
            space_cost,
            left_group,
            right_group,
            left_index,
            right_index,
            worst,
            whole_cost,
            gap,
            mad_left,
            mad_right,
            mix_left,
            mix_right,
        ) = min(options)
        traces.append(
            {
                "left": "+".join(left_group),
                "right": "+".join(right_group),
                "velocity_cost": float(velocity_cost),
                "worst_original_velocity_cost": float(worst),
                "pooled_velocity_cost": float(whole_cost),
                "space_cost": float(space_cost),
                "velocity_gap": float(gap),
                "velocity_mad_left": float(mad_left),
                "velocity_mad_right": float(mad_right),
                "mix_left": float(mix_left),
                "mix_right": float(mix_right),
            }
        )
        groups[left_index] = tuple(sorted(left_group + right_group))
        del groups[right_index]
    natural = sorted(groups)
    scoring = [[node_id] for node_id in node_ids] if len(natural) == 1 else natural
    return scoring, natural, pair_frame, traces


def pair_costs(
    node_ids: list[str],
    members: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    spaces = {**PRIMARY_SPACES, **SUPPORT_SPACES}
    internal = {
        (node_id, space): internal_mst_edges(
            numeric_values(members[node_id], columns)
        )
        for node_id in node_ids
        for space, columns in spaces.items()
    }
    rows: list[dict[str, object]] = []
    for left_index, left in enumerate(node_ids):
        for right in node_ids[left_index + 1 :]:
            row: dict[str, object] = {"left": left, "right": right}
            primary_ratios: list[float] = []
            support_ratios: list[float] = []
            for space, columns in spaces.items():
                left_values = numeric_values(members[left], columns)
                right_values = numeric_values(members[right], columns)
                pooled = np.concatenate(
                    [internal[(left, space)], internal[(right, space)]]
                )
                if (
                    len(left_values) == 0
                    or len(right_values) == 0
                    or len(pooled) == 0
                ):
                    bridge = scale = ratio = np.nan
                else:
                    bridge = float(cdist(left_values, right_values).min())
                    scale = float(np.median(pooled))
                    ratio = bridge / scale if scale > 0 else np.nan
                row[f"{space}_bridge"] = bridge
                row[f"{space}_internal_median"] = scale
                row[f"{space}_relative_bridge"] = ratio
                if np.isfinite(ratio):
                    (
                        primary_ratios
                        if space in PRIMARY_SPACES
                        else support_ratios
                    ).append(float(ratio))
            row["primary_merge_cost"] = (
                max(primary_ratios) if primary_ratios else np.nan
            )
            row["support_merge_cost"] = (
                max(support_ratios) if support_ratios else np.nan
            )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["primary_merge_cost", "left", "right"]
    )


def graph_mst(node_ids, pairs):
    from .numerics import graph_mst as implementation
    return implementation(node_ids, pairs)


def _contract_components(
    node_ids: list[str],
    tree: pd.DataFrame,
    minimum_components: int,
) -> list[list[str]]:
    union_find = UnionFind(node_ids)
    component_count = len(node_ids)
    ordered_tree = tree.sort_values(
        ["primary_merge_cost", "left", "right"],
        kind="mergesort",
    )
    for row in ordered_tree.itertuples(index=False):
        if row.primary_merge_cost <= 1.0:
            if component_count <= minimum_components:
                break
            left_root = union_find.find(row.left)
            right_root = union_find.find(row.right)
            union_find.union(row.left, row.right)
            if left_root != right_root:
                component_count -= 1
    return union_find.components()


def resolve_components(
    node_ids: list[str],
    tree: pd.DataFrame,
) -> tuple[list[list[str]], list[list[str]]]:
    """Return scoring components and unconstrained natural components."""
    if CONTRACTION_MODE == "base_node_direct":
        natural_components = [[node_id] for node_id in node_ids]
    else:
        natural_components = _contract_components(
            node_ids,
            tree,
            minimum_components=1,
        )
    if CONTRACTION_MODE == "min2_constrained":
        components = _contract_components(
            node_ids,
            tree,
            minimum_components=2,
        )
    elif CONTRACTION_MODE in {"fallback_entity_all", "fallback_natural_all"}:
        # The fallback is applied only when the natural contraction would
        # leave fewer than three entities.  For natural >= 3, retain the
        # natural components; B/C differ only in which count opens the gates.
        components = (
            [[node_id] for node_id in node_ids]
            if len(natural_components) < 3
            else natural_components
        )
    elif CONTRACTION_MODE == "fallback_singleton_only":
        components = (
            [[node_id] for node_id in node_ids]
            if len(natural_components) == 1
            else natural_components
        )
    elif len(natural_components) < 3:
        components = [[node_id] for node_id in node_ids]
    else:
        components = natural_components
    return components, natural_components


def intrinsic_components(
    node_ids: list[str],
    tree: pd.DataFrame,
) -> list[list[str]]:
    components, _ = resolve_components(node_ids, tree)
    return components


def aggregate_members(
    components: list[list[str]],
    members: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    return {
        "+".join(component): pd.concat(
            [members[node_id] for node_id in component],
            ignore_index=True,
        )
        for component in components
    }


def aggregate_group(
    original: pd.DataFrame,
    components: list[list[str]],
) -> pd.DataFrame:
    rows = []
    for component in components:
        block = original[original["id_node"].isin(component)]
        weights = pd.to_numeric(block["N"], errors="coerce").fillna(1.0)
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            weights = pd.Series(np.ones(len(block)), index=block.index)
        merged = block.iloc[[0]].copy()
        numeric_columns = merged.select_dtypes(include=[np.number]).columns
        merged[numeric_columns] = merged[numeric_columns].astype(float)
        merged.loc[:, numeric_columns] = np.average(
            block[numeric_columns].to_numpy(float),
            axis=0,
            weights=weights.to_numpy(float),
        )
        merged["id_node"] = "+".join(component)
        merged["id"] = merged["id_node"]
        merged["N"] = block["N"].sum()
        merged["N_RV"] = block["N_RV"].sum()
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def component_signature(components: list[list[str]]) -> str:
    return ";".join("+".join(group) for group in components)


def entity_parts(
    components: list[list[str]],
    node_rows: pd.DataFrame,
) -> dict[str, set[str]]:
    result = {}
    for component in components:
        parts = node_rows.loc[
            node_rows["id_node"].isin(component),
            "id_part",
        ].dropna()
        result["+".join(component)] = set(parts.astype(str))
    return result


def prepare_structure(
    snake: int,
    original: pd.DataFrame,
    snake_members: pd.DataFrame,
) -> dict[str, object]:
    node_ids = sorted(original["id_node"].astype(str).tolist())
    node_rows = snake_members[snake_members["id_node"].notna()].copy()
    node_rows["id_node"] = node_rows["id_node"].astype(str)
    missing = sorted(set(node_ids) - set(node_rows["id_node"]))
    if missing:
        raise ValueError(f"Snake {snake}: missing member rows for {missing}")
    node_members = {
        node_id: node_rows[node_rows["id_node"].eq(node_id)].copy()
        for node_id in node_ids
    }
    relation_status = "computed"
    singleton_fallback = False
    from .entity_merge import MERGE_RULE_ID, resolve_components
    components, natural_components, strict_pairs, strict_trace = resolve_components(node_ids,node_members,n_directions=720)
    merge_rule_id = MERGE_RULE_ID
    if len(natural_components)==1:
        components=[[node_id] for node_id in node_ids]
        relation_status="singleton_fallback"
        singleton_fallback=True
    node_tree=strict_pairs.copy()
    entity_members = aggregate_members(components, node_members)
    entity_ids = sorted(entity_members)
    fixed_tree = graph_mst(
        entity_ids,
        pair_costs(entity_ids, entity_members),
    )
    return {
        "node_ids": node_ids,
        "node_rows": node_rows,
        "node_tree": node_tree,
        "natural_components": natural_components,
        "natural_entity_count": len(natural_components),
        "background": snake_members[snake_members["id_node"].isna()].copy(),
        "components": components,
        "entity_members": entity_members,
        "fixed_tree": fixed_tree,
        "entity_to_parts": entity_parts(components, node_rows),
        "strict_pairs": strict_pairs,
        "strict_trace": strict_trace,
        "relation_status": relation_status,
        "singleton_fallback": singleton_fallback,
        "merge_rule_id": merge_rule_id,
    }


def selected_parts(original: pd.DataFrame) -> set[str]:
    return {
        part
        for values in original["id_part_list"].fillna("").astype(str)
        for part in values.split(",")
        if part
    }


def subset_members(
    members: pd.DataFrame,
    original: pd.DataFrame,
) -> pd.DataFrame:
    node_ids = set(original["id_node"].astype(str))
    parts = selected_parts(original)
    node_text = members["id_node"].astype(str).str.strip()
    part_text = members["id_part"].astype(str).str.strip()
    node_missing = members["id_node"].isna() | node_text.isin(
        ["", "nan", "None", "<NA>"],
    )
    part_present = members["id_part"].notna() & ~part_text.isin(
        ["", "nan", "None", "<NA>"],
    )
    node_mask = node_text.isin(node_ids)
    background_mask = (
        node_missing
        & part_present
        & part_text.isin(parts)
    )
    return members[node_mask | background_mask].copy()


def split_tree(
    n: int,
    edges: list[tuple[int, int]],
    removed_edge: tuple[int, int],
) -> tuple[set[int], set[int]]:
    adjacency = {index: set() for index in range(n)}
    for left, right in edges:
        if {left, right} == set(removed_edge):
            continue
        adjacency[left].add(right)
        adjacency[right].add(left)
    first = {removed_edge[0]}
    stack = [removed_edge[0]]
    while stack:
        current = stack.pop()
        for neighbour in adjacency[current]:
            if neighbour not in first:
                first.add(neighbour)
                stack.append(neighbour)
    return first, set(range(n)) - first


def base_ids_for_entities(
    components: list[list[str]],
    indices: set[int],
) -> list[str]:
    return sorted(
        node_id
        for index in indices
        for node_id in components[index]
    )


def component_weight(group: pd.DataFrame, indices: set[int]) -> float:
    return float(
        pd.to_numeric(
            group.iloc[sorted(indices)]["N"],
            errors="coerce",
        )
        .fillna(1.0)
        .sum()
    )
