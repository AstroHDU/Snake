"""Generic geometry and graph primitives used by SRI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TWO_NODE_NEUTRAL = 0.88


def pairwise_dist(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return np.zeros((len(points), len(points)), dtype=float)
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def upper_values(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < 2:
        return np.asarray([], dtype=float)
    return np.asarray(matrix[np.triu_indices(matrix.shape[0], k=1)], dtype=float)


def mst_edges_from_distance(distance: np.ndarray) -> list[tuple[int, int]]:
    n = int(distance.shape[0])
    if n < 2:
        return []
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    edges: list[tuple[int, int]] = []
    for _ in range(n - 1):
        best = (np.inf, -1, -1)
        for left in np.where(visited)[0]:
            candidates = np.where(~visited)[0]
            if len(candidates) == 0:
                break
            values = distance[left, candidates]
            position = int(np.nanargmin(values))
            value = float(values[position])
            right = int(candidates[position])
            if value < best[0]:
                best = (value, int(left), right)
        if best[1] < 0:
            break
        _, left, right = best
        visited[right] = True
        edges.append((left, right))
    return edges


def edge_smallness_score(
    source_edges: list[tuple[int, int]],
    target_distance: np.ndarray,
) -> float:
    all_target = upper_values(target_distance)
    all_target = all_target[np.isfinite(all_target)]
    if len(source_edges) == 0 or len(all_target) < 2:
        return np.nan
    ranks = []
    for left, right in source_edges:
        value = float(target_distance[left, right])
        if np.isfinite(value):
            ranks.append(float(np.mean(all_target <= value)))
    if not ranks:
        return np.nan
    return float(np.clip(1.0 - np.mean(ranks), 0.0, 1.0))


def normalized_edge_smallness(
    source_edges: list[tuple[int, int]],
    target_distance: np.ndarray,
) -> float:
    n = target_distance.shape[0]
    if n < 3:
        return np.nan
    raw = edge_smallness_score(source_edges, target_distance)
    finite_sample_maximum = (n - 2.0) / (n - 1.0)
    return float(np.clip(raw / finite_sample_maximum, 0.0, 1.0))


@dataclass(frozen=True)
class EdgeProfile:
    score: float
    max_edge: float
    max_over_median_other: float
    max_over_q75_other: float
    max_edge_is_tukey_outlier: bool
    edges: list[tuple[int, int, float]]
    distance: np.ndarray


def edge_profile(
    points: np.ndarray,
    distance: np.ndarray | None = None,
) -> EdgeProfile:
    """自校准 MST 边分: score = clip(2/(1+ratio),0,1), ratio = 最长MST边/其余边q75。
    与本分支固定的自校准边分口径一致，比包默认高斯式更保守。"""
    if distance is None:
        distance = pairwise_dist(points)
    else:
        distance = np.asarray(distance, float)
        if distance.shape != (len(points), len(points)):
            raise ValueError("custom edge distance shape does not match points")
        distance = distance.copy()
        np.fill_diagonal(distance, 0.0)
    edges = sorted([(l, r, float(distance[l, r])) for l, r in mst_edges_from_distance(distance)],
                   key=lambda it: it[2], reverse=True)
    if len(edges) == 0:
        return EdgeProfile(np.nan, np.nan, np.nan, np.nan, False, [], distance)
    if len(edges) == 1:
        return EdgeProfile(np.nan, edges[0][2], np.nan, np.nan, False, edges, distance)
    max_edge = edges[0][2]
    other = np.asarray([e[2] for e in edges[1:]], float)
    fin = other[np.isfinite(other) & (other > 0)]
    if not np.isfinite(max_edge) or max_edge <= 0 or len(fin) == 0:
        score = ratio = np.nan
    else:
        q75 = float(np.quantile(fin, 0.75)); ratio = max_edge / q75 if q75 > 0 else np.nan
        score = float(np.clip(2.0 / (1.0 + ratio), 0.0, 1.0)) if np.isfinite(ratio) else np.nan
    return EdgeProfile(score, max_edge, ratio, ratio, False, edges, distance)
