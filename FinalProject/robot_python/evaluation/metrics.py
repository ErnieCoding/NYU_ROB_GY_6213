"""Evaluation metrics for trajectory accuracy and landmark map consistency."""

from __future__ import annotations

import math
from itertools import combinations

from FinalProject.robot_python.data_types import Pose2D


def compute_ate(estimated: dict[int, Pose2D], ground_truth: dict[int, Pose2D]) -> float:
    """Compute trajectory Absolute Trajectory Error RMSE over matching ids."""
    common_ids = sorted(set(estimated) & set(ground_truth))
    if not common_ids:
        return 0.0

    squared_errors = [
        estimated[node_id].distance_to(ground_truth[node_id]) ** 2
        for node_id in common_ids
    ]
    # TODO: Add trajectory alignment before ATE if ground truth is in a
    # different coordinate frame.
    return math.sqrt(sum(squared_errors) / len(squared_errors))


def compute_map_rmse(
    estimated_landmarks: dict[int, Pose2D],
    ground_truth_landmarks: dict[int, Pose2D],
) -> float:
    """Compute RMSE between pairwise landmark distances and ground truth distances."""
    common_ids = sorted(set(estimated_landmarks) & set(ground_truth_landmarks))
    if len(common_ids) < 2:
        return 0.0

    squared_errors: list[float] = []
    for first_id, second_id in combinations(common_ids, 2):
        estimated_distance = estimated_landmarks[first_id].distance_to(estimated_landmarks[second_id])
        truth_distance = ground_truth_landmarks[first_id].distance_to(ground_truth_landmarks[second_id])
        squared_errors.append((estimated_distance - truth_distance) ** 2)

    # TODO: Confirm this pairwise-distance definition against the final proposal
    # rubric and add per-marker diagnostics.
    return math.sqrt(sum(squared_errors) / len(squared_errors))


def summarize_metrics(ate: float, map_rmse: float) -> dict[str, float]:
    """Return a compact metrics summary for logging or reports."""
    return {
        "ate_rmse_m": ate,
        "map_rmse_m": map_rmse,
    }
