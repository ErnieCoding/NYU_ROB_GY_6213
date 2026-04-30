"""Plotting placeholders for SLAM trajectories, landmarks, and error curves."""

from __future__ import annotations

from collections.abc import Sequence

from FinalProject.robot_python.data_types import Pose2D


def plot_trajectory(
    estimated: dict[int, Pose2D],
    ground_truth: dict[int, Pose2D] | None = None,
    output_path: str | None = None,
) -> None:
    """Plot estimated and optional ground-truth trajectories."""
    # TODO: Use matplotlib to plot x/y paths, equal axes, labels, and save or
    # display depending on output_path.
    _ = estimated
    _ = ground_truth
    _ = output_path


def plot_landmark_map(
    estimated_landmarks: dict[int, Pose2D],
    ground_truth_landmarks: dict[int, Pose2D] | None = None,
    output_path: str | None = None,
) -> None:
    """Plot estimated and optional ground-truth landmark positions."""
    # TODO: Use matplotlib scatter plots with marker ids and consistent axes.
    _ = estimated_landmarks
    _ = ground_truth_landmarks
    _ = output_path


def plot_error_curves(errors: Sequence[float], output_path: str | None = None) -> None:
    """Plot an error sequence over time or frame index."""
    # TODO: Use matplotlib to visualize ATE over time, scan-match residuals, or
    # landmark residual history.
    _ = errors
    _ = output_path
