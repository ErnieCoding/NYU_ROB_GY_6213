"""In-memory pose graph manager for future factor-graph optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from FinalProject.robot_python.data_types import LandmarkObservation, PoseEstimate, RelativeMotion


@dataclass
class PoseGraphNode:
    """Pose graph node storing an initial pose estimate."""

    node_id: int
    pose: PoseEstimate


@dataclass
class PoseGraphFactor:
    """Generic factor record used until a solver-specific representation exists."""

    factor_type: str
    node_ids: tuple[int, ...]
    measurement: Any
    metadata: dict[str, Any] = field(default_factory=dict)


class PoseGraphManager:
    """Manage keyframes, factors, and timestamp association for the back-end."""

    def __init__(self) -> None:
        self.nodes: dict[int, PoseGraphNode] = {}
        self.factors: list[PoseGraphFactor] = []
        self.timestamp_by_node: dict[int, float] = {}
        self._next_node_id = 0

    def add_keyframe(self, pose: PoseEstimate) -> int:
        """Add a keyframe node and return its node id."""
        node_id = self._next_node_id
        self._next_node_id += 1
        self.nodes[node_id] = PoseGraphNode(node_id=node_id, pose=pose)
        self.timestamp_by_node[node_id] = pose.timestamp
        return node_id

    def add_odometry_factor(self, i: int, j: int, motion: RelativeMotion) -> None:
        """Add an odometry factor between two consecutive keyframes."""
        # TODO: Replace with an SE(2) residual and configured odometry noise model.
        self.factors.append(PoseGraphFactor("odometry", (i, j), motion))

    def add_lidar_factor(self, i: int, j: int, motion: RelativeMotion) -> None:
        """Add a LiDAR geometric factor between two keyframes."""
        # TODO: Define scan-matching residuals and covariance from match quality.
        self.factors.append(PoseGraphFactor("lidar", (i, j), motion))

    def add_landmark_factor(self, node_id: int, obs: LandmarkObservation) -> None:
        """Add a visual landmark anchor factor for one keyframe."""
        # TODO: Formulate marker observation residual against known marker world
        # poses, and support unknown landmark initialization if needed.
        self.factors.append(PoseGraphFactor("landmark", (node_id,), obs, {"marker_id": obs.marker_id}))

    def add_loop_closure_factor(self, i: int, j: int, motion: RelativeMotion) -> None:
        """Add a loop-closure factor between two non-consecutive keyframes."""
        # TODO: Convert this placeholder into an SE(2) loop-closure residual
        # with robust kernels and a noise model derived from ICP match quality.
        self.factors.append(PoseGraphFactor("loop_closure", (i, j), motion, {"source": motion.source}))

    def get_latest_node_id(self) -> int | None:
        """Return the most recently added node id, if any."""
        if not self.nodes:
            return None
        return max(self.nodes)

    def get_graph_summary(self) -> dict[str, Any]:
        """Return simple graph statistics for logging and debugging."""
        factor_counts: dict[str, int] = {}
        for factor in self.factors:
            factor_counts[factor.factor_type] = factor_counts.get(factor.factor_type, 0) + 1
        return {
            "num_nodes": len(self.nodes),
            "num_factors": len(self.factors),
            "factor_counts": factor_counts,
        }

    def find_closest_node_by_time(self, timestamp: float) -> int | None:
        """Find the keyframe whose timestamp is closest to an observation."""
        if not self.timestamp_by_node:
            return None
        # TODO: Replace nearest timestamp with a more careful asynchronous data
        # association strategy and interpolation when needed.
        return min(self.timestamp_by_node, key=lambda node_id: abs(self.timestamp_by_node[node_id] - timestamp))
