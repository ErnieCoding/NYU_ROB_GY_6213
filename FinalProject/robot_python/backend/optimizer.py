"""Placeholder graph optimizer interface for future GTSAM or solver integration."""

from __future__ import annotations

from FinalProject.robot_python.backend.pose_graph import PoseGraphManager
from FinalProject.robot_python.config.config import BackendConfig
from FinalProject.robot_python.data_types import Pose2D


class GraphOptimizer:
    """Optimize the pose graph and expose optimized poses."""

    def __init__(self, config: BackendConfig) -> None:
        self.config = config

    def optimize(self, graph_manager: PoseGraphManager) -> dict[int, Pose2D]:
        """Run graph optimization and return optimized trajectory by node id."""
        # TODO: Integrate a nonlinear optimizer such as GTSAM, define SE(2)
        # variables/factors, extract optimized trajectory, and optionally return
        # marginal covariances.
        return {
            node_id: Pose2D(node.pose.pose.x, node.pose.pose.y, node.pose.pose.theta)
            for node_id, node in graph_manager.nodes.items()
        }

    def should_optimize(self, num_new_keyframes: int, elapsed_time: float | None = None) -> bool:
        """Decide whether optimization should run."""
        if num_new_keyframes >= self.config.optimize_every_n_keyframes:
            return True
        if elapsed_time is not None and elapsed_time >= self.config.optimize_every_seconds:
            return True
        return False
