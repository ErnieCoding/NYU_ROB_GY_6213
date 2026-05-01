"""Placeholder graph optimizer interface for future GTSAM or solver integration."""

from __future__ import annotations

from FinalProject.robot_python.backend.pose_graph import PoseGraphManager
from FinalProject.robot_python.config.config import BackendConfig
from FinalProject.robot_python.data_types import Pose2D
import gtsam
import numpy as np

class GraphOptimizer:
    """Optimize the pose graph and expose optimized poses."""

    def __init__(self, config: BackendConfig) -> None:
        self.config = config

    def optimize(self, graph_manager: PoseGraphManager) -> dict[int, Pose2D]:
        """Run graph optimization and return optimized trajectory by node id."""
        

        def pose2_from_pose(pose: Pose2D) -> gtsam.Pose2:
            """Convert the project Pose2D type into a GTSAM Pose2."""
            return gtsam.Pose2(float(pose.x), float(pose.y), float(pose.theta))

        def covariance_noise(covariance) -> gtsam.noiseModel.Gaussian:
            """Convert a 3x3 covariance into a GTSAM Gaussian noise model."""
            if covariance is None:
                covariance = np.eye(3)
            covariance = np.asarray(covariance, dtype=float)
            return gtsam.noiseModel.Gaussian.Covariance(covariance)

        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()

        # Insert all keyframe poses as initial estimates.
        for node_id, node in graph_manager.nodes.items():
            initial.insert(node_id, pose2_from_pose(node.pose.pose))

        if not graph_manager.nodes:
            return {}

        # Anchor the first node so the graph has a fixed global reference.
        first_node_id = min(graph_manager.nodes)
        first_pose = graph_manager.nodes[first_node_id].pose.pose
        prior_noise = gtsam.noiseModel.Gaussian.Covariance(np.diag([1e-6, 1e-6, 1e-6]))
        graph.add(gtsam.PriorFactorPose2(first_node_id, pose2_from_pose(first_pose), prior_noise))

        # Convert stored project factors into GTSAM Pose2 factors.
        for factor in graph_manager.factors:
            if factor.factor_type in {"odometry", "lidar", "loop_closure"}:
                if len(factor.node_ids) != 2:
                    continue
                i, j = factor.node_ids
                measurement = factor.measurement
                relative_pose = gtsam.Pose2(
                    float(measurement.dx),
                    float(measurement.dy),
                    float(measurement.dtheta),
                )
                noise = covariance_noise(measurement.covariance)
                graph.add(gtsam.BetweenFactorPose2(i, j, relative_pose, noise))

            elif factor.factor_type == "landmark":
                if len(factor.node_ids) != 1:
                    continue
                node_id = factor.node_ids[0]
                observation = factor.measurement
                pose_measurement = pose2_from_pose(observation.robot_pose_meas)
                noise = covariance_noise(observation.covariance)
                graph.add(gtsam.PriorFactorPose2(node_id, pose_measurement, noise))

        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
        result = optimizer.optimize()

        # Convert optimized GTSAM poses back into project Pose2D values.
        optimized_poses: dict[int, Pose2D] = {}
        for node_id in graph_manager.nodes:
            pose = result.atPose2(node_id)
            optimized_poses[node_id] = Pose2D(
                x=float(pose.x()),
                y=float(pose.y()),
                theta=float(pose.theta()),
            )

        return optimized_poses

    def should_optimize(self, num_new_keyframes: int, elapsed_time: float | None = None) -> bool:
        """Decide whether optimization should run."""
        if num_new_keyframes >= self.config.optimize_every_n_keyframes:
            return True
        if elapsed_time is not None and elapsed_time >= self.config.optimize_every_seconds:
            return True
        return False
