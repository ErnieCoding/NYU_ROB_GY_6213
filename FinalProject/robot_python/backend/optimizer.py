"""Graph optimizer using GTSAM Pose2 factors."""

from __future__ import annotations

import math
import gtsam
import numpy as np

from FinalProject.robot_python.backend.pose_graph import PoseGraphManager
from FinalProject.robot_python.config.config import BackendConfig
from FinalProject.robot_python.data_types import Pose2D, normalize_angle


class GraphOptimizer:
    def __init__(self, config: BackendConfig) -> None:
        self.config = config

    def optimize(self, graph_manager: PoseGraphManager) -> dict[int, Pose2D]:
        if not graph_manager.nodes:
            return {}

        def pose2_from_pose(pose: Pose2D) -> gtsam.Pose2:
            return gtsam.Pose2(float(pose.x), float(pose.y), float(pose.theta))

        def covariance_noise(covariance) -> gtsam.noiseModel.Base:
            if covariance is None:
                covariance = np.diag([0.10**2, 0.10**2, math.radians(8.0) ** 2])
            covariance = np.asarray(covariance, dtype=float)
            covariance = 0.5 * (covariance + covariance.T)
            covariance += 1e-9 * np.eye(covariance.shape[0])
            return gtsam.noiseModel.Gaussian.Covariance(covariance)

        def robust_noise(covariance, huber_k: float = 1.345) -> gtsam.noiseModel.Base:
            base = covariance_noise(covariance)
            return gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber(huber_k),
                base,
            )

        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()

        for node_id, node in graph_manager.nodes.items():
            initial.insert(node_id, pose2_from_pose(node.pose.pose))

        first_node_id = min(graph_manager.nodes)
        first_pose = graph_manager.nodes[first_node_id].pose.pose
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([1e-3, 1e-3, math.radians(0.1)], dtype=float)
        )
        graph.add(gtsam.PriorFactorPose2(first_node_id, pose2_from_pose(first_pose), prior_noise))

        for factor in graph_manager.factors:
            if factor.factor_type in {"odometry", "lidar"}:
                if len(factor.node_ids) != 2:
                    continue
                i, j = factor.node_ids
                m = factor.measurement
                z = gtsam.Pose2(float(m.dx), float(m.dy), float(m.dtheta))
                graph.add(gtsam.BetweenFactorPose2(i, j, z, covariance_noise(m.covariance)))

            elif factor.factor_type == "loop_closure":
                if len(factor.node_ids) != 2:
                    continue
                i, j = factor.node_ids
                m = factor.measurement
                z = gtsam.Pose2(float(m.dx), float(m.dy), float(m.dtheta))
                graph.add(gtsam.BetweenFactorPose2(i, j, z, robust_noise(m.covariance)))

            elif factor.factor_type == "landmark":
                if len(factor.node_ids) != 1:
                    continue
                node_id = factor.node_ids[0]
                obs = factor.measurement
                z = pose2_from_pose(obs.robot_pose_meas)
                graph.add(gtsam.PriorFactorPose2(node_id, z, covariance_noise(obs.covariance)))

        num_nodes = len(graph_manager.nodes)
        num_factors = len(graph_manager.factors)
        factor_summary = graph_manager.get_graph_summary()

        initial_error = None
        if hasattr(graph, "error"):
            try:
                initial_error = float(graph.error(initial))
            except Exception as exc:
                print(f"[OPTIMIZER][WARN] failed to compute initial error: {exc}")

        print(
            f"[OPTIMIZER][GRAPH] nodes={num_nodes} factors={num_factors} "
            f"summary={factor_summary}"
        )
        if initial_error is not None:
            print(f"[OPTIMIZER][ERROR] initial={initial_error:.6f}")
        
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosityLM("SILENT")
        params.setMaxIterations(int(getattr(self.config, "max_iterations", 100)))
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
        result = optimizer.optimize()

        final_error = None
        if hasattr(graph, "error"):
            try:
                final_error = float(graph.error(result))
            except Exception as exc:
                print(f"[OPTIMIZER][WARN] failed to compute final error: {exc}")

        if initial_error is not None and final_error is not None:
            print(
                f"[OPTIMIZER][ERROR] final={final_error:.6f} "
                f"delta={initial_error - final_error:.6f}"
            )
        
        
        optimized_poses: dict[int, Pose2D] = {}
        for node_id in graph_manager.nodes:
            pose = result.atPose2(node_id)
            optimized_poses[node_id] = Pose2D(
                x=float(pose.x()),
                y=float(pose.y()),
                theta=normalize_angle(float(pose.theta())),
            )

        return optimized_poses

    def should_optimize(self, num_new_keyframes: int, elapsed_time: float | None = None) -> bool:
        if num_new_keyframes >= self.config.optimize_every_n_keyframes:
            return True
        if elapsed_time is not None and elapsed_time >= self.config.optimize_every_seconds:
            return True
        return False