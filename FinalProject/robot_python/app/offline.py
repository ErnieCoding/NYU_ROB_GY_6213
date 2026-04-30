"""Offline replay runner for saved robot frames and landmark observations."""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np

from FinalProject.robot_python.app.slam import SLAMSystem
from FinalProject.robot_python.config.config import Config
from FinalProject.robot_python.data_types import EncoderState, LandmarkObservation, LidarScan, Pose2D, RobotFrame


class OfflineRunner:
    """Replay saved robot frames and processed landmarks into the SLAM system."""

    def __init__(
        self,
        config: Config | None = None,
        robot_pickle_path: str | Path | None = None,
        landmark_pickle_path: str | Path | None = None,
    ) -> None:
        self.config = config or Config()
        self.slam = SLAMSystem(self.config)
        log_dir = Path(self.config.runtime.log_dir)
        self.robot_pickle_path = Path(robot_pickle_path) if robot_pickle_path else log_dir / "robot_frames.pkl"
        self.landmark_pickle_path = (
            Path(landmark_pickle_path) if landmark_pickle_path else log_dir / "landmark_observations.pkl"
        )

    def run(self) -> None:
        """Replay robot frames and landmark observations in timestamp order."""
        for item in self.load_data():
            if isinstance(item, RobotFrame):
                self.slam.run_frontend(item)
            elif isinstance(item, LandmarkObservation):
                self.slam.add_landmark_observation(item)

            self.slam.run_backend()
            self._sleep_for_replay()

    def load_robot_frames(self) -> list[RobotFrame]:
        """Load robot frames from one pickle file."""
        if not self.robot_pickle_path.is_file():
            return []

        with self.robot_pickle_path.open("rb") as file:
            items = pickle.load(file)

        frames: list[RobotFrame] = []
        for item in items:
            if isinstance(item, RobotFrame):
                frames.append(item)
            elif isinstance(item, dict):
                frames.append(self._robot_frame_from_dict(item))
            else:
                raise TypeError(f"Unsupported robot frame item type: {type(item)!r}")

        return sorted(frames, key=lambda frame: frame.timestamp)

    def load_landmark_observations(self) -> list[LandmarkObservation]:
        """Load processed landmark observations from one pickle file."""
        if not self.landmark_pickle_path.is_file():
            return []

        with self.landmark_pickle_path.open("rb") as file:
            items = pickle.load(file)

        observations: list[LandmarkObservation] = []
        for item in items:
            if isinstance(item, LandmarkObservation):
                observations.append(item)
            elif isinstance(item, dict):
                observations.append(self._landmark_observation_from_dict(item))
            else:
                raise TypeError(f"Unsupported landmark observation item type: {type(item)!r}")

        return sorted(observations, key=lambda obs: obs.timestamp)

    def load_data(self) -> list[RobotFrame | LandmarkObservation]:
        """Load robot frames and landmark observations, then merge by timestamp."""
        data: list[RobotFrame | LandmarkObservation] = []
        data.extend(self.load_robot_frames())
        data.extend(self.load_landmark_observations())
        return sorted(data, key=lambda item: item.timestamp)

    def _robot_frame_from_dict(self, item: dict) -> RobotFrame:
        """Convert one project-specific pickle dictionary into a RobotFrame."""
        timestamp = float(item["timestamp"])

        encoder_item = item["encoder"]
        if isinstance(encoder_item, EncoderState):
            encoder = encoder_item
        else:
            # TODO: Confirm these dict keys match the final Arduino pickle schema.
            encoder = EncoderState(
                left_ticks=int(encoder_item["left_ticks"]),
                right_ticks=int(encoder_item["right_ticks"]),
                timestamp=float(encoder_item.get("timestamp", timestamp)),
            )

        lidar_item = item.get("lidar_scan")
        lidar_scan = None
        if isinstance(lidar_item, LidarScan):
            lidar_scan = lidar_item
        elif isinstance(lidar_item, dict):
            # TODO: Confirm LiDAR angle/range field names once logging is final.
            lidar_scan = LidarScan(
                ranges=list(lidar_item["ranges"]),
                angles=list(lidar_item["angles"]),
                timestamp=float(lidar_item.get("timestamp", timestamp)),
                frame_id=str(lidar_item.get("frame_id", "laser")),
                intensities=lidar_item.get("intensities"),
            )

        return RobotFrame(
            timestamp=timestamp,
            encoder=encoder,
            lidar_scan=lidar_scan,
            raw_packet=item.get("raw_packet"),
        )

    def _landmark_observation_from_dict(self, item: dict) -> LandmarkObservation:
        """Convert one project-specific dictionary into a LandmarkObservation."""
        pose_item = item["robot_pose_meas"]
        if isinstance(pose_item, Pose2D):
            robot_pose_meas = pose_item
        else:
            robot_pose_meas = Pose2D(
                x=float(pose_item.get("x", 0.0)),
                y=float(pose_item.get("y", 0.0)),
                theta=float(pose_item.get("theta", 0.0)),
            )

        return LandmarkObservation(
            timestamp=float(item["timestamp"]),
            marker_id=int(item["marker_id"]),
            robot_pose_meas=robot_pose_meas,
            covariance=np.asarray(item.get("covariance", np.eye(3)), dtype=float),
            quality=item.get("quality"),
        )

    def _sleep_for_replay(self) -> None:
        """Throttle replay slightly so logs are easier to inspect."""
        speed = max(self.config.runtime.offline_replay_speed, 1e-6)
        time.sleep(0.01 / speed)


def main() -> None:
    """Executable entry point for offline SLAM replay."""
    runner = OfflineRunner()
    runner.run()


if __name__ == "__main__":
    main()
