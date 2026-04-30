"""Offline replay runner for saved robot pickle data and camera images."""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import cv2

from FinalProject.robot_python.app.slam import SLAMSystem
from FinalProject.robot_python.config.config import Config
from FinalProject.robot_python.data_types import CameraFrame, EncoderState, LidarScan, RobotFrame


class OfflineRunner:
    """Replay saved LiDAR/wheel frames and camera images into the SLAM system."""

    def __init__(
        self,
        config: Config | None = None,
        robot_pickle_path: str | Path | None = None,
        camera_image_dir: str | Path | None = None,
    ) -> None:
        self.config = config or Config()
        self.slam = SLAMSystem(self.config)
        log_dir = Path(self.config.runtime.log_dir)
        self.robot_pickle_path = Path(robot_pickle_path) if robot_pickle_path else log_dir / "robot_frames.pkl"
        self.camera_image_dir = Path(camera_image_dir) if camera_image_dir else log_dir / "camera"

    def run(self) -> None:
        """Replay robot and camera data in timestamp order."""
        for item in self.load_data():
            if isinstance(item, RobotFrame):
                self.slam.run_frontend(item)
            elif isinstance(item, CameraFrame):
                self.slam.detect_landmark(item)

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

    def load_camera_frames(self) -> list[CameraFrame]:
        """Load camera frames from timestamp-named image files."""
        if not self.camera_image_dir.is_dir():
            return []

        frames: list[CameraFrame] = []
        for path in sorted(self.camera_image_dir.iterdir()):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            try:
                timestamp = float(path.stem)
            except ValueError:
                continue

            image = cv2.imread(str(path))
            if image is None:
                continue

            frames.append(CameraFrame(timestamp=timestamp, image=image))

        return sorted(frames, key=lambda frame: frame.timestamp)

    def load_data(self) -> list[RobotFrame | CameraFrame]:
        """Load robot and camera data, then merge by timestamp."""
        data: list[RobotFrame | CameraFrame] = []
        data.extend(self.load_robot_frames())
        data.extend(self.load_camera_frames())
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
