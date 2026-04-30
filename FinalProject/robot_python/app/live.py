"""Live-mode runner for UDP robot data and ESP32-CAM frames."""

from __future__ import annotations

import time
from collections.abc import Iterator

from FinalProject.robot_python.app.slam import SLAMSystem
from FinalProject.robot_python.config.config import Config
from FinalProject.robot_python.data_types import CameraFrame, RobotFrame


class LiveRunner:
    """Run the SLAM system against live asynchronous robot and camera inputs."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.slam = SLAMSystem(self.config)
        self.running = False

    def run(self) -> None:
        """Start the live processing loop."""
        self.running = True
        # TODO: Replace this simple sequential polling with asynchronous I/O,
        # queues, timestamp-aware buffering, and clean shutdown handling.
        robot_stream = self.receive_robot_frames()
        camera_stream = self.read_camera_frames()

        while self.running:
            robot_frame = next(robot_stream, None)
            if robot_frame is not None:
                self.slam.run_frontend(robot_frame)

            camera_frame = next(camera_stream, None)
            if camera_frame is not None:
                self.slam.detect_landmark(camera_frame)

            self.slam.run_backend()
            time.sleep(1.0 / max(self.config.runtime.robot_loop_hz, 1.0))

    def receive_robot_frames(self) -> Iterator[RobotFrame]:
        """Yield decoded robot frames from the Arduino UDP stream."""
        # TODO: Open a UDP socket on config.robot.udp_host/udp_port, decode
        # LiDAR ranges and encoder ticks, and yield RobotFrame instances.
        if False:
            yield

    def read_camera_frames(self) -> Iterator[CameraFrame]:
        """Yield camera frames from the ESP32-CAM stream."""
        # TODO: Use cv2.VideoCapture(config.camera.stream_url), timestamp each
        # frame, and yield CameraFrame objects into the landmark observer.
        if False:
            yield

    def stop(self) -> None:
        """Request a clean shutdown of the live runner."""
        self.running = False


def main() -> None:
    """Executable entry point for live SLAM."""
    runner = LiveRunner()
    runner.run()


if __name__ == "__main__":
    main()
