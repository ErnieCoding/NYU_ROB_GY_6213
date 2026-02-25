############## visualize_pose_realtime.py: 2x2 dashboard
# Top-left: 2D camera frame (raw tvec x,y)
# Top-right: 2D world frame (transformed x_world,y_world)
# Bottom-left: 3D pose (camera origin + marker pose)
# Bottom-right: camera view

import cv2
import numpy as np
import parameters
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


# -----------------------------
# Helpers
# -----------------------------
def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rvec_to_ypr(rvec):
    rvec = np.asarray(rvec).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    yaw = math.atan2(R[1, 0], R[0, 0])
    pitch = math.atan2(-R[2, 0], math.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    roll = math.atan2(R[2, 1], R[2, 2])
    return yaw, pitch, roll


def yaw_for_robot_forward_neg_x(yaw_marker):
    # keep your chosen convention
    return wrap_to_pi(yaw_marker + math.pi / 2)


# -----------------------------
# Your transform (as requested)
# -----------------------------
def transform_camera_to_world(tvec, rvec, tvec_init, rvec_init):
    tvec = np.array(tvec, dtype=np.float32).reshape(3)
    rvec = np.array(rvec, dtype=np.float32).reshape(3)
    t_init = np.array(tvec_init, dtype=np.float32).reshape(3)
    r_init = np.array(rvec_init, dtype=np.float32).reshape(3)

    # 1) Rotation matrix at init
    R_init, _ = cv2.Rodrigues(r_init)

    # 2) Position transform (your method)
    t_diff = tvec - t_init
    p_world_meters = R_init.T @ t_diff

    # 3) meters -> cm
    x_world = p_world_meters[0] * 100.0
    y_world = p_world_meters[1] * 100.0

    # 4) Heading transform
    R_curr, _ = cv2.Rodrigues(rvec)
    R_rel = R_init.T @ R_curr
    yaw_rad = np.arctan2(R_rel[1, 0], R_rel[0, 0])
    theta_world = np.degrees(yaw_rad)
    theta_world = (theta_world + 180) % 360 - 180

    # Return with your sign convention
    return x_world, y_world, theta_world


# -----------------------------
# Matplotlib -> Image
# -----------------------------
def fig_to_bgr(fig: Figure) -> np.ndarray:
    canvas = FigureCanvas(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())  # (H, W, 4)
    rgb = rgba[:, :, :3].copy()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def set_2d_limits(ax, pts2d, margin):
    pts = np.array(pts2d, dtype=float)
    mins = pts.min(axis=0) - margin
    maxs = pts.max(axis=0) + margin
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])


def make_2d_plot(tile_w, tile_h, title, xlabel, ylabel,
                 marker_xy=None, camera_at_origin=True,
                 units_text="", fixed_margin=None):
    """
    Generic 2D plot tile:
      - optionally plot camera at (0,0)
      - plot marker point (x,y)
    """
    dpi = 100
    fig = Figure(figsize=(tile_w / dpi, tile_h / dpi), dpi=dpi)
    ax = fig.add_subplot(111)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")

    pts = []

    if camera_at_origin:
        ax.scatter(0.0, 0.0, s=150)
        ax.text(0.0, 0.0, " Camera", fontsize=10)
        pts.append((0.0, 0.0))

        axis_len = 10.0 if "cm" in units_text else parameters.marker_length * 1.0
        ax.plot([0, axis_len], [0, 0], linewidth=2)
        ax.plot([0, 0], [0, axis_len], linewidth=2)
        ax.text(axis_len, 0, " +X", fontsize=9)
        ax.text(0, axis_len, " +Y", fontsize=9)

    if marker_xy is not None:
        mx, my = marker_xy
        ax.scatter(mx, my, s=200)
        ax.text(mx, my, f" Marker\n ({mx:.2f},{my:.2f}) {units_text}", fontsize=10)
        pts.append((mx, my))

    # limits
    if len(pts) == 0:
        pts = [(0.0, 0.0)]

    if fixed_margin is None:
        # default: use marker_length-scaled margin (meters) or 20cm if cm plot
        fixed_margin = 20.0 if "cm" in units_text else parameters.marker_length * 2.0

    set_2d_limits(ax, pts, margin=fixed_margin)

    bgr = fig_to_bgr(fig)
    plt.close(fig)
    return bgr


def make_3d_plot(tile_w, tile_h, rvec=None, tvec=None):
    dpi = 100
    fig = Figure(figsize=(tile_w / dpi, tile_h / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    ax.set_title("3D: Marker pose in camera frame")
    ax.set_xlabel("X (right)")
    ax.set_ylabel("Y (down)")
    ax.set_zlabel("Z (forward)")
    ax.grid(True)

    axis_len = parameters.marker_length * 1.2

    # Camera origin + axes
    ax.scatter([0], [0], [0], s=80)
    ax.text(0, 0, 0, "Camera", fontsize=9)
    ax.plot([0, axis_len], [0, 0], [0, 0], linewidth=2)
    ax.plot([0, 0], [0, axis_len], [0, 0], linewidth=2)
    ax.plot([0, 0], [0, 0], [0, axis_len], linewidth=2)

    pts = np.array([[0.0, 0.0, 0.0]], dtype=float)

    if rvec is not None and tvec is not None:
        t = np.asarray(tvec).reshape(3).astype(float)
        R, _ = cv2.Rodrigues(np.asarray(rvec).reshape(3, 1))

        ax.scatter([t[0]], [t[1]], [t[2]], s=100)
        ax.text(t[0], t[1], t[2], "Marker", fontsize=9)

        m_len = parameters.marker_length * 0.8
        x_end = t + (R @ np.array([m_len, 0.0, 0.0]))
        y_end = t + (R @ np.array([0.0, m_len, 0.0]))
        z_end = t + (R @ np.array([0.0, 0.0, m_len]))

        ax.plot([t[0], x_end[0]], [t[1], x_end[1]], [t[2], x_end[2]], linewidth=2)
        ax.plot([t[0], y_end[0]], [t[1], y_end[1]], [t[2], y_end[2]], linewidth=2)
        ax.plot([t[0], z_end[0]], [t[1], z_end[1]], [t[2], z_end[2]], linewidth=2)

        pts = np.vstack([pts, t.reshape(1, 3)])

    margin = parameters.marker_length * 2.0
    mins = pts.min(axis=0) - margin
    maxs = pts.max(axis=0) + margin
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(max(0.0, mins[2]), maxs[2])
    ax.view_init(elev=20, azim=-60)

    bgr = fig_to_bgr(fig)
    plt.close(fig)
    return bgr


# -----------------------------
# Main
# -----------------------------
def main():
    # Hardcode your init pose here (meters / radians output from ArUco)
    ##THIS CHANGES EVERY TIME YOY CHANGE POSITION OF CAMERA, SO UPDATE BEFORE RUNNING EACH TIME
    tvec_init = [-0.21500975, 0.56345664, 0.87650994]
    rvec_init = [2.72888006, 0.45845892, 0.90720317]

    # Raw tvec: [-0.21500975  0.56345664  0.87650994], Raw rvec: [2.72888006 0.45845892 0.90720317]

    tile_w = 520
    tile_h = 420

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

    try:
        detector_params = cv2.aruco.DetectorParameters_create()
        detector = None
    except AttributeError:
        detector_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

    cap = cv2.VideoCapture(parameters.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera id {parameters.camera_id}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    print("Press ESC to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        best_rvec = None
        best_tvec = None

        if detector is None:
            corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=detector_params)
        else:
            corners, ids, _ = detector.detectMarkers(frame)

        # Defaults for plots
        cam_xy = None            # meters in camera frame
        world_xy_cm = None       # cm in world frame
        theta_world = None

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners,
                parameters.marker_length,
                parameters.camera_matrix,
                parameters.dist_coeffs,
            )

            best_rvec = rvecs[0]
            best_tvec = tvecs[0]

            cv2.drawFrameAxes(
                frame,
                parameters.camera_matrix,
                parameters.dist_coeffs,
                best_rvec,
                best_tvec,
                parameters.marker_length * 0.5,
            )

            t = np.asarray(best_tvec).reshape(3)
            r = np.asarray(best_rvec).reshape(3)

            cam_xy = (float(t[0]), float(t[1]))

            # Transform to world
            xw, yw, thw = transform_camera_to_world(t, r, tvec_init, rvec_init)
            ##print raw values and transformed values
            print(f"Raw tvec: {t}, Raw rvec: {r}")
            print(f"Transformed world: x={xw:.1f}cm y={yw:.1f}cm th={thw:.1f}deg")
            world_xy_cm = (float(xw), float(yw))
            theta_world = float(thw)

            # Overlay text on camera tile
            cv2.putText(frame, f"tvec: {t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f} (m)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, f"world: x={xw:.1f}cm y={yw:.1f}cm th={thw:.1f}deg",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 0), 2)

        # Top-left: raw camera XY (meters)
        plot_cam = make_2d_plot(
            tile_w, tile_h,
            title="2D Camera Frame (raw): +Y down",
            xlabel="X (m)", ylabel="Y (m)",
            marker_xy=cam_xy,
            camera_at_origin=True,
            units_text="m",
            fixed_margin=parameters.marker_length * 2.0
        )

        # Top-right: WORLD XY (cm)
        plot_world = make_2d_plot(
            tile_w, tile_h,
            title="2D World Frame (transformed): +Y up",
            xlabel="X (cm)", ylabel="Y (cm)",
            marker_xy=world_xy_cm,
            camera_at_origin=True,        # world origin at init -> camera label ok as origin
            units_text="cm",
            fixed_margin=50.0             # adjust viewing window in cm
        )

        # Bottom-left: 3D camera frame plot
        plot_3d = make_3d_plot(tile_w, tile_h, rvec=best_rvec, tvec=best_tvec)

        # Bottom-right: camera view tile
        cam_tile = cv2.resize(frame, (tile_w, tile_h))

        top_row = np.hstack([plot_cam, plot_world])
        bot_row = np.hstack([plot_3d, cam_tile])
        combined = np.vstack([top_row, bot_row])

        cv2.imshow("Frames Debug: (Top) camera XY vs world XY | (Bottom) 3D vs camera", combined)

        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

