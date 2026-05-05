"""
Camera variance calibration script.
Robot must be STATIONARY at a known distance from the tag.
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import time
import parameters
import data_types

# ── CONFIG ──────────────────────────────────────────────────
TEST_MARKER_ID  = 0          # whichever tag is in front of you
NUM_SAMPLES     = 150        # frames to collect per distance
SAMPLE_DELAY    = 0.1        # seconds between samples (10 Hz)
# ────────────────────────────────────────────────────────────

def marker_to_robot_pose(marker_id, x_cm, z_cm):
    """Identical logic to CameraSensor.marker_to_robot_pose."""
    dx_robot = z_cm / 100.0
    dy_robot = -x_cm / 100.0
    x_robot  = parameters.tags[marker_id][0] / 100.0 - dx_robot
    y_robot  = parameters.tags[marker_id][1] / 100.0 - dy_robot
    return x_robot, y_robot

def collect_samples(cap, detector, marker_id, n_samples, delay):
    xs, ys, zs = [], [], []
    collected = 0

    print(f"\nCollecting {n_samples} samples for marker {marker_id}...")
    print("Make sure the robot is COMPLETELY STILL. Starting in 3 s...")
    time.sleep(3.0)

    while collected < n_samples:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARN] Frame drop — retrying...")
            time.sleep(0.05)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is None:
            continue

        for i, mid in enumerate(ids.flatten()):
            if mid != marker_id:
                continue

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[i]],
                parameters.marker_length,
                parameters.camera_matrix,
                parameters.dist_coeffs
            )
            tvec = tvecs[0][0]
            x_cm = tvec[0] * 100.0
            z_cm = tvec[2] * 100.0

            x_r, y_r = marker_to_robot_pose(marker_id, x_cm, z_cm)
            xs.append(x_r)
            ys.append(y_r)
            zs.append(tvec[2])  # raw depth in meters

            collected += 1
            if collected % 25 == 0:
                print(f"  {collected}/{n_samples} samples collected...")

        time.sleep(delay)

    return np.array(xs), np.array(ys), np.array(zs)

def report(distance_label, xs, ys, zs):
    mean_z = np.mean(zs)
    std_x  = np.std(xs)
    std_y  = np.std(ys)
    var_x  = np.var(xs)
    var_y  = np.var(ys)

    print(f"\n{'='*50}")
    print(f"  Distance label : {distance_label}")
    print(f"  Mean depth (z) : {mean_z:.3f} m")
    print(f"  σ_x            : {std_x*100:.2f} cm  →  var_x = {var_x:.6f} m²")
    print(f"  σ_y            : {std_y*100:.2f} cm  →  var_y = {var_y:.6f} m²")
    print(f"  Analytical σ   : {(0.6862 * mean_z / 351.8)*100:.2f} cm  (reference)")
    print(f"  Ratio empirical/analytical: {std_x / max((0.6862 * mean_z / 351.8), 1e-9):.2f}x")
    print(f"{'='*50}")

    return {
        "distance_label": distance_label,
        "mean_z_m": mean_z,
        "std_x_m": std_x,
        "std_y_m": std_y,
        "var_x_m2": var_x,
        "var_y_m2": var_y,
    }

# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
    assert cap.isOpened(), "Cannot open camera stream"

    aruco_dict  = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
    det_params  = aruco.DetectorParameters()
    detector    = aruco.ArucoDetector(aruco_dict, det_params)

    results = []
    distances = ["0.3m", "0.5m", "0.75m", "1.0m", "1.25m", "1.5m"]

    for label in distances:
        input(f"\n>>> Position robot at ~{label} from marker {TEST_MARKER_ID}."
              f" Press ENTER when ready...")
        xs, ys, zs = collect_samples(cap, detector, TEST_MARKER_ID,
                                     NUM_SAMPLES, SAMPLE_DELAY)
        results.append(report(label, xs, ys, zs))

    cap.release()

    # ── SUMMARY TABLE ──────────────────────────────────────
    print("\n\n{'='*60}")
    print("SUMMARY — use these values in get_camera_covariance()")
    print(f"{'Distance':<12} {'Mean z (m)':<14} {'σ_x (cm)':<12} {'σ_y (cm)':<12}")
    print("-" * 52)
    for r in results:
        print(f"{r['distance_label']:<12} {r['mean_z_m']:<14.3f}"
              f" {r['std_x_m']*100:<12.3f} {r['std_y_m']*100:<12.3f}")