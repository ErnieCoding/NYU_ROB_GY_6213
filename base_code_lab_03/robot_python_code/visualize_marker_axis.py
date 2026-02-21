import cv2
import numpy as np
import parameters  # your existing parameters.py

import numpy as np
import cv2
import math

def wrap_to_pi(a):
    """Wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2*np.pi) - np.pi

def rvec_to_ypr(rvec):
    """
    Convert OpenCV rvec (Rodrigues) to yaw, pitch, roll using ZYX convention.
    Returns (yaw, pitch, roll) in radians.
    """
    rvec = np.asarray(rvec).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)

    # ZYX yaw-pitch-roll
    yaw = math.atan2(R[1, 0], R[0, 0])
    pitch = math.atan2(-R[2, 0], math.sqrt(R[2, 1]**2 + R[2, 2]**2))
    roll = math.atan2(R[2, 1], R[2, 2])

    return yaw, pitch, roll

def yaw_for_robot_forward_neg_x(yaw_marker):
    """
    If robot forward aligns with negative X axis, add pi to convert to robot heading.
    """
    return wrap_to_pi(yaw_marker + math.pi)




def main():
    # Pick a dictionary (must match the marker you printed)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

    # Detector params (handles OpenCV version differences)
    try:
        detector_params = cv2.aruco.DetectorParameters_create()
        detector = None
    except AttributeError:
        detector_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

    cap = cv2.VideoCapture(parameters.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera id {parameters.camera_id}")

    # Optional but often helps on Linux USB cams
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    print("Press ESC to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect markers
        if detector is None:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                frame, aruco_dict, parameters=detector_params
            )
        else:
            corners, ids, rejected = detector.detectMarkers(frame)

        if ids is not None and len(ids) > 0:
            # Draw marker boundaries + IDs
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Estimate pose for all detected markers
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners,
                parameters.marker_length,
                parameters.camera_matrix,
                parameters.dist_coeffs,
            )

            # Draw coordinate frame axes on each marker
            for i in range(len(ids)):
                cv2.drawFrameAxes(
                    frame,
                    parameters.camera_matrix,
                    parameters.dist_coeffs,
                    rvecs[i],
                    tvecs[i],
                    parameters.marker_length * 0.5,  # axis length
                )

                # Optional: print translation (in same units as marker_length)
                t = tvecs[i].reshape(3)
                r = rvecs[i].reshape(3)
       

                # Example usage:
                yaw, pitch, roll = rvec_to_ypr(r)
                theta_robot = yaw_for_robot_forward_neg_x(-yaw)
                print(f"yaw={math.degrees(yaw):.1f} deg, pitch={math.degrees(pitch):.1f} deg, roll={math.degrees(roll):.1f} deg")
                print(f"theta_robot (heading)={math.degrees(theta_robot):.1f} deg\n")
                #convert to degrees
                # print(f"rvec_deg=({np.degrees(r[0]):.1f}, {np.degrees(r[1]):.1f}, {np.degrees(r[2]):.1f})")

        cv2.imshow("ArUco detection + axes", frame)
        if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()