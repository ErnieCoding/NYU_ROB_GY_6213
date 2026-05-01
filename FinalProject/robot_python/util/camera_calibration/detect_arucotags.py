import cv2
import numpy as np

# Load calibration
data = np.load("esp32_camera_calibration.npz")
camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]

# Marker real size: need to modify this if you used a different size when generating the markers
marker_length = 0.12  # meters = 12 cm

# ESP32 streamhttp://192.168.0.199
url = "http://192.168.4.1:81/stream"
cap = cv2.VideoCapture(url)

# ArUco setup
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
detector = cv2.aruco.ArucoDetector(aruco_dict)

while True:
    ret, frame = cap.read()

    if not ret:
        print("No frame received")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            marker_length,
            camera_matrix,
            dist_coeffs
        )

        for i in range(len(ids)):
            marker_id = ids[i][0]

            rvec = rvecs[i]
            tvec = tvecs[i][0]

            # Translation in cm
            x_cm = tvec[0] * 100
            y_cm = tvec[1] * 100
            z_cm = tvec[2] * 100

            # Rotation vector to rotation matrix
            R, _ = cv2.Rodrigues(rvec)

            # Approx yaw/pitch/roll in degrees
            yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
            pitch = np.degrees(np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2)))
            roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))

            print(
                f"ID {marker_id} | "
                f"X: {x_cm:.2f} cm, Y: {y_cm:.2f} cm, Z: {z_cm:.2f} cm | "
                f"Yaw: {yaw:.2f}°, Pitch: {pitch:.2f}°, Roll: {roll:.2f}°"
            )

            cv2.drawFrameAxes(
                frame,
                camera_matrix,
                dist_coeffs,
                rvec,
                tvecs[i],
                0.05
            )

            cv2.putText(
                frame,
                f"ID {marker_id}: X={x_cm:.1f} Y={y_cm:.1f} Z={z_cm:.1f} cm",
                (20, 40 + 30 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    cv2.imshow("ESP32 ArUco Pose", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()