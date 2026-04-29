# ###CODE 1: FIRST SAVE MULTIPLE IMAGES OF ARUCO MARKERS TO A FOLDER
# import cv2
# import os

# url = "http://192.168.0.200:81/stream"
# cap = cv2.VideoCapture(url)

# save_dir = "captured_frames"
# os.makedirs(save_dir, exist_ok=True)

# img_count = 0
# clicked = False

# button_top_left = (20, 20)
# button_bottom_right = (140, 70)

# latest_clean_frame = None

# def mouse_callback(event, x, y, flags, param):
#     global clicked
#     if event == cv2.EVENT_LBUTTONDOWN:
#         if (button_top_left[0] <= x <= button_bottom_right[0] and
#             button_top_left[1] <= y <= button_bottom_right[1]):
#             clicked = True

# cv2.namedWindow("ESP32-CAM")
# cv2.setMouseCallback("ESP32-CAM", mouse_callback)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("No frame received")
#         break

#     # Save this clean version, with NO button
#     latest_clean_frame = frame.copy()

#     # Draw button only on display copy
#     display_frame = frame.copy()
#     cv2.rectangle(display_frame, button_top_left, button_bottom_right, (0, 255, 0), -1)
#     cv2.putText(display_frame, "SAVE", (35, 55),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

#     if clicked:
#         filename = os.path.join(save_dir, f"frame_{img_count}.png")
#         cv2.imwrite(filename, latest_clean_frame)
#         print(f"Saved clean image: {filename}")
#         img_count += 1
#         clicked = False

#     cv2.imshow("ESP32-CAM", display_frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()


#####CODE 2: CALIBRATE THE CAMERA:
import cv2
import numpy as np
import glob
import os

# =========================
# SETTINGS
# =========================

image_folder = "captured_frames"
image_paths = glob.glob(os.path.join(image_folder, "*.png"))

# IMPORTANT:
# This is number of INNER corners, not number of squares.
pattern_size = (7, 7)

# Your measured square size
square_size = 0.021  # meters, because 2.1 cm = 0.021 m

output_file = "esp32_camera_calibration.npz"

# =========================
# PREPARE OBJECT POINTS
# =========================

# Example:
# (0,0,0), (0.021,0,0), (0.042,0,0), ...
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)

objp[:, :2] = np.mgrid[
    0:pattern_size[0],
    0:pattern_size[1]
].T.reshape(-1, 2)

objp *= square_size

objpoints = []  # 3D real-world points
imgpoints = []  # 2D image points

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

# =========================
# FIND CHESSBOARD CORNERS
# =========================

valid_images = 0
image_size = None

for path in image_paths:
    img = cv2.imread(path)

    if img is None:
        print(f"Could not read image: {path}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]

    # ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    ret, corners = cv2.findChessboardCornersSB(
    gray,
    pattern_size,
    flags=cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
    )

    if ret:
        valid_images += 1

        # corners_refined = cv2.cornerSubPix(
        #     gray,
        #     corners,
        #     (11, 11),
        #     (-1, -1),
        #     criteria
        # )

        objpoints.append(objp)
        imgpoints.append(corners)

        display = img.copy()
        cv2.drawChessboardCorners(display, pattern_size, corners, ret)
        cv2.imshow("Detected Corners", display)
        cv2.waitKey(300)

        print(f"Detected corners: {path}")
    else:
        print(f"Failed to detect corners: {path}")

cv2.destroyAllWindows()

print("\n=========================")
print(f"Total images: {len(image_paths)}")
print(f"Valid calibration images: {valid_images}")
print("=========================\n")

if valid_images < 10:
    print("Warning: You should ideally use at least 10 good chessboard images.")

if valid_images == 0:
    raise RuntimeError("No valid chessboard detections. Check pattern_size.")

# =========================
# CAMERA CALIBRATION
# =========================

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    image_size,
    None,
    None
)

print("Calibration RMS returned by OpenCV:")
print(ret)

print("\nCamera matrix K:")
print(camera_matrix)

print("\nDistortion coefficients:")
print(dist_coeffs)

# =========================
# REPROJECTION ERROR
# =========================

total_squared_error = 0
total_points = 0
per_image_errors = []

for i in range(len(objpoints)):
    projected_points, _ = cv2.projectPoints(
        objpoints[i],
        rvecs[i],
        tvecs[i],
        camera_matrix,
        dist_coeffs
    )

    error = imgpoints[i] - projected_points
    squared_error = np.sum(error ** 2)

    num_points = len(projected_points)

    rmse_image = np.sqrt(squared_error / num_points)
    per_image_errors.append(rmse_image)

    total_squared_error += squared_error
    total_points += num_points

overall_rmse = np.sqrt(total_squared_error / total_points)

print("\n=========================")
print("Reprojection Error")
print("=========================")
print(f"Overall RMSE: {overall_rmse:.4f} pixels")

for i, err in enumerate(per_image_errors):
    print(f"Image {i}: {err:.4f} pixels")

# =========================
# SAVE CALIBRATION
# =========================

np.savez(
    output_file,
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    rvecs=rvecs,
    tvecs=tvecs,
    reprojection_rmse=overall_rmse,
    square_size=square_size,
    pattern_size=pattern_size
)

print(f"\nSaved calibration to: {output_file}")



# # #ATTEMPT 1 
# # Final images used: 69
# # Final overall RMSE: 0.5206 px
# # OpenCV RMS: 0.5206

# # Camera matrix K:
# # [[318.30270036   0.         170.74196247]
# #  [  0.         326.17928769 107.10260496]
# #  [  0.           0.           1.        ]]

# # Distortion coefficients:
# # [[-0.16213245  0.78018706 -0.02560211  0.01187712  4.39754496]]


##ATTEMPT 2:
# FINAL CALIBRATION RESULT
# ==============================
# Final images used: 97
# Final overall RMSE: 0.6862 px
# OpenCV RMS: 0.6862

# Camera matrix K:
# [[348.0321225    0.         188.28539766]
#  [  0.         355.6030584  105.48395813]
#  [  0.           0.           1.        ]]

# Distortion coefficients:
# [[ 0.08988576 -2.60316437 -0.02748353  0.02115694 13.16262713]]


##FINAL CALIBRATION RESULT:
# Calibration RMS returned by OpenCV:
# 0.6862406713509189

# Camera matrix K:
# [[348.0321225    0.         188.28539766]
#  [  0.         355.6030584  105.48395813]
#  [  0.           0.           1.        ]]

# Distortion coefficients:
# [[ 0.08988576 -2.60316437 -0.02748353  0.02115694 13.16262713]]

# =========================
# Reprojection Error
# =========================
# Overall RMSE: 0.6862 pixels
