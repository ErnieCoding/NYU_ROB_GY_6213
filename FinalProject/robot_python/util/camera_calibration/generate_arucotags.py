import cv2
import os
import numpy as np

output_dir = "arucotags_saved"
os.makedirs(output_dir, exist_ok=True)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)

marker_size_px = 500      # black ArUco marker size
margin_px = 20           # white margin around marker
border_bits = 1
num_markers = 20

for marker_id in range(num_markers):
    marker_img = cv2.aruco.generateImageMarker(
        aruco_dict,
        marker_id,
        marker_size_px,
        borderBits=border_bits
    )

    marker_with_margin = cv2.copyMakeBorder(
        marker_img,
        margin_px, margin_px, margin_px, margin_px,
        cv2.BORDER_CONSTANT,
        value=255
    )

    filename = os.path.join(
        output_dir,
        f"aruco_id_{marker_id}_12cm_with_margin.png"
    )

    cv2.imwrite(filename, marker_with_margin)
    print(f"Saved: {filename}")

print("ArUco marker generation complete.")