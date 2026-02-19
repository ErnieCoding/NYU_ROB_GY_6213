import numpy as np
import cv2 as cv
import glob

def detect_chessboard():
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((7*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)


    images = glob.glob('./calibration_images_webcam/*.jpg')
    # images = images[0:6]

    print(f"Processing {len(images)} images\n\n\n")

    gray_global = None

    for fname in images:
        img = cv.imread(fname)

        # print("Processing: {fname}\n")
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray_global = gray

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (7,7), None)

        # If found, add object points, image points (after refining them)
        
        if ret == True:
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv.drawChessboardCorners(img, (7,7), corners2, ret)
            # cv.imshow('img', img)
            # cv.imwrite(f"calibrated_images/calibrated_{fname.split("/")[2]}", img)
            # cv.waitKey(500)

    cv.destroyAllWindows()

    return objpoints, imgpoints, gray_global


def calibrate_camera(objpoints, imgpoints, gray):
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs

def generate_aruco():
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)

    image_obj = cv.aruco.generateImageMarker(aruco_dict, 23, 500)
    cv.imwrite("robot_marker_23_500.png", image_obj)


if __name__ == "__main__":
    objpoints, imgpoints, gray = detect_chessboard()

    ret, mtx, dist, rvecs, tvecs = calibrate_camera(objpoints, imgpoints, gray)

    print("Generating aruco tag...")
    generate_aruco()

    # print(f"REPROJECTION ERROR: {ret}\n\n\n")
    # print(mtx)
    # print("\n\n\n\nDISTANCE COEFFICIENT")
    # print(dist)

    # mean_error = 0
    # for i in range(len(objpoints)):
    #     imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    #     error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    #     mean_error += error

    # print( "total error: {}".format(mean_error/len(objpoints)) )



# Camera Matrix + Distance Coefficients + Reprojection Error values
# Processing 20 images



# REPROJECTION ERROR: 1.9144267245989395



# [[1.03843829e+03 0.00000000e+00 5.70058553e+02]
#  [0.00000000e+00 1.06325837e+03 3.09600558e+02]
#  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]




# DISTANCE COEFFICIENT
# [[-0.41925883  0.32857265  0.00174434 -0.00148671 -0.21424311]]
# total error: 0.22999579741542794


    