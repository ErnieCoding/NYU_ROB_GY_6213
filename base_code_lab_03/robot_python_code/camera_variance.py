import math
from matplotlib import pyplot as plt
import cv2
import numpy as np
import parameters  # your existing parameters.py


def wrap_to_pi(a):
    """Wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2*np.pi) - np.pi

def wrap_deg(a):
    return (a+180)%360-180

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

# Initial Vector
# PKL file: robot_data_0_0_20_02_26_16_29_58.pkl
tvec_init = [-1.4739865624232023, 0.2944068562536831, 2.484327773897269]
rvec_init = [0.04317648493269671, 2.687062358695416, -0.9727888677264434]
x_init = 0
y_init = 0
angle_init = 0


# Position 1
# PKL file: robot_data_0_0_20_02_26_16_37_01.pkl
tvec_1 = [-1.3240484938041428, -0.49534195560907995, 3.2369861123021564]
rvec_1 = [-1.558489061671923, -3.1386173944018685, 1.0013127444257046]
x_1 = 33
y_1 = 94
angle_1 = 45

# Position 2
# PKL file: robot_data_0_0_20_02_26_16_49_40.pkl
# TVEC -0.9012926461424521, -0.8232292674937541, 3.6174280933881757
# RVEC 1.8241900607244028, 1.559808614842489, -0.35557626959366384
tvec_2 = [-0.9012926461424521, -0.8232292674937541, 3.6174280933881757]
rvec_2 = [1.8241900607244028, 1.559808614842489, -0.35557626959366384]
x_2 = 84
y_2 = 129
angle_2 = 90

# Position 3
# PKL file: robot_data_0_0_20_02_26_17_00_19.pkl
# TVEC -0.3651284810453896, -0.4998885657015005, 3.8463680046375543
# RVEC -1.9879639761252201, 2.4206902243542054, -1.2617237632966203
tvec_3 = [-0.3651284810453896, -0.4998885657015005, 3.8463680046375543]
rvec_3 = [-1.9879639761252201, 2.4206902243542054, -1.2617237632966203]
x_3 = 134
y_3 = 94
angle_3 = -90

# Position 4
# PKL file: robot_data_0_0_20_02_26_17_07_09.pkl
# TVEC -0.3680792525182147, 0.14069471389352536, 2.98313773498547
# RVEC 2.503962865948892, -0.1737895207141775, 0.4173554855256833
tvec_4 = [-0.3680792525182147, 0.14069471389352536, 2.98313773498547]
rvec_4 = [2.503962865948892, -0.1737895207141775, 0.4173554855256833]
x_4 = 111
y_4 = 23
angle_4 = 180

# Position 5
# PKL file: robot_data_0_0_20_02_26_17_11_51.pkl
# TVEC -1.1923970485559425, -0.0006937292076095742, 2.825754482819625
# RVEC 2.2410686879870765, 0.980847097017905, -0.12678033094149121
tvec_5 = [-1.1923970485559425, -0.0006937292076095742, 2.825754482819625]
rvec_5 = [2.2410686879870765, 0.980847097017905, -0.12678033094149121]
x_5 = 33
y_5 = 45
angle_5 = 135


x_measured = [x_1, x_2, x_3, x_4, x_5]
y_measured = [y_1, y_2, y_3, y_4, y_5]

tvec = [tvec_1, tvec_2, tvec_3, tvec_4, tvec_5]
rvec = [rvec_1, rvec_2, rvec_3, rvec_4, rvec_5]

angles = [angle_1, angle_2, angle_3, angle_4, angle_5]

tvec_init_x = tvec_init[0]
tvec_init_y = tvec_init[1]

sum_x = 0
sum_y = 0
sum_theta = 0


x_camera = []
y_camera = []
for i in range(len(x_measured)):
    print("-------------------------------")
    print(f"VALUE: {i+1}\n\n")
    print(f"\nTRUE ANGLE: {angles[i]}")
    yaw, _, _ = rvec_to_ypr(rvec[i])
    theta = -(yaw_for_robot_forward_neg_x(yaw))
    theta = math.degrees(theta)
    print(f"THETA CAMERA: {theta}")

    theta_difference = wrap_deg(angles[i] - theta)
    print(f"THETA DIFFERENCE: {theta_difference}")

    sum_theta += theta_difference ** 2
    print(f"SUM THETA: {sum_theta}\n")

    # print(f"MEASURED X: {x_measured[i] / 100}\n")
    # print(f"MEASURED Y: {y_measured[i] / 100}\n\n")
    tvec_diff_x = abs(tvec_init_x - tvec[i][0])
    tvec_diff_y = abs(tvec_init_y - tvec[i][1])
    # print(f"TVEC_DIFF_X: {tvec_diff_x}\n")
    # print(f"TVEC_DIFF_Y: {tvec_diff_y}\n\n")

    x_camera.append(tvec_diff_x * 100)
    y_camera.append(tvec_diff_y * 100)

    x_difference = (x_measured[i]/100) - tvec_diff_x
    y_difference = (y_measured[i]/100) - tvec_diff_y

    # print(f"X_DIFF: {x_difference}\n")
    # print(f"Y_DIFF: {y_difference}\n\n")

    sum_x += x_difference ** 2
    sum_y += y_difference ** 2

    # print(f"SUM X: {sum_x}\n")
    # print(f"SUM Y: {sum_y}\n\n")
    print("-------------------------------")



plt.figure(figsize=(12, 12))
plt.subplot(3,3,1)
plt.scatter(x = x_measured, y=x_camera)
plt.xlabel("X Measured")
plt.ylabel("Camera X Measured")
plt.savefig("plot.png")

var_x = (sum_x/(len(x_measured) - 1))
var_y = (sum_y/(len(x_measured) - 1))
var_theta = (sum_theta/(len(angles) - 1))

print(f"Variance X: {var_x}\n")
print(f"Variance Y: {var_y}\n")
print(f"Variance Theta: {var_theta}")

# Variance X: 0.03991543833753128

# Variance Y: 0.0258320538210973

# Variance Theta: 194.3414817275661