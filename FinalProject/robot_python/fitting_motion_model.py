import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# YOUR DATA (copy as is)
# -----------------------------
files_and_data = [
    # filename, encoder_left_start, encoder_right_start, encoder_left_finish, encoder_right_finish, x_finish, y_finish, theta_start, theta_finish
   {"motion_type:": "straight",
    "filename": "robot_data_0_0_02_05_26_23_59_25.pkl",
    "encoder_left_start": 47944,
    "encoder_right_start": 48572,
    "encoder_left_finish": 49341,
    "encoder_right_finish": 49956,
    "x_finish": 54,
    "y_finish": 0,
    "theta_start": 102,
    "theta_finish": 100
   },
   {"motion_type:": "straight",
    "filename": "robot_data_0_0_03_05_26_00_00_53.pkl",
    "encoder_left_start": 49341,
    "encoder_right_start": 49956,
    "encoder_left_finish": 52639,
    "encoder_right_finish": 53248,
    "x_finish": 125,
    "y_finish": 0,
    "theta_start": 102,
    "theta_finish": 99
   },
   {
    "motion_type:": "straight",
    "filename": "robot_data_0_0_03_05_26_00_04_04.pkl",
    "encoder_left_start": 57586,
    "encoder_right_start": 58048,
    "encoder_left_finish": 62392,
    "encoder_right_finish": 62697,
    "x_finish": 178,
    "y_finish": -18,
    "theta_start": 103,
    "theta_finish": 111
   },
   {"motion_type:": "straight",
    "filename": "robot_data_0_0_03_05_26_00_06_49.pkl",
    "encoder_left_start": 62397,
    "encoder_right_start": 62703,
    "encoder_left_finish": 69092,
    "encoder_right_finish": 69245,
    "x_finish": 244,
    "y_finish": -5,
    "theta_start": 104,
    "theta_finish": 110,

   },
   {"motion_type:": "straight",
    "filename": "robot_data_0_0_03_05_26_00_09_46.pkl",
    "encoder_left_start": 69128,
    "encoder_right_start": 69260,
    "encoder_left_finish": 77339,
    "encoder_right_finish": 77266,
    "x_finish": 301,
    "y_finish": -7,
    "theta_start": 103,
    "theta_finish": 114
   },
   {"motion_type:": "moving_in_arc",
    "filename": "robot_data_0_0_03_05_26_00_12_09.pkl",
    "encoder_left_start": 77339,
    "encoder_right_start": 77269,
    "encoder_left_finish": 88563,
    "encoder_right_finish": 88004,
    "x_finish": 378,
    "y_finish": 112,
    "theta_start": 105,
    "theta_finish": 134
   },
   {"motion_type:": "moving_in_arc",
    "filename": "robot_data_0_0_03_05_26_00_15_48.pkl",
    "encoder_left_start": 88565,
    "encoder_right_start": 88010,
    "encoder_left_finish": 91788,
    "encoder_right_finish": 91185,
    "x_finish": 95,
    "y_finish": 74,
    "theta_start": 68,
    "theta_finish": 70
   },
   {"motion_type:": "moving_in_arc",
    "filename": "robot_data_0_0_03_05_26_00_16_58.pkl", # starting angle unsure
    "encoder_left_start": 91788,
    "encoder_right_start": 91185,
    "encoder_left_finish": 96945,
    "encoder_right_finish": 96280,
    "x_finish": 150,
    "y_finish": 116,
    "theta_start": 68,
    "theta_finish": 69
   },
   {"motion_type:": "moving_in_arc",
    "filename": "robot_data_0_0_03_05_26_00_18_46.pkl",
    "encoder_left_start": 97433,
    "encoder_right_start": 96409,
    "encoder_left_finish": 103146,
    "encoder_right_finish": 101933,
    "x_finish": 145,
    "y_finish": 151,
    "theta_start": 61,
    "theta_finish": 72
   },
   {"motion_type:": "moving_in_arc",
    "filename": "robot_data_0_0_03_05_26_00_20_15.pkl",
    "encoder_left_start": 103157,
    "encoder_right_start": 101970,
    "encoder_left_finish": 112723,
    "encoder_right_finish": 111245,
    "x_finish": 298,
    "y_finish": 164,
    "theta_start": 73,
    "theta_finish": 82
   },
   {"motion_type:": "moving_in_arc",
    "filename": "robot_data_0_0_03_05_26_00_23_04.pkl",
    "encoder_left_start": 113203,
    "encoder_right_start": 111446,
    "encoder_left_finish": 119954,
    "encoder_right_finish": 117893,
    "x_finish": 127,
    "y_finish": 201,
    "theta_start": 41,
    "theta_finish": 62
   },
   {"motion_type:": "moving_in_arc",
    "filename": "robot_data_0_0_03_05_26_00_27_17.pkl",
    "encoder_left_start": 120065,
    "encoder_right_start": 118062,
    "encoder_left_finish": 130905,
    "encoder_right_finish": 128392,
    "x_finish": 370,
    "y_finish": 94,
    "theta_start": 80,
    "theta_finish": 113
   },
   # Tests with turns:official ones
   {"motion_type":"moving_in_arc",
    "filename": "robot_data_0_0_03_05_26_00_46_02.pkl",
    "encoder_left_start": 241357,
    "encoder_right_start": 237367,
    "encoder_left_finish": 247655,
    "encoder_right_finish": 244888,
    "x_finish": 54,
    "y_finish": 138,
    "theta_start": 116,
    "theta_finish": 10
   },
   {"motion_type":"moving_in_arc",
    "filename": "robot_data_0_0_03_05_26_00_50_46.pkl",
    "encoder_left_start": 247760,
    "encoder_right_start": 244937,
    "encoder_left_finish": 256471,
    "encoder_right_finish": 252057,
    "x_finish": 165,
    "y_finish": 130,
    "theta_start": 26,
    "theta_finish": 159
   },
   {"motion_type":"moving_in_arc",
    "filename": "robot_data_0_0_03_05_26_00_52_07.pkl", # starting angle unsure
    "encoder_left_start": 247760,
    "encoder_right_start": 244937,
    "encoder_left_finish": 266596,
    "encoder_right_finish": 261528,
    "x_finish": 325,
    "y_finish": 95,
    "theta_start": 65,
    "theta_finish": 127
   },
   {"motion_type":"moving_in_arc",
    "filename": "robot_data_0_0_03_05_26_00_54_33.pkl",
    "encoder_left_start": 266598,
    "encoder_right_start": 261554,
    "encoder_left_finish": 277133,
    "encoder_right_finish": 271365,
    "x_finish": 318,
    "y_finish": 102,
    "theta_start": 75,
    "theta_finish": 134 
   },
   {"motion_type":"moving_in_arc",
    "filename": "robot_data_0_0_03_05_26_01_04_09.pkl",
    "encoder_left_start": 277147,
    "encoder_right_start": 271376,
    "encoder_left_finish": 286109,
    "encoder_right_finish": 281384,
    "x_finish": 60,
    "y_finish": 254,
    "theta_start": 357,
    "theta_finish": 88 
   },
   {"motion_type":"moving_in_arc",
    "filename": "robot_data_0_0_03_05_26_01_06_58.pkl",
    "encoder_left_start": 286889,
    "encoder_right_start": 283268,
    "encoder_left_finish": 288261,
    "encoder_right_finish": 286278,
    "x_finish": -5,
    "y_finish": 66,
    "theta_start": 107,
    "theta_finish": 327
   },
   {"motion_type":"moving_in_arc",
    "filename": "robot_data_0_0_03_05_26_01_08_41.pkl",
    "encoder_left_start": 288318,
    "encoder_right_start": 286281,
    "encoder_left_finish": 291519,
    "encoder_right_finish": 291632,
    "x_finish": -9,
    "y_finish": 93,
    "theta_start": 110,
    "theta_finish": 286
   },
   {"motion_type":"moving_in_arc",
    "filename": "robot_data_0_0_03_05_26_01_10_17.pkl",
    "encoder_left_start": 291531,
    "encoder_right_start": 291635,
    "encoder_left_finish": 295132,
    "encoder_right_finish": 293626,
    "x_finish": 82,
    "y_finish": 5,
    "theta_start": 51,
    "theta_finish": 187
   },
   {"motion_type":"moving_in_arc",
    "filename": "robot_data_0_0_03_05_26_01_12_44.pkl",
    "encoder_left_start": 299149,
    "encoder_right_start": 294882,
    "encoder_left_finish": 301506,
    "encoder_right_finish": 295753,
    "x_finish": 48,
    "y_finish": -5,
    "theta_start": 58,
    "theta_finish": 185
   },
#    {"motion_type":"moving_in_arc",
#     "filename": "robot_data_0_0_03_05_26_01_13_51.pkl",
#     "encoder_left_start": 301631,
#     "encoder_right_start": 295821,
#     "encoder_left_finish": 305567,
#     "encoder_right_finish": 297609,
#     "x_finish": 64,
#     "y_finish": 7,
#     "theta_start": 30,
#     "theta_finish": 212
#    },

   # going clockwise
    {
        "motion_type":"rotation_in_place",
        "filename": "robot_data_0_0_03_05_26_13_20_10.pkl",
        "encoder_left_start": 3017,
        "encoder_right_start": -1753,
        "encoder_left_finish": 3762,
        "encoder_right_finish": -2084,
        "theta_start": 17,
        "theta_finish": 136,
        "turn_direction": "clockwise",
        "x_finish": 0,
        "y_finish": 0,
    },
    {   
        "motion_type":"rotation_in_place",
        "filename": "robot_data_0_0_03_05_26_13_21_47.pkl",
        "encoder_left_start": 3766,
        "encoder_right_start": -2071,
        "encoder_left_finish": 5011,
        "encoder_right_finish": -2831,
        "theta_start": 104,
        "theta_finish": 301,
        "turn_direction" : "clockwise",
        "x_finish": 0,
        "y_finish": 0,
    },
    {
        "motion_type":"rotation_in_place",
        "filename": "robot_data_0_0_03_05_26_13_23_37.pkl",
        "encoder_left_start": 5011,
        "encoder_right_start": -2831,
        "encoder_left_finish": 7042,
        "encoder_right_finish": -4143,
        "theta_start": 300,
        "theta_finish": 252,
        "turn_direction": "clockwise",
        "x_finish": 0,
        "y_finish": 0,
    },
    {
        "motion_type":"rotation_in_place",
        "filename": "robot_data_0_0_03_05_26_13_34_11.pkl",
        "encoder_left_start": 4451,
        "encoder_right_start": 6728,
        "encoder_left_finish": 6532,
        "encoder_right_finish": 5374,
        "theta_start": 269,
        "theta_finish": 228,
        "turn_direction" : "clockwise",
        "x_finish": 0,
        "y_finish": 0,
    },
    {
        "motion_type":"rotation_in_place",
        "filename": "robot_data_0_0_03_05_26_13_35_47.pkl",
        "encoder_left_start": 6532,
        "encoder_right_start": 5374,
        "encoder_left_finish": 9526,
        "encoder_right_finish": 3366,
        "theta_start": 227,
        "theta_finish": 317,
        "turn_direction" : "clockwise",
        "x_finish": 0,
        "y_finish": 0,
    },
    # {
    #     "filename": "robot_data_0_0_03_05_26_13_38_35.pkl", # spinning clockwise 3 times
    #     "encoder_left_start": 7315,
    #     "encoder_right_start": 6623,
    #     "encoder_left_finish": 14585,
    #     "encoder_right_finish": 1486,
    #     "theta_start": 174,
    #     "theta_finish": 174,
    # },

    # going counter clockwise
    {
        "motion_type":"rotation_in_place",
        "filename": "robot_data_0_0_03_05_26_13_26_38.pkl",
        "encoder_left_start": 7042,
        "encoder_right_start": -4143,
        "encoder_left_finish": 6887,
        "encoder_right_finish": -3624,
        "theta_start": 251,
        "theta_finish": 161,
        "turn_direction": "anticlockwise",
        "x_finish": 0,
        "y_finish": 0,
    },
    {
        "motion_type":"rotation_in_place",
        "filename": "robot_data_0_0_03_05_26_13_27_55.pkl",
        "encoder_left_start": 6887,
        "encoder_right_start": -3624,
        "encoder_left_finish": 6664,
        "encoder_right_finish": -3046,
        "theta_start": 161,
        "theta_finish": 55,
        "turn_direction": "anticlockwise",
        "x_finish": 0,
        "y_finish": 0,
    },
    {
        "motion_type":"rotation_in_place",
        "filename": "robot_data_0_0_03_05_26_13_28_52.pkl",
        "encoder_left_start": 6664,
        "encoder_right_start": -3046,
        "encoder_left_finish": 6017,
        "encoder_right_finish": -1978,
        "theta_start": 55,
        "theta_finish": 226,
        "turn_direction": "anticlockwise",
        "x_finish": 0,
        "y_finish": 0,

    },
    {
        "motion_type":"rotation_in_place",
        "filename": "robot_data_0_0_03_05_26_13_30_05.pkl",
        "encoder_left_start": 6026,
        "encoder_right_start": -1836,
        "encoder_left_finish": 4803,
        "encoder_right_finish": -83,
        "theta_start": 200,
        "theta_finish": 267,
        "turn_direction": "anticlockwise",
        "x_finish": 0,
        "y_finish": 0,
    },
    {
        "motion_type":"rotation_in_place",
        "filename": "robot_data_0_0_03_05_26_13_32_47.pkl",
        "encoder_left_start": 5186,
        "encoder_right_start": 5419,
        "encoder_left_finish": 4451,
        "encoder_right_finish": 6728,
        "theta_start": 124,
        "theta_finish": 269,
        "turn_direction": "anticlockwise",
        "x_finish": 0,
        "y_finish": 0,
    },
    {
        "motion_type":"rotation_in_place",
        "filename": "robot_data_0_0_03_05_26_13_36_46.pkl",
        "encoder_left_start": 9526,
        "encoder_right_start": 3366,
        "encoder_left_finish": 7315,
        "encoder_right_finish": 6623,
        "theta_start": 317,
        "theta_finish": 175,
        "turn_direction": "anticlockwise",
        "x_finish": 0,
        "y_finish": 0,
    },
]



# -----------------------------
# PARAMETERS
# -----------------------------
b = 0.215  # wheelbase in meters (CHANGE THIS)
cm_to_m = 0.01

# -----------------------------
# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def shortest_delta_theta(theta_start_deg, theta_end_deg):
    """
    For normal arc turns.
    Returns shortest signed angle change in radians.
    Good when robot turns less than 180 degrees.
    """
    delta_deg = theta_end_deg - theta_start_deg
    delta_rad = np.deg2rad(delta_deg)

    # wrap to [-pi, pi]
    return np.arctan2(np.sin(delta_rad), np.cos(delta_rad))


def compass_delta_theta(theta_start_deg, theta_end_deg, turn_direction):
    """
    For rotation-in-place trials.
    Uses known direction because robot may rotate more than 180 degrees.

    turn_direction:
        "clockwise"
        "anticlockwise"
    """
    start = theta_start_deg % 360
    end = theta_end_deg % 360

    if turn_direction == "clockwise":
        delta_deg = (end - start) % 360

    elif turn_direction == "anticlockwise":
        delta_deg = -((start - end) % 360)

    else:
        raise ValueError("rotation_in_place needs turn_direction: clockwise or anticlockwise")

    return np.deg2rad(delta_deg)

# R² FOR ENCODER → DISTANCE
# -----------------------------
def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)


# -----------------------------
# STORAGE
# -----------------------------
eL_list = []
eR_list = []
sL_true_list = []
sR_true_list = []

motion_type_list = []
filename_list = []

# -----------------------------
# LOOP THROUGH DATA
# -----------------------------
for d in files_and_data:

    motion_type = d["motion_type"] if "motion_type" in d else d["motion_type:"]

    # Encoder change
    d_eL = d["encoder_left_finish"] - d["encoder_left_start"]
    d_eR = d["encoder_right_finish"] - d["encoder_right_start"]

    # Position, cm -> meters
    x = d["x_finish"] * cm_to_m
    y = d["y_finish"] * cm_to_m

    # -----------------------------
    # CASE 1: STRAIGHT
    # -----------------------------
    if motion_type == "straight":

        s_true = np.sqrt(x**2 + y**2)

        sL_true = s_true
        sR_true = s_true
        ##print"motion_type, true distance(strue), x, y
        print(f"Straight trial: {d['filename']}, true distance: {s_true:.4f} m (x: {x:.4f} m, y: {y:.4f} m)")

    # -----------------------------
    # CASE 2: ROTATION IN PLACE
    # -----------------------------
    elif motion_type == "rotation_in_place":

        d_theta = compass_delta_theta(
            d["theta_start"],
            d["theta_finish"],
            d["turn_direction"]
        )

        if abs(d_theta) < 1e-3:
            continue

        s_mag = (b / 2) * abs(d_theta)

        # Magnitude from compass angle, direction from encoders
        sL_true = np.sign(d_eL) * s_mag
        sR_true = np.sign(d_eR) * s_mag
        ##print"motion_type, true distance(strue_left,strue_right, x_mesaured, y_measured,thetastart_measured,thetafinish_measured,d_theta(indegrees)
        #prin the motion type, true distance, measureded theta start, measured theta finish and calculated change in dgrees, and x_true and y_true
        print(f"Rotation in place trial: Direction: {d['turn_direction']}, True distance: left:{sL_true:.4f} m, right:{sR_true:.4f} m, theta start: {d['theta_start']} deg, theta finish: {d['theta_finish']} deg, delta theta: {np.rad2deg(d_theta):.2f} deg")

    # -----------------------------
    # CASE 3: MOVING IN ARC
    # -----------------------------
    elif motion_type == "moving_in_arc":

        d_theta = shortest_delta_theta(
            d["theta_start"],
            d["theta_finish"]
        )

        c = np.sqrt(x**2 + y**2)

        if abs(d_theta) < 1e-3:
            continue

        R = c / (2 * np.sin(abs(d_theta) / 2))

        s_inner = (R - b / 2) * abs(d_theta)
        s_outer = (R + b / 2) * abs(d_theta)

        # The wheel with larger encoder change traveled the outer arc
        if abs(d_eL) > abs(d_eR):
            sL_true = np.sign(d_eL) * s_outer
            sR_true = np.sign(d_eR) * s_inner
        else:
            sL_true = np.sign(d_eL) * s_inner
            sR_true = np.sign(d_eR) * s_outer
        
        print(f"Moving in arc trial: True distance: Left: {sL_true:.4f} m, Right: {sR_true:.4f} m, theta start: {d['theta_start']} deg, theta finish: {d['theta_finish']} deg, delta theta: {np.rad2deg(d_theta):.2f} deg")

    else:
        continue

    # Store data for fitting
    eL_list.append(d_eL)
    eR_list.append(d_eR)
    sL_true_list.append(sL_true)
    sR_true_list.append(sR_true)

    motion_type_list.append(motion_type)
    filename_list.append(d["filename"])

    # # store
    # eL_list.append(d_eL)
    # eR_list.append(d_eR)
    # sL_true_list.append(sL_true)
    # sR_true_list.append(sR_true)

# -----------------------------
# CONVERT TO NUMPY
# -----------------------------
eL = np.array(eL_list)
eR = np.array(eR_list)
sL = np.array(sL_true_list)
sR = np.array(sR_true_list)



# -----------------------------
# FIT k (linear through origin): Encoder counter change to Encoder distance for each wheel
# -----------------------------
k_L = np.sum(eL * sL) / np.sum(eL * eL)
k_R = np.sum(eR * sR) / np.sum(eR * eR)

print("k_L:", k_L)
print("k_R:", k_R)

# -----------------------------
# PREDICTIONS
# -----------------------------
sL_pred = k_L * eL
sR_pred = k_R * eR

# -----------------------------
# ERRORS
# -----------------------------
err_L = sL - sL_pred
err_R = sR - sR_pred

print("Mean Variance Left:", np.mean(err_L**2))
print("Mean Variance Right:", np.mean(err_R**2))
r2_L = compute_r2(sL, sL_pred)
r2_R = compute_r2(sR, sR_pred)

print("\nR² (Encoder → Distance):")
print(f"Left wheel R²:  {r2_L:.4f}")
print(f"Right wheel R²: {r2_R:.4f}")
# -----------------------------
# PLOTTING
# -----------------------------
plt.figure(figsize=(10,5))

# Left wheel
plt.subplot(1,2,1)
plt.scatter(eL, sL, label="True")
plt.plot(eL, sL_pred, color='red', label="Fit")
plt.title("Left Wheel-Distance(True-dots and Predicted-Line) vs Encoder Change")
plt.xlabel("Encoder Δ")
plt.ylabel("Distance (m)")
plt.legend()

# Right wheel
plt.subplot(1,2,2)
plt.scatter(eR, sR, label="True")
plt.plot(eR, sR_pred, color='red', label="Fit")
plt.title("Right Wheel--Distance(True-dots and Predicted-Line) vs Encoder Change")
plt.xlabel("Encoder Δ")
plt.ylabel("Distance (m)")
plt.legend()

plt.tight_layout()
plt.show()

print(f"s_L = {k_L:.8f} * Δe_L")
print(f"s_R = {k_R:.8f} * Δe_R")

# -----------------------------
# FIT VARIANCE FUNCTIONS
# -----------------------------
# Squared errors are our observed variance samples
err2_L = err_L**2
err2_R = err_R**2

# Use predicted distance magnitude as x-axis
xL_var = np.abs(sL_pred)
xR_var = np.abs(sR_pred)

# Fit: sigma^2 = a*x^2 + b*x + c
coef_L = np.polyfit(xL_var, err2_L, 2)
coef_R = np.polyfit(xR_var, err2_R, 2)

a_L, b_L, c_L = coef_L
a_R, b_R, c_R = coef_R

print("\nQuadratic variance functions:")
print(f"sigma_L^2 = {a_L:.8f}*|s_L_pred|^2 + {b_L:.8f}*|s_L_pred| + {c_L:.8f}")
print(f"sigma_R^2 = {a_R:.8f}*|s_R_pred|^2 + {b_R:.8f}*|s_R_pred| + {c_R:.8f}")

# Smooth x values for plotting
xL_plot = np.linspace(min(xL_var), max(xL_var), 200)
xR_plot = np.linspace(min(xR_var), max(xR_var), 200)

# Predicted variance values
sigma2_L_plot = a_L*xL_plot**2 + b_L*xL_plot + c_L
sigma2_R_plot = a_R*xR_plot**2 + b_R*xR_plot + c_R

# Variance cannot be negative
sigma2_L_plot = np.maximum(sigma2_L_plot, 0.0)
sigma2_R_plot = np.maximum(sigma2_R_plot, 0.0)
# -----------------------------
# R² FOR VARIANCE MODEL
# -----------------------------
err2_L_pred = a_L*xL_var**2 + b_L*xL_var + c_L
err2_R_pred = a_R*xR_var**2 + b_R*xR_var + c_R

r2_var_L = compute_r2(err2_L, err2_L_pred)
r2_var_R = compute_r2(err2_R, err2_R_pred)

print("\nR² (Variance model):")
print(f"Left wheel variance R²:  {r2_var_L:.4f}")
print(f"Right wheel variance R²: {r2_var_R:.4f}")

# Get index of worst outlier
idx_L = np.argmax(err2_L)
idx_R = np.argmax(err2_R)

print("\nWorst LEFT wheel outlier:")
print("File:", filename_list[idx_L])
print("Error:", err_L[idx_L])
print("Squared error:", err2_L[idx_L])
print("True:", sL_true_list[idx_L])
print("Pred:", sL_pred[idx_L])

print("\nWorst RIGHT wheel outlier:")
print("File:", filename_list[idx_R])
print("Error:", err_R[idx_R])
print("Squared error:", err2_R[idx_R])
print("True:", sR_true_list[idx_R])
print("Pred:", sR_pred[idx_R])


# -----------------------------
# PLOT VARIANCE FIT
# -----------------------------
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(xL_var, err2_L, label="Observed squared error")
plt.plot(xL_plot, sigma2_L_plot, color="red", label="Quadratic variance fit")
plt.title("Left Wheel Variance Fit")
plt.xlabel("|Predicted distance| (m)")
plt.ylabel("Squared error (m²)")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(xR_var, err2_R, label="Observed squared error")
plt.plot(xR_plot, sigma2_R_plot, color="red", label="Quadratic variance fit")
plt.title("Right Wheel Variance Fit")
plt.xlabel("|Predicted distance| (m)")
plt.ylabel("Squared error (m²)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# # Fit: sigma^2 = a * |s_pred| + c
# a_L = np.polyfit(xL_var, err2_L, 2)
# a_R = np.polyfit(xR_var, err2_R, 2)

# # Avoid negative variance intercepts
# c_L = max(c_L, 0.0)
# c_R = max(c_R, 0.0)

# print("\nVariance functions:")
# print(f"sigma_L^2 = {a_L:.8f} * |s_L_pred| + {c_L:.8f}")
# print(f"sigma_R^2 = {a_R:.8f} * |s_R_pred| + {c_R:.8f}")

# # Predicted variance values
# sigma2_L_pred = a_L * xL_var + c_L
# sigma2_R_pred = a_R * xR_var + c_R

# # Avoid negative predicted variances
# sigma2_L_pred = np.maximum(sigma2_L_pred, 0.0)
# sigma2_R_pred = np.maximum(sigma2_R_pred, 0.0)


# # -----------------------------
# # PLOT VARIANCE FIT
# # -----------------------------
# plt.figure(figsize=(10, 5))

# # Left wheel
# plt.subplot(1, 2, 1)
# plt.scatter(xL_var, err2_L, label="Observed squared error")
# plt.plot(xL_var, sigma2_L_pred, color="red", label="Variance fit")
# plt.title("Left Wheel Variance Fit")
# plt.xlabel("|Predicted distance| (m)")
# plt.ylabel("Squared error (m²)")
# plt.legend()
# plt.grid(True)

# # Right wheel
# plt.subplot(1, 2, 2)
# plt.scatter(xR_var, err2_R, label="Observed squared error")
# plt.plot(xR_var, sigma2_R_pred, color="red", label="Variance fit")
# plt.title("Right Wheel Variance Fit")
# plt.xlabel("|Predicted distance| (m)")
# plt.ylabel("Squared error (m²)")
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()



# ##Plotting and seeing some predicted on a single file: 
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from data_handling import get_file_data


# # -----------------------------
# # FITTED ENCODER FUNCTIONS
# # -----------------------------
# k_L= 0.0003010320294050663
# k_R= 0.00031484843554596296

# cm_to_m = 0.01


# # -----------------------------
# # YOUR TRIAL SUMMARY DATA
# # Keep only the files you care about here


# def find_trial_info(filename):
#     """Find the manually measured trial info from files_and_data."""
#     base_filename = os.path.basename(filename)

#     for d in files_and_data:
#         if d["filename"] == base_filename:  # rremove data/ from filename if present
#             return d
#     return None


# def predict_wheel_distance_over_time(filename):
#     """
#     Open robot data file, read encoder counts over time,
#     and convert encoder change to wheel distance.
#     """
#     (
#         time_list,
#         encoder_left_count_list,
#         encoder_right_count_list,
#         velocity_left_list,
#         velocity_right_list,
#         steering_angle_list,
#         x_camera_list,
#         y_camera_list,
#         z_camera_list,
#         yaw_camera_list,
#     ) = get_file_data(filename)

#     if len(encoder_left_count_list) < 2 or len(encoder_right_count_list) < 2:
#         raise ValueError("Not enough encoder data in this file.")

#     # Initial encoder readings
#     eL0 = encoder_left_count_list[0]
#     eR0 = encoder_right_count_list[0]

#     # Encoder change from start at every timestamp
#     d_eL_list = np.array(encoder_left_count_list) - eL0
#     d_eR_list = np.array(encoder_right_count_list) - eR0

#     # Apply fitted functions
#     sL_pred_list = k_L * d_eL_list
#     sR_pred_list = k_R * d_eR_list

#     return (
#         np.array(time_list),
#         d_eL_list,
#         d_eR_list,
#         sL_pred_list,
#         sR_pred_list,
#     )

# def predict_trajectory_from_wheels(sL_pred_list, sR_pred_list, theta0_deg=0.0):
#     """
#     Convert predicted left/right wheel distances over time into x, y, theta trajectory.
#     """
#     x_list = [0.0]
#     y_list = [0.0]
#     theta_list = [np.deg2rad(theta0_deg)]

#     for i in range(1, len(sL_pred_list)):
#         # incremental wheel distances
#         dsL = sL_pred_list[i] - sL_pred_list[i - 1]
#         dsR = sR_pred_list[i] - sR_pred_list[i - 1]

#         ds = (dsR + dsL) / 2.0
#         dtheta = (dsR - dsL) / b

#         theta_prev = theta_list[-1]
#         theta_mid = theta_prev + dtheta / 2.0

#         x_new = x_list[-1] + ds * np.cos(theta_mid)
#         y_new = y_list[-1] + ds * np.sin(theta_mid)
#         theta_new = theta_prev + dtheta

#         x_list.append(x_new)
#         y_list.append(y_new)
#         theta_list.append(theta_new)

#     return np.array(x_list), np.array(y_list), np.array(theta_list)



# def run_one_file(filename):
#     """Run prediction and plotting for one selected file."""

#     (
#         time_list,
#         d_eL_list,
#         d_eR_list,
#         sL_pred_list,
#         sR_pred_list,
#     ) = predict_wheel_distance_over_time(filename)

#     trial_info = find_trial_info(filename)

#     print("\nSelected file:", filename)
#     print("Left encoder change final:", d_eL_list[-1])
#     print("Right encoder change final:", d_eR_list[-1])
#     print(f"Predicted left wheel distance final:  {sL_pred_list[-1]:.4f} m")
#     print(f"Predicted right wheel distance final: {sR_pred_list[-1]:.4f} m")

#     if trial_info is not None:
#         x_true = trial_info["x_finish"] * cm_to_m
#         y_true = trial_info["y_finish"] * cm_to_m
#         s_center_true = np.sqrt(x_true**2 + y_true**2)

#         print(f"Measured final x: {x_true:.4f} m")
#         print(f"Measured final y: {y_true:.4f} m")
#         print(f"Measured center displacement: {s_center_true:.4f} m")
#         theta0_deg = 0.0

#     theta0_deg = trial_info["theta_start"]

#     x_pred, y_pred, theta_pred = predict_trajectory_from_wheels(
#         sL_pred_list,
#         sR_pred_list,
#         theta0_deg=theta0_deg
#     )



#     # -----------------------------
#     # Plot 1: predicted wheel distance over time
#     # -----------------------------
#     plt.figure(figsize=(8, 5))
#     plt.plot(time_list, sL_pred_list, label="Predicted left wheel distance")
#     plt.plot(time_list, sR_pred_list, label="Predicted right wheel distance")
#     plt.xlabel("Time")
#     plt.ylabel("Distance from start (m)")
#     plt.title("Predicted Wheel Distance Over Time")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()


#     ## PLOT: TRAJECTORY
#     plt.figure(figsize=(7, 7))
#     plt.plot(x_pred, y_pred, label="Predicted trajectory")
#     plt.scatter(x_pred[0], y_pred[0], label="Start")
#     plt.scatter(x_pred[-1], y_pred[-1], label="Predicted final")

#     if trial_info is not None:
#         x_true = trial_info["x_finish"] * cm_to_m
#         y_true = trial_info["y_finish"] * cm_to_m
#         plt.scatter(x_true, y_true, marker="x", s=100, label="Measured final")

#     plt.xlabel("x (m)")
#     plt.ylabel("y (m)")
#     plt.title("Predicted Robot Trajectory from Encoders")
#     plt.axis("equal")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


# # -----------------------------
# # CHOOSE FILE HERE
# # -----------------------------
# filename_to_run = "data/robot_data_0_0_02_05_26_23_59_25.pkl"
# run_one_file(filename_to_run)