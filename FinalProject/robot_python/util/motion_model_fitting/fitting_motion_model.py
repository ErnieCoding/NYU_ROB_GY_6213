import numpy as np
import matplotlib.pyplot as plt
from fitting_data import*


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



def fitting_motion_model(files_and_data):
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


    print("\nData summary:")
    print(f"Total trials: {len(eL)}")
    print(f"Motion types: {set(motion_type_list)}")
    #number of trials for each motion type
    print("Trials per motion type:")
    for mt in set(motion_type_list):
        count = motion_type_list.count(mt)
        print(f"  {mt}: {count} trials")    


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

    # # -----------------------------
    # # FIT VARIANCE FUNCTIONS
    # # -----------------------------

    err2_L = err_L**2
    err2_R = err_R**2

    xL = np.abs(sL_pred)
    xR = np.abs(sR_pred)

    # -----------------------------
    # MODEL 1: CONSTANT
    # -----------------------------
    const_L = np.mean(err2_L)
    const_R = np.mean(err2_R)

    pred_const_L = np.full_like(err2_L, const_L)
    pred_const_R = np.full_like(err2_R, const_R)

    # -----------------------------
    # MODEL 2: LINEAR
    # -----------------------------
    aL_lin, bL_lin = np.polyfit(xL, err2_L, 1)
    aR_lin, bR_lin = np.polyfit(xR, err2_R, 1)

    pred_lin_L = aL_lin * xL + bL_lin
    pred_lin_R = aR_lin * xR + bR_lin
    ##print the linear model variance for each wheel:
    print("\nLinear variance functions:")
    print(f"sigma_L^2 = {aL_lin:.8f}*|s_L_pred| + {bL_lin:.8f}")
    print(f"sigma_R^2 = {aR_lin:.8f}*|s_R_pred| + {bR_lin:.8f}")

    # -----------------------------
    # MODEL 3: QUADRATIC
    # -----------------------------
    aL, bL, cL = np.polyfit(xL, err2_L, 2)
    aR, bR, cR = np.polyfit(xR, err2_R, 2)

    pred_quad_L = aL*xL**2 + bL*xL + cL
    pred_quad_R = aR*xR**2 + bR*xR + cR
    print("\nQuadratic variance functions:")
    print(f"sigma_L^2 = {aL:.8f}*|s_L_pred|^2 + {bL:.8f}*|s_L_pred| + {cL:.8f}")
    print(f"sigma_R^2 = {aR:.8f}*|s_R_pred|^2 + {bR:.8f}*|s_R_pred| + {cR:.8f}")


    # # -----------------------------
    # # R² FUNCTION
    # # -----------------------------
    # def compute_r2(y_true, y_pred):
    #     ss_res = np.sum((y_true - y_pred)**2)
    #     ss_tot = np.sum((y_true - np.mean(y_true))**2)
    #     return 1 - ss_res/ss_tot

    # -----------------------------
    # PRINT COMPARISON
    # -----------------------------
    print("\n=== VARIANCE MODEL COMPARISON ===")

    print("\nLEFT wheel:")
    print("Constant R²:", compute_r2(err2_L, pred_const_L))
    print("Linear   R²:", compute_r2(err2_L, pred_lin_L))
    print("Quadratic R²:", compute_r2(err2_L, pred_quad_L))

    print("\nRIGHT wheel:")
    print("Constant R²:", compute_r2(err2_R, pred_const_R))
    print("Linear   R²:", compute_r2(err2_R, pred_lin_R))
    print("Quadratic R²:", compute_r2(err2_R, pred_quad_R))

    # # Get index of worst outlier
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


    x_plot = np.linspace(min(xL), max(xL), 200)

    plt.figure(figsize=(10, 5))

    # LEFT
    plt.subplot(1, 2, 1)
    plt.scatter(xL, err2_L, label="Data")

    plt.plot(x_plot, np.full_like(x_plot, const_L), label="Constant")
    plt.plot(x_plot, aL_lin*x_plot + bL_lin, label="Linear")
    plt.plot(x_plot, aL*x_plot**2 + bL*x_plot + cL, label="Quadratic")

    plt.title("Left variance comparison")
    plt.xlabel("|distance|")
    plt.ylabel("error²")
    plt.legend()
    plt.grid(True)

    # RIGHT
    plt.subplot(1, 2, 2)
    plt.scatter(xR, err2_R, label="Data")

    plt.plot(x_plot, np.full_like(x_plot, const_R), label="Constant")
    plt.plot(x_plot, aR_lin*x_plot + bR_lin, label="Linear")
    plt.plot(x_plot, aR*x_plot**2 + bR*x_plot + cR, label="Quadratic")

    plt.title("Right variance comparison")
    plt.xlabel("|distance|")
    plt.ylabel("error²")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    fitting_motion_model(files_and_data)
    


# ##FINAL EQUATIONS:
# Left wheel R²:  0.9918
# Right wheel R²: 0.9907
# s_L = 0.00035210 * Δe_L
# s_R = 0.00035651 * Δe_R

# Quadratic variance functions:
# sigma_L^2 = 0.00047296*|s_L_pred|^2 + 0.00428335*|s_L_pred| + 0.00742903
# sigma_R^2 = 0.00679702*|s_R_pred|^2 + -0.01167738*|s_R_pred| + 0.00900622