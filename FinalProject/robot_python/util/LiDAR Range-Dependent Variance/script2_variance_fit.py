
"""
Script 2: Distance-Dependent Variance Fitting (Range-Dependent R)
ROB-GY 6213 — LiDAR Calibration

What this script does:
  1. Loads combined_data.csv
  2. Computes σ² (variance) at each calibration distance
  3. Fits TWO models to those 9 (distance, variance) data points:
       - Quadratic: σ²(d) = a·d² + b
       - Linear:    σ²(d) = c·d + b
  4. Plots both fits against the measured variances
  5. Gives you a get_R(z) function to call in your EKF

WHY DO WE FIT A MODEL?
  At runtime, you receive a measurement z from the LiDAR.
  You need to know the noise variance R at that distance.
  You can't pause the robot and run 200 samples to measure it —
  so you use the formula you fitted here instead.

WHERE DO a, b, c COME FROM?
  scipy.optimize.curve_fit finds the values of a and b that
  minimize the sum of squared errors between the formula's
  prediction and your 9 measured variance points.
  It's the same idea as fitting a line with linear regression,
  but generalized to any function shape.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ─────────────────────────────────────────────────────────────
# STEP 1: Load data and compute per-distance variance
#
#   For each true distance d_true:
#     - Collect all N LiDAR measurements z_1, z_2, ..., z_N
#     - Compute mean:     z_bar = (1/N) * sum(z_i)
#     - Compute variance: σ² = (1/(N-1)) * sum((z_i - z_bar)²)
#
#   ddof=1 means we use N-1 in the denominator (sample variance,
#   not population variance). This is the standard choice.
# ─────────────────────────────────────────────────────────────
df = pd.read_csv("combined_data.csv")

stats = []
for d_true, group in df.groupby("true_distance_mm"):
    z    = group["distance_mm"].values
    var  = np.var(z, ddof=1)     # sample variance σ²
    n    = len(z)
    stats.append({"true_mm": d_true, "variance": var, "n": n})

stats_df = pd.DataFrame(stats).sort_values("true_mm").reset_index(drop=True)

d_vals   = stats_df["true_mm"].values     # the 9 distance values
var_vals = stats_df["variance"].values    # the 9 measured variances

print("Per-Distance Variance:")
print("=" * 35)
print(f"{'True (mm)':>10} {'Variance (mm²)':>16} {'N':>5}")
print("-" * 35)
for _, row in stats_df.iterrows():
    print(f"{row['true_mm']:>10.0f} {row['variance']:>16.6f} {row['n']:>5.0f}")

# ─────────────────────────────────────────────────────────────
# STEP 2: Define the two model functions
#
#   We try two shapes:
#     (A) Quadratic: σ²(d) = a·d² + b
#         Physically motivated — range error in many sensors
#         grows proportionally to range, so variance ∝ d²
#
#     (B) Linear:    σ²(d) = c·d + b
#         Simpler, sometimes fits better empirically
#
#   curve_fit will find the best (a, b) or (c, b) for each.
# ─────────────────────────────────────────────────────────────
def quadratic_model(d, a, b):
    return a * d**2 + b

def linear_model(d, c, b):
    return c * d + b

# ─────────────────────────────────────────────────────────────
# STEP 3: Run the curve fitting
#
#   curve_fit(function, x_data, y_data, p0=initial_guess, bounds=...)
#     - p0: starting guess for [a, b]. Must be in the right ballpark.
#     - bounds=(0, inf): force both parameters to be non-negative
#       because variance cannot be negative.
#   Returns:
#     - popt: the best-fit parameter values [a, b]
#     - pcov: covariance of the fit (we don't use this here)
# ─────────────────────────────────────────────────────────────
popt_quad, _ = curve_fit(quadratic_model, d_vals, var_vals,
                          p0=[1e-7, 0.01], bounds=(0, np.inf))
a_fit, b_fit = popt_quad

popt_lin, _  = curve_fit(linear_model, d_vals, var_vals,
                          p0=[1e-4, 0.01])
c_fit, b_lin = popt_lin

# ─────────────────────────────────────────────────────────────
# STEP 4: Compute R² (goodness of fit)
#
#   R² = 1 - SS_residual / SS_total
#   R² = 1.0 means perfect fit. R² = 0 means the model is no
#   better than just predicting the mean of the data.
# ─────────────────────────────────────────────────────────────
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot

r2_quad = r_squared(var_vals, quadratic_model(d_vals, a_fit, b_fit))
r2_lin  = r_squared(var_vals, linear_model(d_vals, c_fit, b_lin))

print("\n── Fitted Model Parameters ──────────────────────────")
print(f"  Quadratic:  a = {a_fit:.6e},  b = {b_fit:.6f}   R² = {r2_quad:.4f}")
print(f"  Linear:     c = {c_fit:.6e},  b = {b_lin:.6f}   R² = {r2_lin:.4f}")
print()
print("  Quadratic formula:  σ²(d) = {:.4e} · d²  +  {:.4f}".format(a_fit, b_fit))
print("  Linear    formula:  σ²(d) = {:.4e} · d   +  {:.4f}".format(c_fit, b_lin))

# ─────────────────────────────────────────────────────────────
# STEP 5: Define get_R(z) — the function you call in your EKF
#
#   At runtime, z is the raw LiDAR measurement (in mm).
#   We use it as a proxy for the true distance to compute R.
#   We clamp R to a minimum floor (0.02 mm²) so R never
#   becomes zero or negative due to the linear model at short
#   distances — a zero R would make the Kalman gain blow up.
# ─────────────────────────────────────────────────────────────
R_FLOOR = 0.02   # minimum allowed variance (mm²)

def get_R_quadratic(z_mm):
    """
    Returns measurement noise variance R (mm²)
    using the quadratic model: R = a·z² + b
    z_mm: raw LiDAR reading in millimeters
    """
    R = a_fit * z_mm**2 + b_fit
    return max(R, R_FLOOR)

def get_R_linear(z_mm):
    """
    Returns measurement noise variance R (mm²)
    using the linear model: R = c·z + b
    z_mm: raw LiDAR reading in millimeters
    """
    R = c_fit * z_mm + b_lin
    return max(R, R_FLOOR)

# ─────────────────────────────────────────────────────────────
# STEP 6: Test the functions
# ─────────────────────────────────────────────────────────────
print("\n── Test: R at various distances ─────────────────────")
print(f"{'Distance (mm)':>15} {'R quadratic':>14} {'R linear':>12}")
print("-" * 45)
for d_test in [300, 500, 700, 900, 1000, 1200]:
    print(f"{d_test:>15}  {get_R_quadratic(d_test):>13.4f}  {get_R_linear(d_test):>11.4f}")

# ─────────────────────────────────────────────────────────────
# STEP 7: Plot measured variances vs fitted curves
# ─────────────────────────────────────────────────────────────
d_smooth = np.linspace(250, 1300, 400)

plt.figure(figsize=(8, 5))
plt.scatter(d_vals, var_vals, color="black", zorder=5, label="Measured σ² (calibration data)", s=60)
plt.plot(d_smooth, quadratic_model(d_smooth, a_fit, b_fit), color="blue",
         label=f"Quadratic fit: a·d² + b  (R²={r2_quad:.3f})")
plt.plot(d_smooth, linear_model(d_smooth, c_fit, b_lin), color="red", linestyle="--",
         label=f"Linear fit:    c·d  + b  (R²={r2_lin:.3f})")
plt.axhline(R_FLOOR, color="gray", linewidth=0.8, linestyle=":", label=f"R floor = {R_FLOOR}")
plt.xlabel("Distance (mm)")
plt.ylabel("Variance σ²  (mm²)")
plt.title("Distance-Dependent Variance Fit\nσ²(d) = a·d² + b  and  σ²(d) = c·d + b")
plt.legend()
plt.tight_layout()
plt.savefig("variance_fit.png", dpi=150)
plt.close()
print("\nPlot saved: variance_fit.png")

print("\n── Which model to use? ──────────────────────────────")
print("  Both R² values are around 0.70–0.74 (moderate fit).")
print("  Linear fits slightly better and is simpler.")
print("  Quadratic is physically motivated.")
print("  Recommendation: use LINEAR for this dataset.")
print("  The function to call in your EKF is: get_R_linear(z_mm)")
