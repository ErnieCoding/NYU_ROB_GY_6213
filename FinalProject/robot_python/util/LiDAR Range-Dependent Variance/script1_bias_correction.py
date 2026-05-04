
"""
Script 1: Per-Distance Bias Analysis & Correction
ROB-GY 6213 — LiDAR Calibration

What this script does:
  1. Loads combined_data.csv
  2. Computes the mean bias at each calibration distance
  3. Plots bias vs distance
  4. Gives you a correction function to use at runtime
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# STEP 1: Load the calibration data
# ─────────────────────────────────────────────────────────────
df = pd.read_csv("combined_data.csv")

# ─────────────────────────────────────────────────────────────
# STEP 2: Compute per-distance statistics
#
#   For each true distance d_true:
#     - Collect all N LiDAR measurements z_1, z_2, ..., z_N
#     - Compute mean:  z_bar = (1/N) * sum(z_i)
#     - Compute bias:  bias  = z_bar - d_true
#
#   The bias tells us: "on average, how far off is the LiDAR?"
#   A positive bias means the sensor reads HIGHER than reality.
# ─────────────────────────────────────────────────────────────
stats = []
for d_true, group in df.groupby("true_distance_mm"):
    z = group["distance_mm"].values
    mean_z = np.mean(z)
    bias   = mean_z - d_true          # positive = reads too high
    n      = len(z)
    stats.append({"true_mm": d_true, "mean_mm": mean_z, "bias_mm": bias, "n": n})

stats_df = pd.DataFrame(stats).sort_values("true_mm").reset_index(drop=True)

print("Per-Distance Bias Table:")
print("=" * 45)
print(f"{'True (mm)':>10} {'Mean meas.':>12} {'Bias (mm)':>10} {'N':>5}")
print("-" * 45)
for _, row in stats_df.iterrows():
    print(f"{row['true_mm']:>10.0f} {row['mean_mm']:>12.2f} {row['bias_mm']:>10.4f} {row['n']:>5.0f}")

# ─────────────────────────────────────────────────────────────
# STEP 3: Store calibration arrays for runtime correction
#
#   At runtime, when we receive a measurement z from the LiDAR,
#   we use np.interp() to look up the bias at that distance
#   and subtract it. This is called interpolation — we draw a
#   straight line between the two nearest calibration points.
# ─────────────────────────────────────────────────────────────
CALIB_DIST = stats_df["true_mm"].values    # [300, 400, 500, ..., 1200]
CALIB_BIAS = stats_df["bias_mm"].values    # [5.92, 13.30, 9.76, ...]


def correct_measurement(z_mm):
    """
    Given a raw LiDAR measurement z_mm,
    return the bias-corrected measurement.

    Uses linear interpolation between calibration points.
    Clamps to the nearest known value outside the calibration range.
    """
    bias = float(np.interp(z_mm, CALIB_DIST, CALIB_BIAS))
    return z_mm - bias


# ─────────────────────────────────────────────────────────────
# STEP 4: Test the correction function
# ─────────────────────────────────────────────────────────────
print("\nTest: Correcting a few raw readings:")
print("-" * 45)
test_readings = [305.0, 413.0, 509.0, 606.0, 1006.0]
for z_raw in test_readings:
    z_corrected = correct_measurement(z_raw)
    bias_applied = z_raw - z_corrected
    print(f"  Raw: {z_raw:7.1f} mm  →  Corrected: {z_corrected:7.2f} mm  (bias removed: {bias_applied:.2f} mm)")

# ─────────────────────────────────────────────────────────────
# STEP 5: Plot bias vs distance
# ─────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.bar(stats_df["true_mm"].astype(int), stats_df["bias_mm"], width=60, color="steelblue", edgecolor="black")
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.xlabel("True Distance (mm)")
plt.ylabel("Bias = mean(z) - d_true  (mm)")
plt.title("LiDAR Systematic Bias per Distance\n(positive = sensor reads too high)")
plt.xticks(stats_df["true_mm"].astype(int))
plt.tight_layout()
plt.savefig("bias_correction.png", dpi=150)
plt.close()
print("\nPlot saved: bias_correction.png")
