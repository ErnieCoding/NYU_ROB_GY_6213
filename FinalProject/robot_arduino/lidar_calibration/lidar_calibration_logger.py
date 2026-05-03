"""
lidar_calibration_logger.py
ROB-GY 6213 Final Project — LiDAR Variance Characterisation Tool

Usage
-----
  python3 lidar_calibration_logger.py

The script will:
  1. Ask you for the COM port and the TRUE measured distance (mm).
  2. Send a start byte to the Arduino, then collect CSV lines in real time.
  3. When the Arduino signals completion, compute and display:
       - Mean measured distance
       - Bias  (mean − true)
       - Variance  σ²
       - Standard deviation  σ
  4. Save every raw measurement to a timestamped CSV file.
  5. Plot a histogram of the residuals (measurement − true) so you can
     visually confirm the noise looks Gaussian.

Running multiple trials
-----------------------
  Run once per (distance, surface) combination.
  Suggested distances: 0.3 m, 0.5 m, 1.0 m, 1.5 m, 2.0 m, 3.0 m.
  Note whether the surface is white wall, dark wood, glass, etc.
  Collect the printed σ² values and use the average as your R matrix
  entry in the EKF / particle filter.
"""

import serial
import serial.tools.list_ports
import csv
import os
import time
import statistics
import math
import datetime

# ── Optional: matplotlib for histogram ──────────────────────────────────
try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("[INFO] matplotlib not found — histogram will be skipped.")
    print("       Install with: pip3 install matplotlib")

BAUD_RATE = 115200
SAMPLES_TARGET = 200   # must match SAMPLES_TO_COLLECT in the .ino


def list_ports():
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("[ERROR] No serial ports found. Is the Arduino connected?")
        return []
    print("\nAvailable serial ports:")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p.device}  — {p.description}")
    return [p.device for p in ports]


def select_port(port_list):
    if len(port_list) == 1:
        print(f"[AUTO] Using the only available port: {port_list[0]}")
        return port_list[0]
    idx = int(input("Select port index: "))
    return port_list[idx]


def collect(port: str, true_distance_mm: float, label: str):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"lidar_cal_{label}_{int(true_distance_mm)}mm_{timestamp}.csv"

    measurements = []

    print(f"\n[INFO] Opening {port} at {BAUD_RATE} baud…")
    with serial.Serial(port, BAUD_RATE, timeout=2) as ser:
        time.sleep(2)   # give the Giga time to reset after DTR toggle

        # Flush startup messages
        print("[INFO] Flushing startup messages…")
        deadline = time.time() + 4
        while time.time() < deadline:
            line = ser.readline().decode("utf-8", errors="replace").strip()
            if line:
                print(f"  ARDUINO: {line}")
            if "Waiting for start command" in line:
                break

        # Send start signal
        print("[INFO] Sending start signal to Arduino…")
        ser.write(b"s")

        print(f"[INFO] Collecting {SAMPLES_TARGET} samples… (do NOT move the robot)")

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_ms", "angle_deg", "distance_mm",
                             "quality", "true_distance_mm", "residual_mm"])

            while True:
                raw = ser.readline().decode("utf-8", errors="replace").strip()
                if not raw:
                    continue

                if raw.startswith("#"):
                    print(f"  ARDUINO: {raw}")
                    if "COMPLETE" in raw or "ERROR" in raw:
                        break
                    continue

                # Parse CSV data line: timestamp_ms,angle_deg,distance_mm,quality
                parts = raw.split(",")
                if len(parts) != 4:
                    continue
                try:
                    ts, angle, dist, qual = (float(p) for p in parts)
                except ValueError:
                    continue

                residual = dist - true_distance_mm
                measurements.append(dist)
                writer.writerow([int(ts), f"{angle:.2f}", f"{dist:.1f}",
                                 int(qual), true_distance_mm, f"{residual:.1f}"])

                # Live progress bar
                n = len(measurements)
                bar = ("█" * (n * 40 // SAMPLES_TARGET)).ljust(40)
                print(f"\r  [{bar}] {n}/{SAMPLES_TARGET}", end="", flush=True)

    print()  # newline after progress bar
    return measurements, filename


def analyse(measurements, true_distance_mm: float, filename: str):
    if len(measurements) < 2:
        print("[ERROR] Too few samples to analyse.")
        return

    n    = len(measurements)
    mean = statistics.mean(measurements)
    var  = statistics.variance(measurements)   # sample variance (N-1)
    std  = math.sqrt(var)
    bias = mean - true_distance_mm
    rmse = math.sqrt(sum((m - true_distance_mm)**2 for m in measurements) / n)

    print("\n" + "═" * 54)
    print("  LiDAR CALIBRATION RESULTS")
    print("═" * 54)
    print(f"  True distance          : {true_distance_mm:.1f} mm")
    print(f"  Samples collected      : {n}")
    print(f"  Mean measured distance : {mean:.2f} mm")
    print(f"  Bias  (mean − true)    : {bias:+.2f} mm")
    print(f"  Variance  σ²           : {var:.4f} mm²")
    print(f"  Std-dev  σ             : {std:.4f} mm")
    print(f"  RMSE                   : {rmse:.4f} mm")
    print("═" * 54)
    print(f"\n  ➤  Use  R = {var:.4f}  (mm²)  as your EKF measurement noise")
    print(f"     or convert: R = {var / 1e6:.8f}  (m²)  if working in metres.")
    print(f"\n  Raw data saved to: {filename}\n")

    if not HAS_PLOT:
        return

    residuals = [m - true_distance_mm for m in measurements]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"LiDAR Calibration — True distance: {true_distance_mm} mm", fontsize=13)

    # ── Histogram of residuals ──────────────────────────────────────────
    ax = axes[0]
    ax.hist(residuals, bins=30, color="#4f98a3", edgecolor="white", linewidth=0.6)
    ax.axvline(0,    color="#a12c7b", linewidth=1.5, linestyle="--", label="Zero error")
    ax.axvline(bias, color="#da7101", linewidth=1.5, linestyle="-",  label=f"Bias = {bias:+.2f} mm")
    ax.set_xlabel("Residual  (measured − true)  [mm]")
    ax.set_ylabel("Count")
    ax.set_title("Residual Histogram")
    ax.legend(fontsize=9)

    # ── Time series of raw distance ────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(measurements, color="#4f98a3", linewidth=0.8, alpha=0.8)
    ax2.axhline(true_distance_mm, color="#a12c7b", linewidth=1.5,
                linestyle="--", label=f"True = {true_distance_mm} mm")
    ax2.axhline(mean, color="#da7101", linewidth=1.2,
                linestyle="-", label=f"Mean = {mean:.1f} mm")
    ax2.set_xlabel("Sample index")
    ax2.set_ylabel("Distance [mm]")
    ax2.set_title("Raw Distance Over Time")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plot_filename = filename.replace(".csv", ".png")
    plt.savefig(plot_filename, dpi=150, bbox_inches="tight")
    print(f"  Plot saved to: {plot_filename}")
    plt.show()


def main():
    print("╔══════════════════════════════════════════════╗")
    print("║  LiDAR Variance Calibration Tool             ║")
    print("║  ROB-GY 6213 — Robot Navigation & Localiz.  ║")
    print("╚══════════════════════════════════════════════╝")

    port_list = list_ports()
    if not port_list:
        return
    port = select_port(port_list)

    true_dist = float(input("\nEnter TRUE distance to wall in mm (e.g. 500): "))
    label = input("Short label for this trial (e.g. white_wall, dark_wood): ").strip()
    label = label.replace(" ", "_") or "trial"

    measurements, filename = collect(port, true_dist, label)
    analyse(measurements, true_dist, filename)


if __name__ == "__main__":
    main()
