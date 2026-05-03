/*
 * LiDAR Variance Calibration Sketch
 * ROB-GY 6213 Final Project
 *
 * Purpose:
 *   The LiDAR spins continuously. This sketch collects ONLY the
 *   measurements whose angle falls within a narrow window around 0°
 *   (i.e., the "front" ray) and streams them to Serial as CSV lines.
 *
 * Setup:
 *   - Connect the Arduino Giga R1 via USB to your laptop.
 *   - Place the LiDAR facing a flat wall at a KNOWN measured distance.
 *   - Open a terminal / Serial Monitor at 115200 baud OR run the
 *     companion Python script (lidar_calibration_logger.py).
 *
 * Output format (one line per accepted measurement):
 *   <timestamp_ms>,<angle_deg>,<distance_mm>
 *
 * How to determine "0 degrees":
 *   The RPLIDAR A-series defines 0° as the direction the sensor
 *   faces when the cable exits toward the back. In your mount,
 *   you may need to rotate this window (ANGLE_TARGET) after a
 *   quick visual check — see the companion script for a polar plot
 *   that helps you identify which angle corresponds to the front wall.
 */

#include "RPLidar.h"

// ─── Tuneable parameters ───────────────────────────────────────────────
// Centre of the angle window to capture (degrees, 0–360)
// After a first run, adjust this to match whichever angle cluster
// shows the shortest / most stable distance toward your front wall.
#define ANGLE_TARGET      0.0f

// Half-width of the acceptance window (±degrees around ANGLE_TARGET).
// At 10 Hz spin rate the RPLIDAR A1 emits ~8000 samples/rev, so
// a ±2° window gives you ~44 samples per revolution — plenty.
#define ANGLE_HALF_WIDTH  2.0f

// Minimum plausible distance (mm). Filters out 0 mm "no-return" values.
#define DIST_MIN_MM       50.0f

// Maximum plausible distance (mm). Keep this generous.
#define DIST_MAX_MM       6000.0f

// Minimum quality threshold (0–15 for RPLIDAR A1).
// Readings below this are too noisy to trust.
#define QUALITY_MIN       10

// How many samples to collect before auto-stopping.
// 200 gives a solid statistical baseline per distance trial.
#define SAMPLES_TO_COLLECT 200
// ───────────────────────────────────────────────────────────────────────

#define LidarMotorPin 1   // matches your main project

RPLidar lidar;
int samplesCollected = 0;
bool collecting = false;

void setup() {
  Serial.begin(115200);
  while (!Serial);   // wait for USB serial on Giga R1

  Serial.println("# LiDAR Variance Calibration");
  Serial.println("# Ensure LiDAR faces a flat wall at a measured distance.");
  Serial.println("# Send any character over Serial to START collection.");
  Serial.println("# Output: timestamp_ms, angle_deg, distance_mm");

  // Start LiDAR
  Serial2.begin(460800);
  lidar.begin(Serial2);
  delay(1000);

  if (!lidar.begin(Serial2)) {
    Serial.println("# ERROR: LiDAR not detected. Check wiring.");
    while (true);
  }

  pinMode(LidarMotorPin, OUTPUT);
  analogWrite(LidarMotorPin, 255);  // spin at max speed
  delay(2000);                       // let the motor reach stable RPM

  // Attempt to start scan
  rplidar_response_device_info_t info;
  if (IS_OK(lidar.getDeviceInfo(info, 100))) {
    lidar.startScan();
    Serial.println("# LiDAR scan started. Waiting for start command...");
  } else {
    Serial.println("# ERROR: Could not get device info.");
    while (true);
  }
}

void loop() {
  // Wait for a keypress to begin
  if (!collecting) {
    if (Serial.available()) {
      while (Serial.available()) Serial.read(); // flush
      collecting = true;
      samplesCollected = 0;
      Serial.println("# Collection STARTED");
      Serial.println("timestamp_ms,angle_deg,distance_mm,quality");
    }
    // Keep draining the LiDAR buffer while waiting, otherwise it backs up
    if (IS_OK(lidar.waitPoint())) { /* discard */ }
    return;
  }

  // --- Collection phase ---
  if (samplesCollected >= SAMPLES_TO_COLLECT) {
    // Done
    analogWrite(LidarMotorPin, 0);   // stop motor
    lidar.stop();
    Serial.print("# Collection COMPLETE. ");
    Serial.print(samplesCollected);
    Serial.println(" samples recorded.");
    Serial.println("# Copy the CSV above into lidar_calibration_logger.py for analysis.");
    collecting = false;
    while (true);  // halt — reset the board to run again
  }

  if (!IS_OK(lidar.waitPoint())) return;

  float angle    = lidar.getCurrentPoint().angle;
  float distance = lidar.getCurrentPoint().distance;
  uint8_t quality = lidar.getCurrentPoint().quality;

  // ── Angle window filter ──────────────────────────────────────────────
  // The angle is 0–360°. We need wrap-around safe comparison.
  float diff = angle - ANGLE_TARGET;
  // Normalise diff to [-180, 180]
  while (diff >  180.0f) diff -= 360.0f;
  while (diff < -180.0f) diff += 360.0f;
  bool inWindow = (diff >= -ANGLE_HALF_WIDTH && diff <= ANGLE_HALF_WIDTH);

  if (!inWindow)                        return;
  if (distance < DIST_MIN_MM)           return;
  if (distance > DIST_MAX_MM)           return;
  if (quality  < QUALITY_MIN)           return;

  // ── Emit CSV line ────────────────────────────────────────────────────
  Serial.print(millis());
  Serial.print(",");
  Serial.print(angle, 2);
  Serial.print(",");
  Serial.print(distance, 1);
  Serial.print(",");
  Serial.println(quality);

  samplesCollected++;
}
