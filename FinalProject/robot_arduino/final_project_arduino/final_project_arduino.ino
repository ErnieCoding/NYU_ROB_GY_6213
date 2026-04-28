#include <SPI.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include "RPLidat.h"

#define LidarMotorPin 1
#define NumLidarRaysPerMsg 50
RPLidar lidar;
String current_lidar_scan_data;
int current_num_lidar_rays;

#define LeftSpeedPin 9
#define LeftMotorDirPin1 12
#define LeftMotorDirPin2 11

#define RightSpeedPin 6
#define RightMotorDirPin1 10
#define RightMotorDirPin2 13

// Encoder outputs setup
#define RightEncoderOutputA 2
#define RightEncoderOutputB 3
#define LeftEncoderOutputA 5
#define LeftEncoderOutputB 4
int encoder_left_state;
int encoder_left_last_state;
int encoder_left_count;

int encoder_right_state;
int encoder_right_last_state;
int encoder_right_count;

// Network setup
#define SendDeltaTimeInMs 100
#define ReceiveDeltaTimeInMs 10
#define NoSignalDeltaTImeInMs 2000
char ssid[] = "Tenda_9C9620";      // REPLACE with your team's router ssid
char pass[] = "90650529";          // REPLACE with your team's router password 
char remoteIP[] = "192.168.0.200"; // REPLACE with your laptop's IP address on your team's router
unsigned int localPort = 4010;
unsigned int remotePort = 4010;    
int status = WL_IDLE_STATUS;
int last_time_rx = 0;
int last_time_tx = 0;
WiFiUDP Udp;
char packetBuffer[256];

// Control signals from laptop
struct ControlSignals 
{
  int speed_left = 0;
  int speed_right = 0;
};
ControlSignals last_control_signal;

// Sensor signals to laptop
struct SensorSignals
{
  int encoder_left_count = 0;
  int encoder_right_count = 0;
  int num_lidar_rays = 0;
  String lidar_scan_data = "";
};
SensorSignals last_sensor_signal;

void setup() {
  Serial.begin(115200);
  Serial.println("=============Running robot controls=============");

  if (WiFi.status() == WL_NO_MODULE) {
    Serial.println("Communication with WiFi module failed!");
    while(true);
  }
  
  while (status != WL_CONNECTED) {
    Serial.print("Attempting to connect to SSID: ");
    Serial.println(ssid);
    status = WiFi.begin(ssid, pass);

    delay(10000);
  }

  Serial.println("Connected to WiFi");
  printWifiStatus();
  Serial.println("\nStarting connection to server...");
  Udp.begin(localPort);

  // LiDAR setup
  Serial2.begin(460800);
  lidar.begin(Serial2);
  delay(1000);

  if (lidar.begin(Serial2)) {
    Serial.println("Started LiDAR");
  } else {
    Serial.println("Failed LiDAR");
  }

  pinMode(LidarMotorPin, OUTPUT);
  reset_lidar_message();

  // Speed controls setup
  pinMode(RightSpeedPin, OUTPUT);
  pinMode(RightMotorDirPin1, OUTPUT);
  pinMode(RightMotorDirPin2, OUTPUT);

  pinMode(LeftSpeedPin, OUTPUT);
  pinMode(LeftMotorDirPin1, OUTPUT);
  pinMode(LeftMotorDirPin2, OUTPUT);

  // Encoder input setup
  pinMode(RightEncoderOutputA, INPUT);
  pinMode(RightEncoderOutputB, INPUT);
  pinMode(LeftEncoderOutputA, INPUT);
  pinMode(LefttEncoderOutputB, INPUT);
  
  last_time_rx = millis();
  last_time_tx = millis();
}

void loop() {
  // put your main code here, to run repeatedly:

}

// Reset LiDAR
void reset_lidar_message()
{
  current_num_lidar_rays = 0;
  current_lidar_scan_data = "";
}


// Stop all robot motors
void stop()
{
  digitalWrite(RightMotorDirPin1, LOW);
  digitalWrite(RightMotorDirPin2,LOW);
  digitalWrite(LeftMotorDirPin1,LOW);
  digitalWrite(LeftMotorDirPin2,LOW);
}



/*Receive control signals from laptop*/

ControlSignals receive_control_signals(ControlSignals last_control_signal) 
{
  ControlSignals control_signal = last_control_signal;

  int new_time_rx = millis();
  if (new_time_rx - last_time_rx > ReceiveDeltaTimeInMs) {
    int packetSize = Upd.parsePacket();
    if (packetSize) {
      int len = Upd.read(packetBuffer, 255);
      if (len > 0) {
        packetBuffer[len] = 0;
      }

      control_signal = unpack_control_signal(packetBuffer);
      // Serial.print("Received cmd: ");
      // Serial.print(control_signal.speed);
      // Serial.print(", ");
      // Serial.println(control_signal.steering_angle);
      last_time_rx = new_time_rx;
    }
  }

  if (new_time_rx - last_time_rx > NoSignalDeltaTimeInMs) {
    control_signal.speed_left = 0;
    control_signal.speed_right = 0;
  }

  return control_signal;
}

ControlSignals unpack_control_signal(char* packed_control_signal_as_char) 
{
  ControlSignals control_signal;
  char* token;

  // Unpack the desired speeed
  token = strtok(packed_control_signal_as_char, ",");
  control_signal.speed = atof(token);

  // Unpack the desired steering angle
  token = strtok(NULL, ",");
  control_signal.steering_angle = atof(token);

  return control_signal;
}

/*------------------------------------------------------------*/


/*Control robot from control signals received
  - left and right speeds
*/

void control_robot(ControlSignals control_signal)
{
  //TODO: receive controls for motor speeds LEFT and RIGHT

  // Direction for left and right motors going forward
  digitalWrite(RightMotorDirPin1, HIGH);
  digitalWrite(RightMotorDirPin1, LOW);
  digitalWrite(LeftMotorDirPin1, HIGH);
  digitalWrite(LeftMotorDirPin2, LOW);


  int left_speed = control_signal.speed_left * 2;
  int right_speed = control_signal.speed_right * 2;
  analogWrite(LeftSpeedPin, left_speed);
  analogWrite(RightSpeedPin, right_speed);
}

/*------------------------------------------------------------*/

/*Gather and send control signals to the laptop*/

SensorSignal get_sensor_signal()
{
  encoder_update();
  last_sensor_signal.encoder_left_count = encoder_left_count;
  last_sensor_signal.encoder_right_count = encoder_right_count;

  lidar_update();

  return last_sensor_signal;
}

// Get new lidar measurements
void lidar_update() {
  if (IS_OK(lidar.waitPoint())) {
    float distance = lidar.getCurrentPoint().distance;
    if (distance > 100 && current_num_lidar_rays < NumLidarRaysPerMsg) {
      int angle = int(lidar.getCurrentPoint().angle);
      current_num_lidar_rays += 1;
      current_lidar_scan_data +=  "," + String(angle) + "," + String(int(distance));
    }
  } else {
    analogWrite(RPLidarMotorPin, 255); //stop the rplidar motor
    
    // try to detect RPLIDAR... 
    rplidar_response_device_info_t info;
    if (IS_OK(lidar.getDeviceInfo(info, 100))) {
       // Detected...
       lidar.startScan();
       
       // Start motor rotating at max allowed speed
       analogWrite(RPLidarMotorPin, 255);
       delay(1000);
    }
  }
}

/*TODO: CHANGE THIS TO DIFFERENTIAL DRIVE LOGIC
  - Update states of both left and right motor
  - what happens when the robot is spinning in place?
  - are both motor encoder updated at the same time with the same numbers?
*/

// Get new encoder measurements
void encoder_update() { 
   a_state = digitalRead(EncoderOutputA); // Reads the "current" state of the outputA
   // If the previous and the current state of the outputA are different, that means a Pulse has occured
   if (a_state != encoder_a_last_state){     
     // If the outputB state is different to the outputA state, that means the encoder is rotating clockwise
     if (digitalRead(EncoderOutputB) != a_state) { 
       encoder_count ++;
     } else {
       encoder_count --;
     }
   } 
   encoder_a_last_state = a_state; // Updates the previous state of the outputA with the current state
 }

// Send a message with sensor signals to the laptop, but only after a preset time, e.g. 100 ms
void send_sensor_signal(SensorSignal sensor_signal)
{
  int new_time_tx = millis();
  if (new_time_tx - last_time_tx > SendDeltaTimeInMs) {
    String msg = String(sensor_signal.encoder_left_count) + ",";
    msg = msg + String(sensor_signal.encoder_right_count) + ",";
    msg = msg + String(current_num_lidar_rays);
    msg = msg + current_lidar_scan_data;
    reset_lidar_message();
    //Serial.print("Sending msg: ");
    //Serial.println(msg);

    Udp.beginPacket(remoteIP, remotePort);
    int   array_length  = msg.length()+1;
    char  msg_as_char_array[array_length];
    msg.toCharArray(msg_as_char_array, array_length);
    Udp.write(msg_as_char_array);
    Udp.endPacket();

    // Restart next msg
    last_sensor_signal.lidar_scan_data = "";
    last_sensor_signal.num_lidar_rays = 0;
    last_time_tx = new_time_tx;
  }
}

/*------------------------------------------------------------*/

// Wifi Status
void printWifiStatus() 
{
  // print the SSID of the network you're attached to:
  Serial.print("SSID: ");
  Serial.println(WiFi.SSID());

  // print your board's IP address:
  IPAddress ip = WiFi.localIP();
  Serial.print("IP Address: ");
  Serial.println(ip);

  // print the received signal strength:
  long rssi = WiFi.RSSI();
  Serial.print("signal strength (RSSI):");
  Serial.print(rssi);
  Serial.println(" dBm");
}
