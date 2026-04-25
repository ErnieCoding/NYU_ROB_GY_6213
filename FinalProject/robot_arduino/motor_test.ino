#define LeftSpeedPin 9
#define LeftMotorDirPin1 12
#define LeftMotorDirPin2 11

#define RightSpeedPin 6
#define RightMotorDirPin1 10
#define RightMotorDirPin2 13

#define RightEncoderOutputA 2
#define RightEncoderOutputB 3

#define LeftEncoderOutputA 5
#define LeftEncoderOutputB 4

#define LidarMotorPin 1


void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  Serial.println("Running motor test script!");
  
  pinMode(LeftMotorDirPin1, OUTPUT);
  pinMode(LeftMotorDirPin2, OUTPUT);
  pinMode(LeftSpeedPin, OUTPUT);

  pinMode(LeftEncoderOutputA, INPUT);
  pinMode(LeftEncoderOutputB, INPUT);
  
  pinMode(RightMotorDirPin1, OUTPUT);
  pinMode(RightMotorDirPin2, OUTPUT);
  pinMode(RightSpeedPin, OUTPUT);

  pinMode(RightEncoderOutputA, INPUT);
  pinMode(RightEncoderOutputB, INPUT);

  pinMode(LidarMotorPin, OUTPUT);

  
  digitalWrite(RightMotorDirPin1, HIGH);
  digitalWrite(RightMotorDirPin1, LOW);
  digitalWrite(LeftMotorDirPin1, HIGH);
  digitalWrite(LeftMotorDirPin2, LOW);

}

void loop() {

  Serial.println("IN THE LOOP!");
  // put your main code here, to run repeatedly:

  analogWrite(LeftSpeedPin, 100);
  analogWrite(RightSpeedPin, 100);

  analogWrite(LidarMotorPin, 255);

  delay(10000);
}
