#define LeftSpeedPin 9
#define LeftMotorDirPin1 12
#define LeftMotorDirPin2 11

#define RightSpeedPin 6
#define RightMotorDirPin1 7
#define RightMotorDirPin2 8

#define RightEncoderOutputA 4 // S2 - WORKS
#define RightEncoderOutputB 5

#define LeftEncoderOutputA 5 // DOESN'T WORK
#define LeftEncoderOutputB 3 // S2 - WORKS 

int encoder_left_state;
int encoder_left_last_state;
int encoder_left_count;

int encoder_right_state;
int encoder_right_last_state;
int encoder_right_count;

#define LidarMotorPin 1


void setup() {
  // put your setup code here, to run once:
  Serial.begin(460800);
  Serial.println("Running motor test script!");
  
  pinMode(LeftMotorDirPin1, OUTPUT);
  pinMode(LeftMotorDirPin2, OUTPUT);
  pinMode(LeftSpeedPin, OUTPUT);
  
  pinMode(RightMotorDirPin1, OUTPUT);
  pinMode(RightMotorDirPin2, OUTPUT);
  pinMode(RightSpeedPin, OUTPUT);

  pinMode(LeftEncoderOutputA, INPUT_PULLUP);
  pinMode(LeftEncoderOutputB, INPUT_PULLUP);
  pinMode(RightEncoderOutputA, INPUT_PULLUP);
  pinMode(RightEncoderOutputB, INPUT_PULLUP);

  pinMode(LidarMotorPin, OUTPUT);

  
  digitalWrite(RightMotorDirPin1, HIGH);
  digitalWrite(RightMotorDirPin2, LOW);
  digitalWrite(LeftMotorDirPin1, HIGH);
  digitalWrite(LeftMotorDirPin2, LOW);

}

int last_print_time = 0;
#define PrintDeltaTimeInMs 500

void loop() {
  // put your main code here, to run repeatedly:
  analogWrite(LidarMotorPin, 255);
  // analogWrite(LeftSpeedPin, 100);
  // analogWrite(RightSpeedPin, 100);

  // // Serial.print("RAW pin2: ");
  // // Serial.println(digitalRead(RightEncoderOutputA));

  // encoder_update();

  // int now = millis();

  // if (now - last_print_time > PrintDeltaTimeInMs) {
  //   Serial.println("======================");
  //   Serial.print("LEFT Encoder: ");
  //   Serial.println(encoder_left_count);

  //   Serial.print("RIGHT Encoder: ");
  //   Serial.println(encoder_right_count);

  //   last_print_time = now;
  // }


}


// Get new encoder measurements
void encoder_update() 
{

  encoder_left_state = digitalRead(LeftEncoderOutputB);
  encoder_right_state = digitalRead(RightEncoderOutputA);

  // Update encoder count for LEFT motor
  if (encoder_left_state != encoder_left_last_state) {
    encoder_left_count++;
  }

  // Update encdoder count for RIGHT motor
  if (encoder_right_state != encoder_right_last_state) {
    encoder_right_count++;
  }

  encoder_left_last_state = encoder_left_state;
  encoder_right_last_state = encoder_right_state;
  
}

