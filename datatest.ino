/***********************************************************
 *
 * Copyright 2021 Emma Li and Brian Li. All rights reserved.
 *
 *
 **********************************************************/

#include <CmdMessenger.h>  
#include <Servo.h>
#include <LiquidCrystal_I2C.h>
#include <Wire.h>

// Attach a new CmdMessenger object to the default Serial port
CmdMessenger cmdMessenger = CmdMessenger(Serial);
Servo baseServo;
Servo tiltServo;
int basePort = 8;
int tiltPort = 11;
double xAngle = 0;
double yAngle = 0;
double prevX = 60;
double prevY = 80;

int redButtonPort = 23;
unsigned long lastButtonPressTime;


//I2C pins declaration
LiquidCrystal_I2C lcd(0x27, 2, 1, 0, 4, 5, 6, 7, 3, POSITIVE); 

enum {
  kEulerangle,
  kTimeTicker,
};

void attachCommandCallbacks() {
  cmdMessenger.attach(kEulerangle, OnEulerangle);
}

// ------------------  C A L L B A C K S -----------------------

void OnEulerangle() {
  xAngle = cmdMessenger.readDoubleArg();
  yAngle = cmdMessenger.readDoubleArg();
  prevX += xAngle;
  prevY += yAngle;
  if(prevX < 0) {
    prevX = 0;
  }
  if(prevY < 0) {
    prevY = 0;
  }
  if(prevX > 120) {
    prevX = 120;
  }
  if(prevY > 150) {
    prevY = 150;
  }
  baseServo.write(prevX);
  tiltServo.write(prevY);
}

// ------------------ M A I N ( ) ----------------------


void setup() {
  // Listen on serial connection for messages from the PC
  // 115200 is the max speed on Arduino Uno, Mega, with AT8u2 USB
  Serial.begin(115200); 
  //Set motors to starting position (0,0).
  baseServo.attach(basePort);
  baseServo.write(60); 
  tiltServo.attach(tiltPort);
  tiltServo.write(80); 
  lcd.begin(16,4); 
  lcd.backlight();
  lcd.setCursor(0,0);
  lcd.print("System is ready...");
  cmdMessenger.printLfCr(); 
  attachCommandCallbacks();
  lastButtonPressTime = millis();
}

void debugMsg(char *debugMsg) {
  Serial.println(debugMsg);
}

void debugV(int v) {
  Serial.println(v, DEC);
}

void loop() {
  // Detect any button being pushed.
  int redState = digitalRead(redButtonPort); 
  if(redState == HIGH && millis() - lastButtonPressTime > 2000) {
    lastButtonPressTime = millis();
    // Send a command to desktop gaze tracking for output time ticker.
    Serial.write(1);
  } 

  // Process incoming serial data, and perform callbacks
  cmdMessenger.feedinSerialData();
}
