#include <ros.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>

#define LIMIT_SWITCH_PIN 9  // Limit switch connected to D7 (Interrupt Pin) 
#define TOP_LIMIT_SWITCH_PIN 22
// Define pins
const int ledPin = 2;       // PWM pin for LED brightness control
const int ledDir = 3;       // Direction pin for the LED (optional)
const int relayPin = 4;     // Relay control pin (End-effector LED)
const int pumpPin = 5;      // Pump control pin
volatile bool switchPressed = false;  // Flag for switch state
volatile bool switchPressedTop = false;

std_msgs::Bool switch_top_msg;
std_msgs::Bool switch_msg;

// ROS NodeHandle
ros::NodeHandle nh;
ros::Publisher switch_top_pub("top_limit_switch_state", &switch_top_msg);
ros::Publisher switch_pub("limit_switch_state", &switch_msg);

// Callback for /led_brightness topic
void brightnessCallback(const std_msgs::Int32 &msg) {
  // Constrain the input value between 0 and 100
  int inputBrightness = constrain(msg.data, 0, 100);

  // Map the brightness value from 0-100 to 0-255
  int pwmValue = map(inputBrightness, 0, 100, 0, 255);

  // Set the direction and brightness of the LED
  digitalWrite(ledDir, LOW);  // Optional if direction control is needed
  analogWrite(ledPin, pwmValue);
}

// Callback for /endeffector_led topic
void endEffectorCallback(const std_msgs::Bool &msg) {
  if (msg.data) {
    digitalWrite(relayPin, HIGH); // Turn ON the relay
  } else {
    digitalWrite(relayPin, LOW);  // Turn OFF the relay
  }
}

// Callback for /pump topic
void pumpCallback(const std_msgs::Bool &msg) {
  if (msg.data) {
    digitalWrite(pumpPin, HIGH); // Turn ON the pump
  } else {
    digitalWrite(pumpPin, LOW);  // Turn OFF the pump
  }
}

// ROS Subscribers
ros::Subscriber<std_msgs::Int32> brightness_sub("/base_led_brightness", &brightnessCallback);
ros::Subscriber<std_msgs::Bool> endeffector_sub("/endeffector_led", &endEffectorCallback);
ros::Subscriber<std_msgs::Bool> pump_sub("/pump", &pumpCallback);

void setup() {
  
  // Initialize pins
  pinMode(ledPin, OUTPUT);
  pinMode(ledDir, OUTPUT);
  pinMode(relayPin, OUTPUT);
  pinMode(pumpPin, OUTPUT);
  pinMode(LIMIT_SWITCH_PIN, INPUT_PULLUP);  // Enable internal pull-up resistor
  pinMode(TOP_LIMIT_SWITCH_PIN, INPUT_PULLUP);  
  // Ensure initial states
  digitalWrite(ledDir, LOW);  // Direction LOW initially
  digitalWrite(relayPin, LOW); // Relay OFF initially
  digitalWrite(pumpPin, LOW);  // Pump OFF initially

  // Initialize ROS
  nh.initNode();
  nh.subscribe(brightness_sub);
  nh.subscribe(endeffector_sub);
  nh.subscribe(pump_sub);
  nh.advertise(switch_pub);
  nh.advertise(switch_top_pub);

}

void loop() {

  // Handle ROS communication
  switchPressedTop = digitalRead(TOP_LIMIT_SWITCH_PIN);
  switchPressed = digitalRead(LIMIT_SWITCH_PIN);  // True when pressed, false when released
  // Publish Limit Switch State
  switch_top_msg.data = switchPressedTop;
  switch_msg.data = switchPressed;
  switch_top_pub.publish(&switch_top_msg);
  switch_pub.publish(&switch_msg);
  nh.spinOnce();
  delay(10);
}

