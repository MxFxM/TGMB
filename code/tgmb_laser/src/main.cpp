#include <Arduino.h>
#include <Servo.h>

// use this define to include demonstration functionality
// this functionality may block real execution
#define ALLOW_DEMO

#define X_CENTER 80
#define Y_CENTER 95

// define the pins for the servos
#define X_AXIS 22
#define Y_AXIS 21

// define the pin fot the laser
#define LASER 20

// this is the USB connection
#define USB Serial

// two servos, one for each axis
Servo x_servo;
Servo y_servo;

// keep the current position
int x_location = 0;
int y_location = 0;

// initialize a buffer to receive serial data
String buffer = "                                                                ";
uint8_t buffer_pointer = 0;

// state of the laser
bool laser_state = false;
bool auto_laser = false;
unsigned long laser_timer = 0;

void setup() {
  // attach the control to the pins
  x_servo.attach(X_AXIS);
  y_servo.attach(Y_AXIS);

  // center both servos
  x_servo.write(90);
  y_servo.write(90);
  x_location = 90;
  y_location = 90;

  // make laser pin an output
  pinMode(LASER, OUTPUT);

  USB.begin(9600);
  while (USB.available() == 0) {}
  USB.println("welcome");
  
}

void loop() {
  // if a new character is received
  if (USB.available()) {
    // how many new characters are there?
    // this limits the run time of each loop
    uint8_t new_chars = USB.available();

    // only accept the number of new characters from the beginning of the loop
    // this will guarante that the loop is executed some what consistently
    for (uint8_t nc = 0; nc < new_chars; nc++) {
      // read the next character
      buffer.setCharAt(buffer_pointer, (char)USB.read());
      buffer_pointer++;

      // if a newline character is received
      if (buffer.charAt(buffer_pointer-1) == '\n') {
        // print the received string
        USB.println(buffer.substring(0, buffer_pointer-2));

        // command for setting the x axis
        if (buffer.charAt(0) == 'x') {
          // the characters after the x are interpreted as the angle
          int angle = buffer.substring(1,buffer_pointer-2).toInt();
          // the bounds are checked
          if (angle >= 30 && angle <= 150) {
            // servo angle is set
            x_location = angle;
            x_servo.write(x_location);
            // if the laser is set to auto mode
            if (auto_laser) {
              // turn the laser on
              laser_state = true;
              digitalWriteFast(LASER, HIGH);
              // set the timer to turn off
              laser_timer = micros() + 250000;
            }
          } else {
            USB.println("That is out of bounds.");
          }
        }

        // command for setting the y axis
        if (buffer.charAt(0) == 'y') {
          // the characters after the y are interpreted as the angle
          int angle = buffer.substring(1,buffer_pointer-2).toInt();
          // the bounds are checked
          if (angle >= 30 && angle <= 150) {
            // servo angle is set
            y_location = angle;
            y_servo.write(y_location);
            // if the laser is set to auto mode
            if (auto_laser) {
              // turn the laser on
              laser_state = true;
              digitalWriteFast(LASER, HIGH);
              // set the timer to turn off
              laser_timer = micros() + 250000;
            }
          } else {
            USB.println("That is out of bounds.");
          }
        }

        // laser command
        if (buffer.charAt(0) == 'l') {
          if (buffer.charAt(1) == '1') { // turn on
            // this disables the auto laser by default
            laser_state = true;
            auto_laser = false;
          } else if (buffer.charAt(1) == 'a') { // automatic mode
            // automatic laser will turn on for 0.5 seconds after setting a new location
            auto_laser = true;
            laser_state = false;
          } else if (buffer.charAt(1) == 'm') { // manual mode
            auto_laser = false;
            laser_state = false;
          } else { // every other command turns the laser off
            laser_state = false;
          }
          
          // update the laser state
          digitalWriteFast(LASER, laser_state);
        }

        // for demonstration purposes there is a sweep function
        // thif function will block further code execution while it is running
        // it can be disabled
        #ifdef ALLOW_DEMO
        // s for sweep
        if (buffer.charAt(0) == 's') {
          // determine sweep size
          int min_angle = 0;
          int max_angle = 180;
          if (buffer.charAt(1) == 's') {
            min_angle = 80;
            max_angle = 100;
          } else if (buffer.charAt(1) == 'm') {
            min_angle = 70;
            max_angle = 110;
          } else if (buffer.charAt(1) == 'l') {
            min_angle = 60;
            max_angle = 120;
          } else {
            min_angle = 30;
            max_angle = 150;
          }

          // disable the laser
          digitalWriteFast(LASER, LOW);

          // slowly to one corner
          bool both_returned = false;
          int x_temp_location = x_location;
          int y_temp_location = y_location;
          while (both_returned == false) {
            both_returned = true;
            if (x_temp_location > min_angle) {
              x_temp_location--;
              both_returned = false;
            }
            if (y_temp_location > min_angle) {
              y_temp_location--;
              both_returned = false;
            }
            x_servo.write(x_temp_location);
            y_servo.write(y_temp_location);
            delay(16);
          }
          
          // wait a little
          delay(500);

          // enable the laser
          digitalWriteFast(LASER, HIGH);

          // sweep x axis forward
          for (uint8_t angle = min_angle; angle < max_angle; angle++) {
            x_servo.write(angle);
            delay(16); // about 1 second for the sweep
          }

          // sweep y axis forward
          for (uint8_t angle = min_angle; angle < max_angle; angle++) {
            y_servo.write(angle);
            delay(16); // about 1 second for the sweep
          }

          // sweep x axis backwards
          for (uint8_t angle = max_angle; angle > min_angle; angle--) {
            x_servo.write(angle);
            delay(16); // about 1 second for the sweep
          }

          // sweep y axis backwards
          for (uint8_t angle = max_angle; angle > min_angle; angle--) {
            y_servo.write(angle);
            delay(16); // about 1 second for the sweep
          }

          // disable the laser
          digitalWriteFast(LASER, LOW);

          // return to starting position
          both_returned = false;
          x_temp_location = 30;
          y_temp_location = 30;
          while (both_returned == false) {
            both_returned = true;
            if (x_temp_location < x_location) {
              x_temp_location++;
              both_returned = false;
            }
            if (y_temp_location < y_location) {
              y_temp_location++;
              both_returned = false;
            }
            x_servo.write(x_temp_location);
            y_servo.write(y_temp_location);
            delay(16);
          }

          // set the laser to its previous state
          digitalWriteFast(LASER, laser_state);
        }
        #endif

        // reset the buffer
        buffer = "                                                                ";
        buffer_pointer = 0;
      }

      // avoid overflows
      if (buffer_pointer == 64) {
        // reset the buffer
        buffer = "                                                                ";
        buffer_pointer = 0;
      }
    }
  }

  // check the auto laser
  if (auto_laser) {
    // if the laser is on
    if (laser_state) {
      // if the timer is over
      if (micros() >= laser_timer) {
        // turn the laser off
        laser_state = false;
      }
    }
  }

  // update the position
  x_servo.write(x_location);
  y_servo.write(y_location);
  digitalWriteFast(LASER, laser_state);
}