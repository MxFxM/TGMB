# SALT
## System for Advanced Laser Targeting

### General information

Plug in via USB and send commands via serial.
The Teensy serial will use 12Mbit/s default data, not depending on the baud rate.

Pins can be configured in the code.
Calibration values for the center and the maximum allowed angles can be set in the code.

### Commands

A command is sent as ascii characters followed by a \n to indicate a command is complete.
The first letter in the command specifies the command and the following letter(s) give further options.

xaaa with x being the command and aaa an integer number that is the angle to which the x axis servo shall move.

yaaa with y being the command and aaa an integer number that is the angle to which the y axis servo shall move.

sa with s being the command for a sweep demonstration.
The sweep command is only available if it was included during the compilation with the define ALLOW_DEMO flag.
This is because the command is blocking and will not allow further commands to be recognized while the demo is running, which could be fatal in the real application.
The sweep angles are set with a.
a can be either s for small (+/-10 degrees), m for medium (+/-20 degrees) or l for large (+/-30 degrees).
Alternatively a can be omitted for the most (+/-60 degrees).

la is the laser command.
For safety reasons, every wrong laser command will turn the laser off.
a as 1 turns the laser on and the auto mode off.
a as a turns the laser off and enables the auto mode.
a as m returns to the manual mode, so the auto mode is off, and the laser turns off. 
The auto mode will turn on the laser for 0.5 seconds after a new position for either axis is entered.
