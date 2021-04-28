import time
import math
from tqdm import tqdm
import serial
import struct

class SALT():

    ser = None

    def __init__(self):
        self.ser = serial.Serial('/dev/ttyACM0')

    def write(self, string):
        self.ser.write(string.encode('utf-8'))

    def set_angle(self, axis, angle):
        if axis == 'x':
            command = "x{}\r\n".format(angle)
            self.write(command)
        if axis == 'y':
            command = "y{}\r\n".format(angle)
            self.write(command)

    def set_laser(self, status):
        if status == 'on':
            command = "l1\r\n"
            self.write(command)
        if status == 'off':
            command = "l0\r\n"
            self.write(command)

    def close(self):
        self.ser.close()

s = SALT()

print("SALT")
print("test script running")

s.set_laser('on')
print("Sweeping x axis")
for angle in tqdm(range(60)):
    s.set_angle('x', angle+60)
    time.sleep(0.02)
print("Sweeping y axis")
for angle in tqdm(range(60)):
    s.set_angle('y', angle+60)
    time.sleep(0.02)
s.set_laser('off')

s.set_angle('x', 90)
s.set_angle('y', 90)
time.sleep(1)

s.set_laser('on')
print("Drawing a circle")
for angle in tqdm(range(500)):
    rads = angle * (2*math.pi)/500
    x = math.cos(rads)
    y = math.sin(rads)
    s.set_angle('x', x*20+90)
    s.set_angle('y', y*20+90)
    time.sleep(0.02)
s.set_laser('off')

s.set_angle('x', 90)
s.set_angle('y', 90)

s.close()
