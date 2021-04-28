import serial

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
