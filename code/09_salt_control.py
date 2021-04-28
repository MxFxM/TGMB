import time
from tqdm import tqdm
import serial

ser = serial.Serial('/dev/ttyACM0', 115200)

print("SALT")
print("test script running")

print()

print("Sweeping x axis")
for angle in tqdm(range(60)):
    #ser.write(b'x')
    time.sleep(0.016)

for angle in tqdm(range(60)):
    time.sleep(0.016)

ser.close()
