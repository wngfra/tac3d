import serial

ser  = serial.Serial(port='/dev/ttyACM0', baudrate=115200, timeout=1)

def readout(ser):
    data = ser.readline()
    return data

while True:
    value = readout(ser)
    print(value)