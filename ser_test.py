'''
Serial read for images.  needs work.  
Written R. Tillman 1.18.25
'''


import serial

port = 'COM3'
baud_rate = 115200

try:
    with serial.Serial(port, baud_rate, timeout=1) as ser:
        print(f"Connected to {port}. Listening for data...")
        
        while True:
            if ser.in_waiting > 0:  # Check if data is available
                data = ser.readline().decode('utf-8').strip()
                print(f"Received: {data}")
                ser.close()
except serial.SerialException as e:
    print(f"Error: {e}")



