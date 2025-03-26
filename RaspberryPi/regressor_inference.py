'''
Regressor inferencing script to run on raspberry pi
Written R. Tillman 3.18.2025

WIP- will need to integrate into Pi, maybe swap to tflite, etc

Copyright (c) 2025 Tillman. All Rights Reserved.
'''

import serial
import os
import time
import numpy as np
import tensorflow as tf

VERBOSE = True

def parseFeatureVector(data_line):
    """
    Parses a comma-separated string of numbers into a numpy array.
    Returns a numpy feature array of type np.float32.
    """
    try:
        parts = data_line.strip().split(',')
        feature_vector = [float(x) for x in parts if x]
        return np.array(feature_vector, dtype=np.float32)
    except Exception as e:
        VERBOSE and print("parseFeatureVector: Error parsing feature vector:", e)
        return None

def serOpen(serial_port, baud_rate):
    """
    Open the serial port and return the serial object.
    """
    ser = None
    while ser is None:
        try:
            ser = serial.Serial(serial_port, baud_rate, timeout=1)
            VERBOSE and print(f"SerOpen: Serial connected on {serial_port} at {baud_rate} baud.")
        except Exception as e:
            VERBOSE and print("SerOpen: Failed to open serial port:", e)
            time.sleep(1)
    
    time.sleep(2)
    return ser

def readSerial(ser, serial_port, baud_rate):
    """
    Reads a data line from the serial port and handles communication errors.
    If an error occurs, attempts to reconnect and returns (updated ser, None).
    Otherwise, returns (ser, data_line) if data is available or (ser, None) if not.
    """
    try:
        if ser.in_waiting:
            data_line = ser.readline().decode('utf-8')
            return ser, data_line
    except (serial.SerialException, OSError) as se:
        VERBOSE and print("readSerial: Communication error:", se)
        try:
            ser.close()
        except Exception:
            pass
        VERBOSE and print("readSerial: Attempting to reconnect to serial device...")
        ser = serOpen(serial_port, baud_rate)
        return ser, None
    except Exception as e:
        VERBOSE and print("readSerial: Unexpected error:", e)
        time.sleep(0.1)
        return ser, None
    return ser, None

def writeSerial(ser, data):
    """
    Sends data over an open serial connection.

    Parameters:
    - ser: a serial.Serial object that is already open.
    - data: the data to send. This can be a string, number, or list/array of values.

    The function converts the data to a comma-separated string and appends a newline.
    """
    try:
        # Convert data to string
        if isinstance(data, (list, np.ndarray)):
            msg = ','.join(str(x) for x in data)
        else:
            msg = str(data)
        
        msg += '\n'  # Add newline so the receiver knows the message ended
        ser.write(msg.encode('utf-8'))
        VERBOSE and print(f"writeSerial: Sent -> {msg.strip()}")
    except Exception as e:
        VERBOSE and print("writeSerial: Failed to send data:", e)

def infer(model, data_line, expected_feature_shape):
    """
    Given a data line from the serial port, parses it and, if valid,
    performs inference with the given model.
    Returns the predicted angle (or 0 if invalid input).
    """
    angle = 0
    if not data_line:
        return angle

    feature_vector = parseFeatureVector(data_line)
    if feature_vector is None:
        return angle

    if feature_vector.shape[0] != expected_feature_shape[0]:
        VERBOSE and print("INFER: Received feature vector of incorrect length:", feature_vector.shape)
        return angle

    input_data = np.expand_dims(feature_vector, axis=0)
    prediction = model.predict(input_data)
    angle = prediction[0][0]  # Assumes a 1x1 output
    VERBOSE and print("INFER: Predicted angle: {:.2f} degrees".format(angle))
    return angle

class Command:
    def __init__(self, num_fingers=5, num_sensors=4, verb = False):
        self.num_fingers = num_fingers
        self.num_sensors = num_sensors
        self.finger_positions = np.zeros(num_fingers)
        self.wrist_rotation = 0
        self.grip_target = 100           # Fully closed finger position
        self.release_target = 0          # Fully open finger position
        self.sensor_threshold = 50       # Target average pressure
        self.trigger_threshold = 5       # Minimum force to start feedback grip
        self.in_grip_state = [False] * num_fingers
        self.Kp = 0.2                     # Finger Kp
        self.wrist_Kp = 0.3               # Wrist Kp
        self.verb = verb

    def start_grip(self, force_matrix=None):
        if force_matrix is not None and force_matrix.shape == (self.num_fingers, self.num_sensors):
            for i in range(self.num_fingers):
                avg_force = np.mean(force_matrix[i])
                if avg_force > self.trigger_threshold:
                    self.in_grip_state[i] = True
                    self.verb and print(f"Command: Finger {i+1} detected contact, switching to continue grip.")
                else:
                    self.finger_positions[i] = self.grip_target
                    self.verb and print(f"Command: Finger {i+1} gripping to {self.grip_target}")
        else:
            self.finger_positions[:] = self.grip_target
            self.in_grip_state = [True] * self.num_fingers
            self.verb and print("Command: All fingers starting grip.")

    def continue_grip(self, force_matrix):
        if force_matrix.shape != (self.num_fingers, self.num_sensors):
            raise ValueError(f"Expected force matrix of shape ({self.num_fingers}, {self.num_sensors})")
        
        for i in range(self.num_fingers):
            if not self.in_grip_state[i]:
                continue

            avg_force = np.mean(force_matrix[i])
            error = self.sensor_threshold - avg_force
            adjustment = self.Kp * error
            self.finger_positions[i] += adjustment
            self.finger_positions[i] = np.clip(self.finger_positions[i], self.release_target, self.grip_target)

            self.verb and print(f"Command: Finger {i+1} | Force = {avg_force:.2f} | Error = {error:.2f} | New Pos = {self.finger_positions[i]:.2f}")

    def release_grip(self):
        self.finger_positions[:] = self.release_target
        self.in_grip_state = [False] * self.num_fingers
        self.verb and print("Command: Releasing all fingers.")

    def manual_finger_command(self, positions):
        if len(positions) != self.num_fingers:
            raise ValueError(f"Expected {self.num_fingers} positions, got {len(positions)}")
        self.finger_positions = np.clip(np.array(positions), 0, 100)
        self.verb and print("Command: Manually setting finger positions.")

    def wrist_command(self, angle):
        """
        Proportional controller to align wrist with some object
        """
        error = 0 - angle
        adjustment = self.wrist_Kp * error
        self.wrist_rotation += adjustment
        self.wrist_rotation = np.clip(self.wrist_rotation, -180, 180)

        self.verb and print(f"Command: Wrist angle = {angle:.2f} | Error = {error:.2f} | New Wrist Pos = {self.wrist_rotation:.2f}")

    def get_command_packet(self):
        return list(self.finger_positions) + [self.wrist_rotation]

def main():
    # Portenta USB port
    port_port = '/dev/ttyUSB0'
    # Nano USB port
    nano_port = '/dev/ttyUSB1' #acm0?
    # Pyboard port
    pyb_port = '/dev/ttyUSB2'

    # Baud rate for all USB connections
    baud_rate = 115200

    # Initialize serial objects
    portenta = serOpen(port_port, baud_rate)
    nano = serOpen(nano_port, baud_rate)
    pyb = serOpen(pyb_port, baud_rate)

    # Initialize finger data
    fingers = 5 # n fingers
    sensors = 4 # m sensors per finger
    fingers_forces = np.zeros((fingers,sensors))
    fingers_positions = np.zeros(fingers)

    # Create the command object for controls
    cmd = Command(fingers, sensors, VERBOSE)

    # Load the model
    model = tf.keras.models.load_model("regressor.keras")
    VERBOSE and print("REGINF: Model loaded successfully.")
    model.summary()

    # Expected feature vector shape (excluding batch dimension)
    expected_feature_shape = model.input_shape[1:]
    VERBOSE and print("REGINF: Expected feature vector shape:", expected_feature_shape)
    VERBOSE and print("REGINF: Listening for feature vectors on serial port:", port_port)

    # MAIN LOOP
    while True:
        start = time.time()
        portenta, data1 = readSerial(portenta, port_port, baud_rate)
        if data1:
            angle = infer(model, data1, expected_feature_shape)
        
        pyb, data2 = readSerial(pyb, pyb_port, baud_rate)
        if data2:
            pass # Add what to do with finger data

        end = time.time()
        elapsed = end - start
        print(f"REGINF: Main loop time: {elapsed:.6f} seconds")

if __name__ == "__main__":
    main()