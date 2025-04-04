'''
Regressor inferencing and arm control script to run on raspberry pi
Written R. Tillman 3.18.2025

WIP- will need to integrate into Pi, maybe swap to tflite, etc

Should have enough error handling to run with no inputs for testing, if modifying, please keep this up.

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
    Returns a numpy feature array of type np.float32, or None if errors occur.
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
    Open the serial port at a specified baud rate.
    Returns the serial object.
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
    Reads a data line from the serial port and handles communication errors.\n
    If an error occurs, attempts to reconnect and returns (updated ser, None).\n
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

def writeSerial(ser, serial_port, baud_rate, data):
    """
    Sends data over a serial connection.

    Parameters:
    - ser: a serial.Serial object.
    - serial_port: the port of the device.
    - baud_rate: self explanitory.
    - data: the data to send. can be a string, number, or list/array of values.

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
        try:
            ser.close()
        except Exception:
            pass
        VERBOSE and print("WriteSerial: Attempting to reconnect to serial device...")
        ser = serOpen(serial_port, baud_rate)
        return ser
    return ser

def infer(model, data_line, expected_feature_shape):
    """
    Given a data line from ReadSerial, parses it and if valid,
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
    def __init__(self, num_fingers=5, num_sensors=4, verb=VERBOSE):
        self.num_fingers = num_fingers
        self.num_sensors = num_sensors
        self.finger_positions = np.zeros(num_fingers)

        # Position list: roration, lateral, pitch
        self.wrist_rotation = np.array([0,0,0])
        self.max_wrist_rotation = np.array([180,30,60])
        self.min_wrist_rotation = np.array([-180,-30,-60])

        self.wrist_gains = np.array([0.3,0.3,0])
        self.wrist_lambda = 1.0  # regularization constant

        self.thumb_rotation = 0
        self.max_thumb_rotation = 90
        self.min_thumb_rotation = 0

        self.grip_target = 100
        self.release_target = 0
        self.sensor_threshold = 50
        self.trigger_threshold = 5
        self.in_grip_state = [False] * num_fingers
        self.all_fingers_at_target = False

        self.Kp = 0.2
        self.wrist_Kp = 0.3
        self.verb = verb

    def get_force_matrix(self, data_line):
        """
        Given a data line from ReadSerial, generates a force matrix.

        Incoming serial data must be of the correct size to work properly.
        """
        fv = parseFeatureVector(data_line)

        # const matrices for calibration. Format: A + Bx 
        # currently initialized to do nothing
        A = np.zeros((self.num_fingers, self.num_sensors))
        B = np.ones((self.num_fingers, self.num_sensors))

        if fv is None:
            return None
        expected_length = self.num_fingers * self.num_sensors
        if fv.shape[0] != expected_length:
            self.verb and print(f"get_force_matrix: Expected {expected_length} values, got {fv.shape[0]}")
            return None
        
        fm = fv.reshape((self.num_fingers, self.num_sensors))
        fm = np.multiply(B,fm) + A

        return fm
    
    def start_grip(self, force_matrix=None):
        """ obsolete """
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
            # currently tries to grab fully.  
            self.finger_positions[:] = self.grip_target
            self.in_grip_state = [True] * self.num_fingers
            self.verb and print("Command: All fingers starting grip.")

    def continue_grip(self, force_matrix):
        """ obsolete """
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

    def finger_grip(self, force_matrix = None):
        """
        combines start grip and continue grip into one control function.  

        Needs to be called within a loop with updating force matrix.
        """
        if force_matrix.shape != (self.num_fingers, self.num_sensors):
            raise ValueError(f"Expected force matrix of shape ({self.num_fingers}, {self.num_sensors})")
        # working here        
        for i in range(self.num_fingers):
            avg_force = np.mean(force_matrix[i])
            if avg_force < self.sensor_threshold:
                error = self.sensor_threshold - avg_force
                adjustment = self.Kp * error
                self.finger_positions[i] += adjustment
                self.finger_positions[i] = np.clip(self.finger_positions[i], self.release_target, self.grip_target)
                self.all_fingers_at_target = False
        self.verb and print("Finger positions:", self.finger_positions)

    def release_grip(self, target = None):
        """
        Releases the grip.\n
        Optional param: release target.  
        """
        if target is None:
            self.finger_positions[:] = self.release_target
        else:
            self.finger_positions[:] = target
        self.in_grip_state = [False] * self.num_fingers
        self.verb and print("Command: Releasing all fingers.")

    def manual_finger(self, positions):
        """
        Manual finger control for debugging or gestures.\n
        Param: list of size num_fingers
        """
        if len(positions) != self.num_fingers:
            raise ValueError(f"Expected {self.num_fingers} positions, got {len(positions)}")
        self.finger_positions = np.clip(np.array(positions), self.release_target, self.grip_target)
        self.verb and print("Command: Manually setting finger positions.")

    def wrist_align(self, angle):
        """
        Attempts to control both rotation and lateral motion based on scalar error as camera is at 45 degrees\n
        Single degree control might work fine.  

        Parameter: angle (measured angle of object to grab)
        """
        error = 0 - angle

        # a and b represent the effectiveness (gains) of each wrist axis
        a = self.wrist_gains[0]
        b = self.wrist_gains[1]
        lam = self.wrist_lambda

        # Compute control commands using the least-squares solution
        u_rot = a * error / (lam + a**2 + b**2)
        u_lat = b * error / (lam + a**2 + b**2)

        # Update wrist commands
        self.wrist[0] += u_rot
        self.wrist[1] += u_lat

        self.wrist[0] = np.clip(self.wrist_rotation[0], self.min_wrist_rotation[0], self.max_wrist_rotation[0])
        self.wrist[1] = np.clip(self.wrist_rotation[1], self.min_wrist_rotation[1], self.max_wrist_rotation[1])

    def wrist_align_1D(self,angle):
        """
        1-degree wrist alignment. 
        """
        error = 0 - angle

        adjustment = self.wrist_Kp * error
        self.wrist_rotation += adjustment
        self.wrist_rotation = np.clip(self.wrist_rotation, -180, 180)
        self.verb and print(f"Command: Wrist angle = {angle:.2f} | Error = {error:.2f} | New Wrist Pos = {self.wrist_rotation:.2f}")


    def thumb_rotate(self, value):
        """
        rotates thumb to a specified value

        param: rotation (degrees)
        """
        self.thumb_rotation = np.clip(value, self.min_thumb_rotation, self.max_thumb_rotation)
        self.verb and print(f"Command: Thumb rotation set to {self.thumb_rotation:.2f}")

    def get_command_packet(self):
        """
        Builds a command packet for all positions of the hand.  

        Returns list to send via serialWrite.
        """
        return list(self.finger_positions) + list(self.wrist_rotation), [self.thumb_rotation]

    def flip(self):
        pose = 100 - [20, 20, 100, 20, 20]
        self.wrist_rotation = np.array([0,0,0])
        self.manual_finger(pose)

    def fist(self):
        pose = 100 - [0, 0, 0, 0, 0]
        self.wrist_rotation = np.array([0,0,0])
        self.manual_finger(pose)

    def point(self):
        pose = 100 - [20, 100, 20, 20, 20]
        self.wrist_rotation = np.array([0,0,0])
        self.manual_finger(pose)

    def reset(self):
        """Sets the hand command to the home position"""
        pose = [0, 0, 0, 0, 0]
        self.wrist_rotation = np.array([0,0,0])
        self.thumb_rotation = 0
        self.manual_finger(pose)


def main():
    # Portenta USB port
    port_port = '/dev/ttyUSB0'
    # Nano USB port
    nano_port = '/dev/ttyUSB1' #acm0?
    # Pyboard port
    pyb_port = '/dev/ttyUSB2'

    # Baud rate for all USB connections
    baud_rate = 115200

    # Acceptable angle to grip:
    GRIP_THRESH = 5

    # need a distance sensor!!!
    dist = None

    # initialize release variable.  need to implement release method
    release = False

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

        # if released last cycle, wait, then reset release var
        if release:
            time.sleep(1)
            release = False
        
        # Read portenta data and align wrist
        portenta, data1 = readSerial(portenta, port_port, baud_rate)
        time1 = time.time()

        # If portenta data available and not already gripping, infer angle and align wrist
        if data1 and not grip:
            angle = infer(model, data1, expected_feature_shape)
            inferTime = time.time() - time1
            VERBOSE and print(f"REGINF LOOP: inference time {inferTime:.3f} seconds")

            if (angle < GRIP_THRESH) and True: # replace true w/ dist sensor and thresh
                grip = True
            else:
                cmd.wrist_align(angle)
        
        # Read pyb data, then grip if ready
        pyb, data2 = readSerial(pyb, pyb_port, baud_rate)

        if data2 and grip:
            forceMatrix = cmd.get_force_matrix(data2)
            cmd.finger_grip(forceMatrix)

        if release:
            cmd.release_grip()

        # Generate command packet, send to nano
        commands = cmd.get_command_packet()
        nano = writeSerial(nano, nano_port, baud_rate, commands)

        end = time.time()
        elapsed = end - start
        VERBOSE and print(f"REGINF LOOP: Main loop time: {elapsed:.6f} seconds")

if __name__ == "__main__":
    main()