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
        print("PARSE: Error parsing feature vector:", e)
        return None

def serOpen(serial_port, baud_rate):
    """
    Open the serial port and return the serial object.
    """
    ser = None
    while ser is None:
        try:
            ser = serial.Serial(serial_port, baud_rate, timeout=1)
            print(f"SerOpen: Serial connected on {serial_port} at {baud_rate} baud.")
        except Exception as e:
            print("SerOpen: Failed to open serial port:", e)
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
        print("SERIAL: Communication error:", se)
        try:
            ser.close()
        except Exception:
            pass
        print("SERIAL: Attempting to reconnect to serial device...")
        ser = serOpen(serial_port, baud_rate)
        return ser, None
    except Exception as e:
        print("SERIAL: Unexpected error:", e)
        time.sleep(0.1)
        return ser, None
    return ser, None

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
        print("INFER: Received feature vector of incorrect length:", feature_vector.shape)
        return angle

    input_data = np.expand_dims(feature_vector, axis=0)
    prediction = model.predict(input_data)
    angle = prediction[0][0]  # Assumes a 1x1 output
    print("INFER: Predicted angle: {:.2f} degrees".format(angle))
    return angle

def main():
    # Portenta USB port
    serial_port = '/dev/ttyUSB0'
    # Nano USB port
    nano_port = '/dev/ttyUSB1'

    # Baud rate for all USB connections
    baud_rate = 115200

    # Initialize serial objects
    ser = serOpen(serial_port, baud_rate)

    # Load the model
    model = tf.keras.models.load_model("regressor.keras")
    print("REGINF: Model loaded successfully.")
    model.summary()

    # Expected feature vector shape (excluding batch dimension)
    expected_feature_shape = model.input_shape[1:]
    print("REGINF: Expected feature vector shape:", expected_feature_shape)
    print("REGINF: Listening for feature vectors on serial port:", serial_port)

    # MAIN LOOP
    while True:
        ser, data_line = readSerial(ser, serial_port, baud_rate)
        if data_line:
            angle = infer(model, data_line, expected_feature_shape)

if __name__ == "__main__":
    main()