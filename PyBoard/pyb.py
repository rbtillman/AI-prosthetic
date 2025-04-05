"""
pyboard script

Lucas put your code here
"""

import serial
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time

# Initialize serial connection
ser = serial.Serial('COM3', 9600, timeout=0.1)  # Small timeout to avoid blocking

# Set up plot for time series data
plt.ion()  # Interactive mode on
fig, ax = plt.subplots(figsize=(10, 6))

# We'll keep a rolling window of data points to prevent the plot from getting too crowded
max_points = 100  # Number of points to display at once
time_points = deque(maxlen=max_points)
data_channels = {
    'ch1_1': deque(maxlen=max_points),  # Row 1, Column 1
    'ch1_2': deque(maxlen=max_points),  # Row 1, Column 2
    'ch2_1': deque(maxlen=max_points),  # Row 2, Column 1
    'ch2_2': deque(maxlen=max_points)   # Row 2, Column 2
}

# Voltage limits
VOLTAGE_MIN = -120
VOLTAGE_MAX = 120

# Create lines for each channel
lines = {
    'ch1_1': ax.plot([], [], label='Row 1, Col 1')[0],
    'ch1_2': ax.plot([], [], label='Row 1, Col 2')[0],
    'ch2_1': ax.plot([], [], label='Row 2, Col 1')[0],
    'ch2_2': ax.plot([], [], label='Row 2, Col 2')[0]
}

ax.set_xlabel('Time (s)')
ax.set_ylabel('Voltage (mV)')
ax.set_title('Live Voltage Data from Arduino (Clipped to ±120)')
ax.legend()
ax.grid(True)

# Add horizontal lines showing the limits
ax.axhline(y=VOLTAGE_MAX, color='r', linestyle='--', alpha=0.5)
ax.axhline(y=VOLTAGE_MIN, color='r', linestyle='--', alpha=0.5)
ax.text(0.02, 0.95, f'Max: {VOLTAGE_MAX}', transform=ax.transAxes, color='r')
ax.text(0.02, 0.05, f'Min: {VOLTAGE_MIN}', transform=ax.transAxes, color='r')

def clamp_voltage(value):
    """Ensure voltage stays within specified bounds"""
    return max(VOLTAGE_MIN, min(VOLTAGE_MAX, value))

def get_arduino_data():
    if ser.in_waiting > 0:
        try:
            # Read data until a newline character is encountered
            line = ser.read_until(b'\n').decode('utf-8').strip()
            if not line or ';' not in line:
                return None
            
            # Split rows and columns
            rows = line.split(';')
            if len(rows) != 2:
                return None

            values = []
            for row in rows:
                cols = row.split(',')
                if len(cols) != 2:
                    return None  # 2 columns per row
                # Apply voltage clamping to each value
                values.append([clamp_voltage(float(x)) for x in cols])

            return values
        except Exception as e:
            print(f"Error parsing data: {e}")
            return None

def update_plot():
    new_data = get_arduino_data()
    if new_data:
        current_time = time.time()
        time_points.append(current_time)
        
        # Update data for each channel (values already clamped)
        data_channels['ch1_1'].append(new_data[0][0])
        data_channels['ch1_2'].append(new_data[0][1])
        data_channels['ch2_1'].append(new_data[1][0])
        data_channels['ch2_2'].append(new_data[1][1])
        
        # Update the plot for each channel
        for channel in data_channels:
            lines[channel].set_data(time_points, data_channels[channel])
        
        # Adjust the x-axis to show the most recent data
        if len(time_points) > 1:
            ax.set_xlim(time_points[0], time_points[-1])
        
        # Set fixed y-axis limits
        ax.set_ylim(VOLTAGE_MIN - 5, VOLTAGE_MAX + 5)
        
        plt.draw()
        plt.pause(0.001)  # Small pause to allow the plot to update

try:
    start_time = time.time()
    while True:
        update_plot()
except KeyboardInterrupt:
    print("Program interrupted. Exiting gracefully.")
    ser.close()  # close serial
    plt.close()  # close plot
