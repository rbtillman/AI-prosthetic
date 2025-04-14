from machine import Pin, ADC
from time import sleep

# Define pins for first and second 2x2 sensor arrays
rowpins1 = [Pin(5, Pin.OUT), Pin(6, Pin.OUT)]
rowpins2 = [Pin(7, Pin.OUT), Pin(8, Pin.OUT)]

# Column ADC inputs
colpins = [ADC(Pin(26)), ADC(Pin(27))]

# Read a single 2x2 sensor array, given its row pin list
def read_array(rowpins):
    sensor = []
    for row in range(2):
        # Set current row HIGH, others LOW
        for i in range(2):
            rowpins[i].value(1 if i == row else 0)

        rowdata = []
        for col in range(2):
            value = colpins[col].read_u16()
            adjusted = value - 36000  # Similar to `sensorValue - 560` in scaled 16-bit
            rowdata.append(str(adjusted))

        sensor.append(",".join(rowdata))
    
    # Deactivate all rows after reading
    for pin in rowpins:
        pin.value(0)

    return ";".join(sensor)

# Main loop
while True:
    data1 = read_array(rowpins1)
    data2 = read_array(rowpins2)
    print(data1 + "|" + data2)
    sleep(0.05)  # 50ms delay

