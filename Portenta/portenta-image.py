# Untitled - By: rtill - Tue Apr 15 2025

import sensor, image, time, pyb

# Initialize the camera sensor.
sensor.reset()
sensor.set_pixformat(sensor.GRAYSCALE)
sensor.set_framesize(sensor.QVGA)  # QVGA: 320x240; adjust if needed.
sensor.skip_frames(time=2000)      # Give the sensor some time to adjust.

# Initialize the USB virtual COM port.
usb = pyb.USB_VCP()

clock = time.clock()

while True:
    start = pyb.millis()
    img = sensor.snapshot()
    # data = img
    # data.to_ndarray("B")
    data = img.compress(quality=50)

    print("new thing\n")

    usb.send(data)

    end = pyb.millis()

    print("\n")
    print("elapsed (ms)", end-start)
    time.sleep(0.1)
