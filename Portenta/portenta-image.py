# Untitled - By: rtill - Tue Apr 15 2025

import sensor, image, time, pyb, struct

# Initialize the camera sensor.
sensor.reset()
sensor.set_pixformat(sensor.GRAYSCALE)
sensor.set_framesize(sensor.QVGA)  # QVGA: 320x240; adjust if needed.
sensor.skip_frames(time=2000)      # Give the sensor some time to adjust.

# Initialize the USB virtual COM port.
usb = pyb.USB_VCP()

clock = time.clock()

#while True:

    #img = sensor.snapshot()

    #print(img.to_ndarray("B"))
    #jpeg = img.compress(quality=100)

while True:
    # Grab a frame
    img = sensor.snapshot()

    # Compress to JPEG (quality 80–100 is fine for feature extraction)
    jpeg = img.compress(quality=80)

    # Prefix with 4‑byte little‑endian length
    length = struct.pack("<I", len(jpeg))

    # Send length + JPEG
    usb.send(length + jpeg)

    # Throttle frame rate (optional)
    time.sleep(0.05)
