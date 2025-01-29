import serial
import numpy as np
import cv2
import tensorflow as tf
import time

# Define the serial port and baud rate for communication
SERIAL_PORT = 'COM3'  # Change to your correct port
BAUD_RATE = 115200

# Open the serial connection to the OpenMV camera
ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
time.sleep(2)  # Allow time for the serial connection to establish

# Load your model (assuming you've already converted it to a TensorFlow Lite model)
interpreter = tf.lite.Interpreter(model_path="model_int8.tflite")
interpreter.allocate_tensors()

# Get the input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Check the expected input shape (should be (1, 96, 96, 1) for a 96x96 grayscale image)
print(f"Expected input shape: {input_details[0]['shape']}")

# Main loop for receiving images from OpenMV and running inference
while True:
    # Wait for an image from OpenMV (this will be raw grayscale pixel data)
    if ser.in_waiting > 0:
        # Read the raw image data from OpenMV (assuming 96x96 image)
        raw_image = ser.read(96 * 96)  # Adjust size if needed for your image size

        # Convert the raw bytes to a NumPy array and reshape it to 96x96
        img = np.frombuffer(raw_image, dtype=np.uint8).reshape((96, 96))

        # Resize to 96x96 (in case image size varies slightly)
        img_resized = cv2.resize(img, (96, 96))

        # Normalize the image to [0, 1] range and convert to INT8
        # Rescale the image to match INT8 input range [0, 255]
        img_resized = (img_resized * 255.0).astype(np.int8)

        # Add batch dimension (model expects a batch of images)
        img_input = np.expand_dims(img_resized, axis=0)  # Shape becomes (1, 96, 96)

        # Reshape the input to match model's input shape (1, 96, 96, 1)
        img_input = np.expand_dims(img_input, axis=-1)  # Add the channel dimension, making it (1, 96, 96, 1)

        # Set the input tensor (ensure it's in the correct INT8 format)
        interpreter.set_tensor(input_details[0]['index'], img_input)

        # Run inference
        interpreter.invoke()

        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Print the predicted output (e.g., confidence for each class)
        predicted_class = np.argmax(output_data)  # Get the class with the highest score
        confidence = output_data[0][predicted_class]

        # Print the result
        print(f"Predicted class: {predicted_class} with confidence: {confidence:.2f}")

        # Optionally, send the result back to OpenMV via serial
        # For example, send back the predicted class and confidence
        result_message = f"{predicted_class},{confidence:.2f}\n"
        ser.write(result_message.encode())

    time.sleep(0.1)  # Delay to avoid overloading the serial buffer
