import serial
import numpy as np
import cv2
import tensorflow.lite as tflite

# Load TensorFlow Lite model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Labels for the classes
labels = ["Vertical", "Horizontal", "Nothing"]  # Replace with your model's labels

# Set up serial communication
ser = serial.Serial('COM3', 115200, timeout=1)  # Replace 'COM3' with your OpenMV port

while True:
    try:
        # Read the size of the incoming image (4 bytes)
        img_size = int.from_bytes(ser.read(4), 'big')

        if img_size > 0:
            # Read the actual image data
            img_data = ser.read(img_size)

            # Convert byte array to a numpy array
            img_np = np.frombuffer(img_data, dtype=np.uint8)

            # Decode the image using OpenCV
            img = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                # Resize the image to match the model input dimensions
                input_shape = input_details[0]['shape']
                img_resized = cv2.resize(img, (input_shape[1], input_shape[2]))  # Height x Width
                img_normalized = img_resized / 255.0  # Normalize pixel values
                img_input = np.expand_dims(img_normalized, axis=0)  # Add batch dimension

                # Set the model input
                interpreter.set_tensor(input_details[0]['index'], img_input.astype(np.float32))

                # Run the model
                interpreter.invoke()

                # Get the model output
                output_data = interpreter.get_tensor(output_details[0]['index'])
                predicted_label = labels[np.argmax(output_data)]
                confidence = np.max(output_data)

                # Display the result
                cv2.putText(
                    img, f"{predicted_label}: {confidence:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2
                )
                cv2.imshow("Received Image", img)
                cv2.waitKey(1)

                # Debug feedback
                print(f"Prediction: {predicted_label}, Confidence: {confidence:.2f}")
    except Exception as e:
        print(f"Error: {e}")
