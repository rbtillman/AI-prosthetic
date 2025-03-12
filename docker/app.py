'''
Image masking program
Written R. Tillman 1.28.25

Updated 3.11.2025 for web app via docker

See ManImageMask for a simpler version that does the same thing locally

Copyright (c) 2025 Tillman. All Rights Reserved.
'''


import os
import cv2
import numpy as np
import math
import csv
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory

app = Flask(__name__)

# Directories for input images, output images, and CSV file
INPUT_DIR = 'images'
OUTPUT_DIR = '/persistent/CANS-REGMASK2' # changed to match better azure FS structure
CSV_FILE = os.path.join(OUTPUT_DIR, 'angles.csv')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def get_unprocessed_images():
    """Return list of filenames in INPUT_DIR that have not yet been processed."""
    all_images = [f for f in os.listdir(INPUT_DIR)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    # Check if the image exists in OUTPUT_DIR; if so, assume it’s been processed.
    unprocessed = [f for f in all_images if not os.path.exists(os.path.join(OUTPUT_DIR, f))]
    return unprocessed

# Home page: list images to process (only untraced images)
@app.route('/')
def index():
    images = get_unprocessed_images()
    return render_template('index.html', images=images)

# Page to trace an image
@app.route('/trace/<filename>')
def trace(filename):
    # If the image has already been processed, redirect to the next one.
    if os.path.exists(os.path.join(OUTPUT_DIR, filename)):
        next_img = get_next_image(filename)
        if next_img:
            return redirect(url_for('trace', filename=next_img))
        else:
            return "All images have been processed."
    return render_template('trace.html', filename=filename)

# Serve images from the input directory for tracing
@app.route('/input_image/<filename>')
def input_image(filename):
    return send_from_directory(INPUT_DIR, filename)

# Endpoint to process the traced polygon
@app.route('/process/<filename>', methods=['POST'])
def process(filename):
    data = request.get_json()
    if not data or 'points' not in data:
        return jsonify({"error": "No polygon points provided"}), 400

    mask_points = data['points']
    
    # Apply processing similar to your original script
    image_path = os.path.join(INPUT_DIR, filename)
    image = cv2.imread(image_path)
    if image is None:
        return jsonify({"error": "Image not found"}), 404

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if len(mask_points) > 2:
        pts = np.array(mask_points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    else:
        return jsonify({"error": "Polygon must have at least 3 points"}), 400

    masked_image = cv2.bitwise_and(image, image, mask=mask)
    output_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(output_path, masked_image)

    avg_angle = calculate_average_angle(mask_points)

    # Append result to CSV
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([filename, avg_angle])

    # Find next unprocessed image
    next_img = get_next_image(filename)
    return jsonify({"message": "Processing complete", "avg_angle": avg_angle, "next_image": next_img})

def get_next_image(current_filename):
    """Return the next unprocessed image filename after the current image, if any."""
    unprocessed = get_unprocessed_images()
    if current_filename in unprocessed:
        current_index = unprocessed.index(current_filename)
    else:
        current_index = -1
    if current_index + 1 < len(unprocessed):
        return unprocessed[current_index + 1]
    elif unprocessed:  # if current not in list but some remain, return first
        return unprocessed[0]
    else:
        return None

# Helper functions for angle calculations
def calculate_angle(p1, p2):
    delta_y = p2[1] - p1[1]
    delta_x = p2[0] - p1[0]
    angle = math.degrees(math.atan2(delta_y, delta_x))
    return angle

def calculate_average_angle(polygon_points):
    segment_lengths = []
    for i in range(len(polygon_points)):
        p1 = polygon_points[i]
        p2 = polygon_points[(i + 1) % len(polygon_points)]
        length = np.linalg.norm(np.array(p1) - np.array(p2))
        segment_lengths.append((length, p1, p2))
    segment_lengths.sort(reverse=True, key=lambda x: x[0])
    longest_segments = segment_lengths[:2]
    angles = [calculate_angle(seg[1], seg[2]) for seg in longest_segments]
    return sum(angles) / len(angles)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
