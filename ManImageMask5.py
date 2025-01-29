'''
Image masking program
Written R. Tillman 1.28.25

Usage: 
- put image folder in same directory as program, set input directory name in main().
- run program
- left click on image to place a point.  Ensure that the longest 2 segments of the polygon are the edges of the object.
- when polygon is fully drawn, press enter to proceed
- to remove the polygon, press esc.  The polygon will clear when you start drawing a new one

Requires cv2, numpy, and matplotlib.  To install, run "pip install cv2 numpy matplotlib" in terminal

Copyright (c) 2025 Tillman. All Rights Reserved.
'''

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import patches
import math
import csv

# set the path for saving the output images and CSV file here:
output_dir = 'CANS-MANMASK'
csv_file = os.path.join(output_dir, 'angles.csv')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def main():
    input_directory = 'CANS-ALLIMAGES'  # input images: put directory name of images here, ie, 'CANS-BATCH1' or whatever
    process_images_in_directory(input_directory, output_dir)

# Function to calculate the angle of a line segment
def calculate_angle(p1, p2):
    delta_y = p2[1] - p1[1]
    delta_x = p2[0] - p1[0]
    angle = (math.degrees(math.atan2(delta_y, delta_x)))
    return angle

# Function to calculate the average angle of the two longest segments
def calculate_average_angle(polygon_points):

    segment_lengths = []
    for i in range(len(polygon_points)):
        p1 = polygon_points[i]
        p2 = polygon_points[(i + 1) % len(polygon_points)]
        length = np.linalg.norm(np.array(p1) - np.array(p2))
        segment_lengths.append((length, p1, p2))

    # angle of can by 2 longest polygon segments
    segment_lengths.sort(reverse=True, key=lambda x: x[0])
    longest_segments = segment_lengths[:2]
    angles = [calculate_angle(seg[1], seg[2]) for seg in longest_segments]
    
    return sum(angles) / len(angles)

# Function to manually mask an image
def manual_mask_image(image_path, output_dir):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots()

    ax.imshow(image_rgb)

    polygon = None
    mask_points = []

    # Function to handle mouse events
    def onclick(event):
        nonlocal polygon, mask_points

        # Capture the click coordinates
        if event.inaxes:
            x, y = int(event.xdata), int(event.ydata)
            mask_points.append((x, y))

            # Clear the current polygon if it exists and draw the new one
            if polygon:
                polygon.remove()

            # Create a new polygon and plot it
            polygon = patches.Polygon(mask_points, closed=True, fill=False, edgecolor='r', linewidth=2)
            ax.add_patch(polygon)

            plt.draw()

    # Function to handle key press events
    def on_key(event):
        nonlocal mask_points

        if event.key == 'enter':
            if len(mask_points) > 2:
                plt.close()  # Close the current image and proceed
            else:
                print("bruh draw a polygon first")
        elif event.key == 'escape':
            mask_points = []
            print('you done messed up a-aron, begin drawing new polygon')

    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', on_key)

    print("Click on the image to draw a polygon, press enter when done")
    plt.show()

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if len(mask_points) > 2:
        pts = np.array(mask_points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

    # Apply the mask to the image (only keep the masked area)
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, masked_image)

    print(f"Saved masked image to {output_path}\n")

    avg_angle = calculate_average_angle(mask_points)
    print(f"Average angle of the two longest segments: {avg_angle} degrees\n")

    # Save the result to a CSV file
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([os.path.basename(image_path), avg_angle])

def process_images_in_directory(input_dir, output_dir):

    for filename in os.listdir(input_dir):
        image_path = os.path.join(input_dir, filename)
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            print(f"Processing {filename}")
            manual_mask_image(image_path, output_dir)


if __name__ == "__main__":
    main()
