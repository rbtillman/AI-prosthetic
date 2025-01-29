import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.widgets import PolygonSelector

def main():
    input_dir = "CANS-ALLIMAGES"
    output_file = "angles.csv"
    process_images(input_dir, output_file)

# Function to calculate the angle of a line relative to the x-axis
def calculate_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = np.degrees(np.arctan2(dy, dx))
    return abs(angle)

# Function to handle manual polygon selection
def on_select(verts):
    global polygon_points, mask_ready
    polygon_points = np.array(verts, dtype=np.int32)
    mask_ready = True

# Main function for processing a directory of images
def process_images(input_dir, output_file):
    global polygon_points, mask_ready

    # Create an output file to save angles
    with open(output_file, "w") as out_file:
        out_file.write("Image,Angle\n")

        # Loop through all images in the directory
        for image_file in os.listdir(input_dir):
            if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            # Load image
            image_path = os.path.join(input_dir, image_file)
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Display the image using Matplotlib
            fig, ax = plt.subplots()
            ax.imshow(image_rgb)
            ax.set_title(f"Draw a polygon on: {image_file}")

            # Reset global variables
            polygon_points = None
            mask_ready = False

            # Use Matplotlib's PolygonSelector for manual masking
            selector = PolygonSelector(ax, on_select, useblit=True)
            plt.show()

            # Wait until the user finishes selecting a polygon
            if polygon_points is None or not mask_ready:
                print(f"Skipping {image_file} (no polygon drawn).")
                continue

            # Calculate angles of the polygon's longest sides
            polygon_points = np.vstack([polygon_points, polygon_points[0]])  # Close the polygon
            distances = [
                (np.linalg.norm(polygon_points[i] - polygon_points[i + 1]), i)
                for i in range(len(polygon_points) - 1)
            ]
            distances.sort(reverse=True)  # Sort by longest distance
            (length1, idx1), (length2, idx2) = distances[:2]

            angle1 = calculate_angle(polygon_points[idx1], polygon_points[idx1 + 1])
            angle2 = calculate_angle(polygon_points[idx2], polygon_points[idx2 + 1])
            average_angle = (angle1 + angle2) / 2

            # Write the result to the output file
            out_file.write(f"{image_file},{average_angle:.2f}\n")
            print(f"Processed {image_file}: Average Angle = {average_angle:.2f}")

if __name__ == "__main__":
    main()
