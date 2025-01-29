import cv2
import numpy as np
import os
import math

# Output settings
output_dir = "CANS-AUTOMASKED"  # Directory to save extracted cans
angle_file = "angles.csv"  # File to save angles

def main():
    input_directory = "CANS-ALLIMAGES"  # input directory
    process_directory(input_directory)

def calculate_angle(pt1, pt2):
    """Calculate the angle of a line relative to the horizontal axis."""
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    angle = math.degrees(math.atan2(dy, dx))
    return angle


def find_longest_sides(contour):
    """Find the two longest sides of a contour."""
    sides = []
    for i in range(len(contour)):
        pt1 = tuple(contour[i][0])
        pt2 = tuple(contour[(i + 1) % len(contour)][0])  # Wrap around to the first point
        length = math.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
        sides.append((length, pt1, pt2))
    sides.sort(reverse=True, key=lambda x: x[0])  # Sort by length (descending)
    return sides[:2]  # Return the two longest sides


def process_image(image_path, image_name):
    """Process a single image to extract cans and calculate angles."""
    # Load the image and convert to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to segment the cans
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Prepare output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    angles = []  # Store angles for this image

    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < 500:  # Skip small contours (noise)
            continue

        # Find the convex hull (polygon approximation of the contour)
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Find the two longest sides
        longest_sides = find_longest_sides(approx)
        if len(longest_sides) < 2:
            continue

        # Calculate angles of the longest sides relative to the horizontal axis
        side_angles = [calculate_angle(pt1, pt2) for _, pt1, pt2 in longest_sides]
        average_angle = sum(side_angles) / len(side_angles)
        angles.append(average_angle)

        # Draw the contour and the longest sides
        output_image = image.copy()
        cv2.drawContours(output_image, [approx], -1, (0, 255, 0), 2)
        for _, pt1, pt2 in longest_sides:
            cv2.line(output_image, pt1, pt2, (255, 0, 0), 2)

        # Save the extracted contour with its angles
        output_file = os.path.join(output_dir, f"{image_name}_contour_{i}.png")
        cv2.imwrite(output_file, output_image)
        print(f"Saved contour image to {output_file}")

    # Save the angles for this image
    if angles:
        with open(angle_file, "a") as f:
            f.write(f"{image_name},{','.join(f'{a:.2f}' for a in angles)}\n")
        print(f"Saved angles for {image_name}: {angles}")


def process_directory(input_dir):
    """Process all images in a directory."""
    # Prepare angle file
    if not os.path.exists(angle_file):
        with open(angle_file, "w") as f:
            f.write("Image,Angles\n")  # Add header to the file

    # Get all image files in the directory
    image_files = [
        f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ]

    for img_file in image_files:
        image_path = os.path.join(input_dir, img_file)
        print(f"Processing {img_file}...")
        process_image(image_path, img_file)


if __name__ == "__main__":
    main()