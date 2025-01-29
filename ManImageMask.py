import cv2
import numpy as np
import os
import math

# Variables
points = []  # Store points for the polygon
current_image = None
output_dir = "CANS-MASKED"  # Directory to save extracted cans
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


def find_longest_sides(points):
    """Find the two longest sides of a polygon."""
    sides = []
    for i in range(len(points)):
        pt1 = points[i]
        pt2 = points[(i + 1) % len(points)]  # Wrap around to the first point
        length = math.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
        sides.append((length, pt1, pt2))
    sides.sort(reverse=True, key=lambda x: x[0])  # Sort by length (descending)
    return sides[:2]  # Return the two longest sides


def draw_polygon(event, x, y, flags, param):
    global points, current_image, original_image, image_name
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))  # Add point on left click
        cv2.circle(current_image, (x, y), 5, (0, 255, 0), -1)  # Draw a small circle at the point
        if len(points) > 1:
            cv2.line(current_image, points[-2], points[-1], (255, 0, 0), 2)  # Draw lines between points
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(points) > 2:
            # Close the polygon
            cv2.line(current_image, points[-1], points[0], (255, 0, 0), 2)
            mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(points)], 255)
            extracted = cv2.bitwise_and(original_image, original_image, mask=mask)
            cv2.imshow("Extracted ROI", extracted)

            # Save the extracted image
            output_file = os.path.join(output_dir, f"extracted_{image_name}")
            cv2.imwrite(output_file, extracted)
            print(f"Saved extracted ROI to {output_file}")

            # Find the two longest sides and calculate their angles
            longest_sides = find_longest_sides(points)
            angles = [calculate_angle(pt1, pt2) for _, pt1, pt2 in longest_sides]
            average_angle = sum(angles) / len(angles)

            # Save the angle to the file
            with open(angle_file, "a") as f:
                f.write(f"{image_name},{average_angle:.2f}\n")
            print(f"Saved angle for {image_name}: {average_angle:.2f} degrees")

            points.clear()  # Reset points for the next image


def process_directory(input_dir):
    global current_image, original_image, image_name

    # Create output and angle directories/files if they don't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(angle_file):
        with open(angle_file, "w") as f:
            f.write("Image,Average Angle\n")  # Add header to the file

    # Get all image files in the directory
    image_files = [
        f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ]

    for img_file in image_files:
        image_path = os.path.join(input_dir, img_file)
        image_name = img_file
        print(f"Processing {image_name}...")

        # Load the image
        original_image = cv2.imread(image_path)
        current_image = original_image.copy()

        # Display the image and set the callback
        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", draw_polygon)

        print("Instructions:")
        print("- Left-click to mark points.")
        print("- Right-click to finish tracing and extract the region.")
        print("- Press Esc to skip to the next image.")

        while True:
            cv2.imshow("Image", current_image)
            key = cv2.waitKey(1)
            if key == 27:  # Press Esc to skip to the next image
                print(f"Skipping {image_name}.")
                points.clear()  # Clear points for the next image
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()