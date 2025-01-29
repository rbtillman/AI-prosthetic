import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import patches

# Set the path for saving the output images
output_dir = 'CANS-MANMASK'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def main():
    input_directory = 'CANS-ALLIMAGES'  # input images
    process_images_in_directory(input_directory, output_dir)

# Function to manually mask an image by drawing a polygon
def manual_mask_image(image_path, output_dir):
    # Load the image
    image = cv2.imread(image_path)
    # Convert the image to RGB for displaying with matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a figure and axis for matplotlib
    fig, ax = plt.subplots()

    # Show the image
    ax.imshow(image_rgb)

    # Initialize the polygon for drawing the mask
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

    # Function to handle key press events (to move to the next image)
    def on_key(event):
        nonlocal mask_points

        # Check if Enter key was pressed
        if event.key == 'enter':
            if len(mask_points) > 2:
                plt.close()  # Close the current image and proceed
            else:
                print("Please draw a polygon before proceeding.")
        elif event.key == 'escape':
            mask_points = []

    # Connect the mouse click event to the onclick function
    fig.canvas.mpl_connect('button_press_event', onclick)
    # Connect the keyboard press event to the on_key function
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Wait for the user to finish clicking and press Enter to continue
    print("Click on the image to draw a polygon. Press Enter to finish.")
    plt.show()

    # Create a mask from the polygon points
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if len(mask_points) > 2:
        pts = np.array(mask_points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

    # Apply the mask to the image (only keep the masked area)
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Save the masked image to the output directory
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, masked_image)

    print(f"Saved masked image to {output_path}")

# Function to apply the manual masking to all images in a directory
def process_images_in_directory(input_dir, output_dir):
    # Loop through all images in the input directory
    for filename in os.listdir(input_dir):
        image_path = os.path.join(input_dir, filename)
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            print(f"Processing {filename}")
            manual_mask_image(image_path, output_dir)

if __name__ == "__main__":
    main()

