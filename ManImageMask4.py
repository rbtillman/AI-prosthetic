import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.widgets import PolygonSelector

# Initialize variables
mask_points = []  # List to store points of the polygon
current_image_index = 0  # Index to keep track of the current image

# Directory paths
input_dir = 'CANS-ALLIMAGES'
output_dir = 'CANSMASK4'

# Function to handle mouse button events (right-click to save)
def on_button_press(event):
    global mask_points, current_image_index

    # Right-click to save the current polygon
    if event.button == 3:  # Right-click (button 3)
        if len(mask_points) > 2:  # Check if a polygon has been drawn
            print("Polygon drawn. Saving the image.")
            plt.close()  # Close the image and proceed
            save_masked_image()  # Save the current masked image
            current_image_index += 1  # Move to the next image
            if current_image_index < len(images):
                mask_points = []  # Reset points for next image
                display_image()  # Display the next image
            else:
                print("All images processed.")
                plt.close()

# Function to handle key events (Esc to clear polygon)
def on_key(event):
    global mask_points

    if event.key == 'escape':
        # Clear the points and reset the image for redrawing
        print("Polygon cleared. Start drawing again.")
        mask_points = []  # Clear the points
        plt.clf()  # Clear the current figure
        display_image()  # Redraw the image

# Function to display the image and set up the polygon tool
def display_image():
    global mask_points

    img = cv2.imread(images[current_image_index])
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    ax.set_title(f"Image {current_image_index + 1} / {len(images)}")
    polygon_selector = PolygonSelector(ax, onselect)

    # Connect mouse button press event (right-click to save)
    fig.canvas.mpl_connect('button_press_event', on_button_press)
    
    # Connect key press events for clearing (Esc key)
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()

# Function to save the masked image to the output directory
def save_masked_image():
    global mask_points

    img = cv2.imread(images[current_image_index])
    mask = np.zeros_like(img)

    # Create a polygon mask
    polygon = np.array(mask_points, np.int32)
    polygon = polygon.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [polygon], (255, 255, 255))  # White polygon mask

    # Mask the image
    masked_img = cv2.bitwise_and(img, mask)

    # Save the masked image
    output_path = os.path.join(output_dir, f"masked_image_{current_image_index + 1}.png")
    cv2.imwrite(output_path, masked_img)
    print(f"Saved masked image to {output_path}")

# Function to handle the selection of points for the polygon
def onselect(verts):
    global mask_points
    mask_points = verts  # Store the vertices of the polygon
    print(f"Polygon points: {mask_points}")

# Load all images from the directory
images = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Start processing the images
if len(images) > 0:
    display_image()
else:
    print("No images found in the specified directory.")
