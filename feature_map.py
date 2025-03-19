"""
Creates feature maps for some CNN model

Usage: python feature_map.py <path/to/model> <path/to/image> <mode>

Copyright (c) 2025 Tillman. All Rights Reserved.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from image_regression import LocalContrastLayer # Needed to load models w/ LCL. Probably a better way to handle this idk

def main():
    import sys
    # CHANGE: added <mode> argument
    if len(sys.argv) < 4:
        print("Usage: python feature_map.py <path/to/model> <path/to/image> <mode> <path/to/labels")
        print("  <path/to/model>: Path to the saved model file (use \ on Windows)")
        print("  <path/to/image>: Path to the image file to process  (use \ on Windows)")
        print("  <mode>: 'map' to visualize feature maps or 'predict' to print predictions")
        print("  <path/to/labels>: For predict only, path to labels file. Default option available")
        sys.exit(1)
    else:
        model_path = sys.argv[1]
        img_path = sys.argv[2]
        mode = sys.argv[3].lower()
    
    model = tf.keras.models.load_model(model_path)
    print(model.summary())

    if mode == "map":
        feature_map(model,img_path)
    elif mode == "predict":
        try:
            predict(model,img_path, sys.argv[4])
        except:
            predict(model,img_path)

def load_image(image_path, image_size = (96, 96)):
    """
    Loads and lightly preprocesses images 
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=1)
    img = tf.image.resize(img, image_size) / 255.0 # add a dynamic change for different sizes
    img = tf.expand_dims(img, axis=0)
    return img

# Predicts angle and returns stuff
def predict(model, image_path, labels_csv_path = './CANS-ALLALLREG/angles.csv', verbose = True):
    """
    Predict the angle of a can (or whatever!) in image
    
    Expected CSV format: header with columns 'filename','angle'.
    """
    img = load_image(image_path)
    
    prediction = model.predict(img)
    predicted_angle = prediction[0][0]  # assuming model output shape is (1,1)
    
    filename = os.path.basename(image_path)
    
    # Read the CSV file to find the actual angle for this filename
    labels_df = pd.read_csv(labels_csv_path, header=0, names=['filename', 'angle'])
    actual_angles = labels_df.loc[labels_df['filename'] == filename, 'angle'].values
    if len(actual_angles) == 0:
        print("Predict: WARN: No actual angle found for", filename)
        return
    actual_angle = actual_angles[0]
    
    error = abs(predicted_angle - actual_angle)
    
    if verbose:
        print("Predict: Predicted Angle: {:.2f} degrees".format(predicted_angle))
        print("Predict: Actual Angle: {:.2f} degrees".format(actual_angle))
        print("Predict: Absolute Error: {:.2f} degrees".format(error))

    data = [predicted_angle,actual_angle,error]

    # print(f"\ndata\n = {data}")
    return data

def feature_map(model, image_path, layer_indices=None):
    """
    Visualizes feature maps.  

    Adjust conv_layers with isinstance() for types of layers to show.
    Or, provide optional layer indices.
    """
    img = load_image(image_path)
    print(img.shape)

    # predict angle
    prediction = model.predict(img)
    predicted_angle = prediction[0][0]
    print(f"feature_map: Predicted Angle: {predicted_angle:.2f} degrees  (FMap)")

    conv_layers = [layer for layer in model.layers if ( isinstance(layer,tf.keras.layers.Add)
                                                       or isinstance(layer,tf.keras.layers.ReLU) 
                                                       or isinstance(layer,tf.keras.layers.Concatenate))] # adjust for what you want to see
    
    # If layer_indices is provided, select specific layers
    if layer_indices:
        conv_layers = [conv_layers[i] for i in layer_indices if i < len(conv_layers)]
    
    # Create a model that outputs feature maps
    outputs = [layer.output for layer in conv_layers]
    feature_model = tf.keras.Model(inputs=model.inputs, outputs=outputs)

    feature_maps = feature_model.predict(img)

    # Plot feature maps for each selected layer
    for layer_num, fmap in enumerate(feature_maps):
        num_filters = fmap.shape[-1]  # Get the number of filters in the layer
        
        cols = (16 if num_filters > 65 else 8)
        rows = np.ceil(num_filters / cols).astype(int)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes = axes.flatten()

        # Bug fix: check if feature map is 4D (batch, height, width, channels)
        if len(fmap.shape) == 4:
            fmap = fmap[0]  # Remove batch dimension if present

        for i in range(num_filters):
            axes[i].imshow(fmap[:, :, i], cmap='viridis')
            # axes[i].imshow(fmap[0, :, :, i], cmap='viridis') # old version
            axes[i].axis('off')

        for i in range(num_filters, len(axes)):
            axes[i].axis('off')

        fig.suptitle(f"Layer {layer_num + 1}: {conv_layers[layer_num].name}", fontsize=16)
        plt.show()


if __name__ == '__main__':
    main()