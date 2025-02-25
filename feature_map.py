"""
Creates feature maps for some CNN model

Usage: python feature_map.py <path/to/model> <path/to/image>

Copyright (c) 2025 Tillman. All Rights Reserved.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def main():
    import sys

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        img_path = sys.argv[2]
    else:
        print('Usage: python feature_map.py <path/to/model> <path/to/image>')
        exit()
    
    model = tf.keras.models.load_model(model_path)
    print(model.summary())
    feature_map(model,img_path)



# visualize feature maps
def feature_map(model, image_path, layer_indices=None):
    def load_image(image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=1)
        img = tf.image.resize(img, (96, 96)) / 255.0
        img = tf.expand_dims(img, axis=0)
        return img

    img = load_image(image_path)

    conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    
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

        for i in range(num_filters):
            axes[i].imshow(fmap[0, :, :, i], cmap='viridis')
            axes[i].axis('off')

        for i in range(num_filters, len(axes)):
            axes[i].axis('off')

        fig.suptitle(f"Layer {layer_num + 1}: {conv_layers[layer_num].name}", fontsize=16)
        plt.show()


if __name__ == '__main__':
    main()