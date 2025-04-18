'''
Script for visualizing preprocessing layers without needing an existing model

no best practices in this script but it works.  deal with it

copyright (c) R. TIllman.  All Rights Reserved.  
'''
import tensorflow as tf
import numpy as np
from feature_map import feature_map
import keras
from tensorflow.keras import Model, Sequential, layers, activations
import tensorflow_addons as tfa
import cv2
import math


sobel_y = np.array([[[[-1]], [[-2]], [[-1]]], 
                     [[[0]], [[0]], [[0]]], 
                     [[[1]], [[2]], [[1]]]], dtype=np.float32).reshape(3, 3, 1, 1)

conv_y_layer = layers.Conv2D(1, (3, 3), padding="same", use_bias=False, kernel_initializer=tf.constant_initializer(sobel_y), name = 'convy', trainable=False)

@tf.keras.utils.register_keras_serializable()
class LocalContrastLayer(layers.Layer):
    def __init__(self, clipLimit=2.0, tileGridSize=(6, 6), epsilon=1e-5, **kwargs):
        super(LocalContrastLayer, self).__init__(**kwargs)
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize
        self.epsilon = epsilon

    def call(self, inputs):
        kernel_h, kernel_w = self.tileGridSize
        kernel = tf.ones((kernel_h, kernel_w, 1, 1), dtype=inputs.dtype) / (kernel_h * kernel_w)
        
        local_mean = tf.nn.conv2d(inputs, kernel, strides=[1, 1, 1, 1], padding='SAME')
        local_mean_sq = tf.nn.conv2d(tf.square(inputs), kernel, strides=[1, 1, 1, 1], padding='SAME')
        local_variance = local_mean_sq - tf.square(local_mean)
        local_std = tf.sqrt(tf.maximum(local_variance, 0.0) + self.epsilon)
        
        normalized = (inputs - local_mean) / local_std
        enhanced = 0.5 * (tf.tanh(normalized) + 1.0)
        return enhanced

    def get_config(self):
        config = super(LocalContrastLayer, self).get_config()
        config.update({
            "clipLimit": self.clipLimit,
            "tileGridSize": self.tileGridSize,
            "epsilon": self.epsilon
        })
        return config

@tf.keras.utils.register_keras_serializable()
class AffineAlignmentLayer(layers.Layer):
    def __init__(self, ideal_template, alignment_loss_weight=1.0, **kwargs):
        """
        ideal_template: A numpy array or tensor of shape (H, W, 1) representing the ideal can shape.
        alignment_loss_weight: Weight factor for the auxiliary alignment loss.
        """
        super(AffineAlignmentLayer, self).__init__(**kwargs)
        self.ideal_template = tf.constant(ideal_template, dtype=tf.float32)
        self.alignment_loss_weight = alignment_loss_weight

    def build(self, input_shape):
        # Get image dimensions from input shape.
        self.img_height = input_shape[1]
        self.img_width = input_shape[2]
        # Use GlobalAveragePooling to extract summary features from the input.
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        # Predict 4 parameters: [angle, scale, tx, ty]
        self.param_dense = tf.keras.layers.Dense(4, activation=None)
        super(AffineAlignmentLayer, self).build(input_shape)

    def call(self, inputs):
        # Predict transformation parameters from the input.
        x = self.gap(inputs)                # (batch, channels)
        params = self.param_dense(x)          # (batch, 4)

        # Split predicted parameters.
        theta_raw = params[:, 0]  # raw angle parameter
        scale_raw = params[:, 1]  # raw scale parameter
        tx_raw    = params[:, 2]  # raw translation in x
        ty_raw    = params[:, 3]  # raw translation in y

        # Process parameters:
        # - Constrain angle to [-pi/2, pi/2] via tanh.
        theta = tf.tanh(theta_raw) * (math.pi / 2)
        # - Use an exponential to ensure positive scale (initially near 1).
        scale = tf.exp(scale_raw)
        # - Constrain translations with tanh scaled to half the image dimensions.
        tx = tf.tanh(tx_raw) * (self.img_width / 2.0)
        ty = tf.tanh(ty_raw) * (self.img_height / 2.0)

        # Compute components for the full affine transformation:
        a0 = scale * tf.cos(theta)
        a1 = -scale * tf.sin(theta)
        a2 = tx
        a3 = scale * tf.sin(theta)
        a4 = scale * tf.cos(theta)
        a5 = ty

        # Build the transformation vector as expected by tfa.image.transform:
        # The vector is [a0, a1, a2, a3, a4, a5, 0, 0] for each sample.
        transform_full = tf.stack(
            [a0, a1, a2, a3, a4, a5, tf.zeros_like(a0), tf.zeros_like(a0)], axis=1
        )

        # Apply the full affine transformation (rotation, scaling, translation).
        transformed = tfa.image.transform(inputs, transform_full, interpolation='BILINEAR')

        # Now create a "rotation-only" transformation:
        # For rotation-only, fix scale = 1 and tx = ty = 0.
        a0_r = tf.cos(theta)
        a1_r = -tf.sin(theta)
        a2_r = tf.zeros_like(theta)
        a3_r = tf.sin(theta)
        a4_r = tf.cos(theta)
        a5_r = tf.zeros_like(theta)
        transform_rot = tf.stack(
            [a0_r, a1_r, a2_r, a3_r, a4_r, a5_r, tf.zeros_like(theta), tf.zeros_like(theta)], axis=1
        )

        # Apply the rotation-only transformation.
        rotated_only = tfa.image.transform(inputs, transform_rot, interpolation='BILINEAR')

        # Compute the dot product between each rotated image and the ideal template.
        # The ideal_template must have the same dimensions as the input (e.g., 96x96x1).
        dot_products = tf.reduce_sum(rotated_only * self.ideal_template, axis=[1, 2, 3])
        mean_dot = tf.reduce_mean(dot_products)
        # The auxiliary loss is defined as the negative dot product (to maximize similarity).
        alignment_loss = -self.alignment_loss_weight * mean_dot
        self.add_loss(alignment_loss)
        self.add_metric(mean_dot, name='mean_dot_product', aggregation='mean')

        return transformed, theta

    def get_config(self):
        config = super(AffineAlignmentLayer, self).get_config()
        config.update({
            'alignment_loss_weight': self.alignment_loss_weight,
            # Note: ideal_template is not serialized in this example.
        })
        return config


input_layer = keras.Input(shape=(96,96,1), name='x_input')
x = layers.Identity()(input_layer)

# Put whatever between here

CLAHE = LocalContrastLayer(clipLimit=4.0, tileGridSize=(5, 5))(x)

x = layers.Concatenate()([x,CLAHE])

# and here
x = layers.Flatten()(x)
output_layer = layers.Dense(1, activation='linear')(x)

model = Model(inputs=input_layer, outputs=output_layer)
print(model.summary())

feature_map(model, './CANS-ALLREG2/2monster00045.jpg')