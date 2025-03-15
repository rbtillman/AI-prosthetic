'''
WIP NN trainer for regression on can angle.  

Copyright (c) 2025 Tillman. All Rights Reserved.
'''

import time
start_time = time.time()

import math
import os
import glob
import pandas as pd
import numpy as np

# Attempt to avoid issues with some CLHAE functions. Since removed, may be unnecessary
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
import tensorflow as tf
tf.config.optimizer.set_jit(False)
# Leaving import tf alone might be ok

import keras
import csv
from tensorflow.keras import Model, Sequential, layers, activations
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.optimizers import Adam, Lamb, Lion
from tensorflow.keras.losses import MeanSquaredError, Huber
from tensorflow.keras.callbacks import ModelCheckpoint

try:
    # import feature map script if present
    from feature_map import feature_map
    visualize = True
    if visualize:
        print("Feature map script available.  See code to use automatically followinf training\n")
except:
    print("No feature map script found.\n")
    visualize = False

load_time = time.time() - start_time
print(f"\nLibraries imported in {load_time:.2f} seconds.\n")

## CONSTANTS
INPUT_SHAPE = (96, 96, 1)
BATCH_SIZE = 64 # change based on available gpu memory, increased may help stabilize gradients
EPOCHS = 30
FINE_TUNE_EPOCHS = 10
FINE_TUNE_PERCENTAGE = 65

# Paths to stuff
PARENT_DIR = './CANS-ALLALLSPLIT'
LEARNING_RATE = 4e-5 # started w 4e-4

CYCLICAL = True # Enable/disable cyclical learning rate (priority 1)
CYC_MODE = 'triangular2' # triangular, triangular2, or 

SCHEDULE = False # Enable/disable schedule (priority 2, will not run if CYCLICAL == True)
# else: reduceLRonPlateau

visualize = False # Change this if you want to run a feature map following training

RANDOM_SEED = 42 # "Nothing is random" -my dad.  ensures reprocucability during training

## END CONSTANTS

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

train_dir = os.path.join(PARENT_DIR, 'TRAIN')
validation_dir = os.path.join(PARENT_DIR, 'TEST')
labels_csv = os.path.join(PARENT_DIR, 'angles.csv')

# Load CSV, create a filename-to-angle dictionary
labels_df = pd.read_csv(labels_csv, header=0, names=['filename', 'angle'])
# print(labels_df.columns)
label_dict = {row['filename']: row['angle'] for _, row in labels_df.iterrows()}

# Cyclical LR feature, may help escape local minema for GD
class CyclicalLR(tf.keras.callbacks.Callback):
    def __init__(self, base_lr=1e-5, max_lr=6e-4, step_size=2000, mode='triangular'):
        super(CyclicalLR, self).__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.iterations = 0
        self.history = {}

    def clr(self):
        cycle = np.floor(1 + self.iterations / (2 * self.step_size))
        x = np.abs(self.iterations / self.step_size - 2 * cycle + 1)
        if self.mode == 'triangular':
            lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x))
        elif self.mode == 'triangular2':
            lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) / (2 ** (cycle - 1))
        elif self.mode == 'exp_range':
            gamma = 0.99994  # or tune as needed
            lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * (gamma ** self.iterations)
        return lr

    def on_train_begin(self, logs=None):
        logs = logs or {}
        self.iterations = 0
        # Attempt to update the learning rate:
        lr_attr = self.model.optimizer.learning_rate
        if hasattr(lr_attr, 'assign'):
            lr_attr.assign(self.base_lr)
        else:
            self.model.optimizer.learning_rate = self.base_lr

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.iterations += 1
        new_lr = self.clr()
        lr_attr = self.model.optimizer.learning_rate
        if hasattr(lr_attr, 'assign'):
            lr_attr.assign(new_lr)
        else:
            self.model.optimizer.learning_rate = new_lr
        self.history.setdefault('lr', []).append(new_lr)

# Warmup routine for gpu, works byt doesnt work
def warmup_gpu():
    # Create two random tensors
    a = tf.random.uniform((1000, 1000))
    b = tf.random.uniform((1000, 1000))
    # Run a dummy matrix multiplication a few times to warm up the GPU
    for _ in range(5):
        _ = tf.matmul(a, b)

# Changed to work w/ angles.csv
def get_label(file_path):
    filename = tf.strings.split(file_path, '/')[-1]
    angle = labels_df.loc[labels_df['filename'] == filename.numpy().decode(), 'angle'].values
    return tf.constant(angle[0] if len(angle) > 0 else 0.0, dtype=tf.float32)

def load_dataset(directory, batch_size):
    file_paths = glob.glob(os.path.join(directory, "*.jpg"))  # Adjust for image types

    dataset = tf.data.Dataset.from_tensor_slices(file_paths)

    def process_path(file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=1)
        img = tf.image.resize(img, (96, 96)) / 255.0
        
        label = tf.numpy_function(get_label, [file_path], tf.float32)
        label.set_shape([])

        return img, label

    dataset = dataset.map(process_path).batch(batch_size)
    return dataset

# logs history of model performance
def save_model_info(model, history, lr, epochs, metrics_csv_filename = 'metrics_history.csv', summary_csv_filename = 'model_history.csv'):

    # Determine trial number from the metrics CSV.
    trial_num = 1
    if os.path.exists(metrics_csv_filename):
        with open(metrics_csv_filename, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
            if rows:
                # If header exists, assume the first row contains "Trial" in the first column.
                if rows[0] and rows[0][0] == "Trial":
                    # The number of completed trials is (number of rows - 1).
                    trial_num = len(rows)
                else:
                    trial_num = len(rows) + 1
            else:
                trial_num = 1

    summary_list = []
    def append_to_summary(line):
        summary_list.append(line)
    model.summary(print_fn=append_to_summary)
    model_summary = "\n".join(summary_list)

    # Extract performance metrics from history.
    final_mae = history.history['mae'][-1]  # Last recorded MAE.
    final_loss = history.history['loss'][-1]  # Last recorded loss.
    val_mae = history.history.get('val_mae', [None])[-1]  # If validation was used.
    val_loss = history.history.get('val_loss', [None])[-1]  # If validation was used.
    optimizer = model.optimizer.__class__.__name__
    learning_rate = lr

    # Prepare row data for training metrics CSV.
    metrics_row = [
        trial_num,
        optimizer,
        learning_rate,
        epochs,
        final_loss,
        final_mae,
        val_loss,
        val_mae
    ]
    metrics_headers = ["Trial", "Optimizer", "Learning Rate", "Epochs", "Final Loss", "Final MAE", "Val Loss", "Val MAE"]

    # Prepare row data for model summary CSV.
    summary_row = [trial_num, model_summary]
    summary_headers = ["Trial", "Model Summary"]

    # Write training metrics to the metrics CSV file.
    metrics_file_exists = os.path.exists(metrics_csv_filename)
    with open(metrics_csv_filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not metrics_file_exists:
            writer.writerow(metrics_headers)
        writer.writerow(metrics_row)

    # Write model summary to the summary CSV file.
    summary_file_exists = os.path.exists(summary_csv_filename)
    with open(summary_csv_filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not summary_file_exists:
            writer.writerow(summary_headers)
        writer.writerow(summary_row)

    print(f"Model information saved to {metrics_csv_filename} and {summary_csv_filename}")

# Define a local contrast layer
@tf.keras.utils.register_keras_serializable() # ensures this layer doesnt give issues upon model reload
class LocalContrastLayer(layers.Layer):
    def __init__(self, clipLimit=2.0, tileGridSize=(8, 8), epsilon=1e-5, **kwargs):
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

# Define const image filters
class ConstantFilters:
    def __init__(self):
        # Define constant filter kernels as instance variables
        self.sobel_x = np.array([[[[-1]], [[0]], [[1]]],
                                  [[[-2]], [[0]], [[2]]],
                                  [[[-1]], [[0]], [[1]]]], dtype=np.float32).reshape(3, 3, 1, 1)
        
        self.sobel_y = np.array([[[[-1]], [[-2]], [[-1]]],
                                  [[[0]], [[0]], [[0]]],
                                  [[[1]], [[2]], [[1]]]], dtype=np.float32).reshape(3, 3, 1, 1)
        
        self.laplacian = np.array([[0, 1, 0],
                                   [1, -4, 1],
                                   [0, 1, 0]], dtype=np.float32).reshape(3, 3, 1, 1)
        
        # for experimentation
        self.wtfisthis = np.array([[2, 1, 2],
                                   [1, -4, 1],
                                   [2, 1, 2]], dtype=np.float32).reshape(3, 3, 1, 1)
        
        # Create the Gabor filter using a helper method
        self.gabor_filter = self._create_gabor_filter(frequency=2, theta=0, sigma=1.0, size=31)

    def _create_gabor_filter(self, frequency, theta, sigma=1.0, size=31):
        # Potential improvement that sucks
        def gabor_kernel(frequency, theta, sigma=1.0, size=5):
            """
            Generate a Gabor kernel.
            :param frequency: Frequency of the sinusoidal wave
            :param theta: Orientation angle of the wave
            :param sigma: Width of the Gaussian function
            :param size: Size of the kernel
            :return: Gabor filter kernel
            """
            x = np.linspace(-size // 2, size // 2, size)
            y = np.linspace(-size // 2, size // 2, size)
            X, Y = np.meshgrid(x, y)
            
            # Rotation matrix for the kernel
            X_rot = X * np.cos(theta) + Y * np.sin(theta)
            Y_rot = -X * np.sin(theta) + Y * np.cos(theta)
            
            # Gabor function
            gaussian = np.exp(-(X_rot**2 + Y_rot**2) / (2 * sigma**2))
            sinusoidal = np.cos(2 * np.pi * frequency * X_rot)
            
            gabor = gaussian * sinusoidal
            return gabor
        gabor = gabor_kernel(frequency=frequency, theta=theta, sigma=sigma, size=size)
        gabor = np.expand_dims(gabor, axis=-1)
        gabor = np.expand_dims(gabor, axis=-1)
        return gabor

    def get_layers(self):
        """
        Returns a dictionary of Conv2D layers initialized with the constant filters.
        """
        conv_x_layer = layers.Conv2D(
            1, (3, 3), padding="same", use_bias=False,
            kernel_initializer=tf.constant_initializer(self.sobel_x),
            name='convx', trainable=False
        )
        conv_y_layer = layers.Conv2D(
            1, (3, 3), padding="same", use_bias=False,
            kernel_initializer=tf.constant_initializer(self.sobel_y),
            name='convy', trainable=False
        )
        laplacian_layer = layers.Conv2D(
            1, (3, 3), padding="same", use_bias=False,
            kernel_initializer=tf.constant_initializer(self.laplacian),
            name='edge', trainable=False
        )
        gabor_layer = layers.Conv2D(
            1, (31, 31), padding="same",
            kernel_initializer=tf.constant_initializer(self.gabor_filter),
            name='gabor', trainable=False
        )
        wtf_layer = layers.Conv2D(
            1, (3, 3), padding="same",
            kernel_initializer=tf.constant_initializer(self.wtfisthis),
            name='wtf', trainable=False
        )
        return {
            "conv_x": conv_x_layer,
            "conv_y": conv_y_layer,
            "laplacian": laplacian_layer,
            "gabor": gabor_layer,
            "wtf": wtf_layer
        }

def main():

    if SCHEDULE:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            LEARNING_RATE,
            decay_steps = 200,  # Adjust based on dataset size
            decay_rate=0.96,  # Adjust the rate of decay
            staircase=True  # stepwise decay
        )

    warmup_gpu()
    train_dataset = load_dataset(train_dir, BATCH_SIZE)
    validation_dataset = load_dataset(validation_dir, BATCH_SIZE)


    ## MODEL STRUCTURE:
    input_layer = keras.Input(shape=INPUT_SHAPE, name='x_input')

    filters = ConstantFilters()
    filter_layers = filters.get_layers()

    conv_x = filter_layers["conv_x"](input_layer)
    conv_y = filter_layers["conv_y"](input_layer)
    edge = filter_layers["laplacian"](input_layer)
    # gabor = filter_layers["gabor"](input_layer)
    # wtf = filter_layers["wtf"](input_layer)
    LCL = LocalContrastLayer(tileGridSize=(8, 8))(input_layer)

    # Base CNN block
    x = layers.AutoContrast(value_range=(0,1))(input_layer)

    x = layers.Concatenate()([x, conv_x, conv_y, edge, LCL])

    x = layers.Conv2D(64, (3, 3), kernel_initializer="he_normal")(x)
    # x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=None, negative_slope=0.1, threshold=0.0)(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(32, (3, 3),kernel_initializer="he_normal")(x) # from 3,3, better
    x = layers.ReLU(max_value=None, negative_slope=0.1, threshold=0.0)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3))(x) # from 64, better performance
    x = layers.ReLU(max_value=None, negative_slope=0.1, threshold=0.0)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Flatten and feed into dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(64, kernel_regularizer=l2(0.001))(x) # kernel_regularizer=l2(0.001),
    x = layers.ReLU(max_value=None, negative_slope=0.1, threshold=0.0)(x)
    #x = Dense(16, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    output_layer = layers.Dense(1, activation='linear')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    print(model.summary())
    ## END MODEL STRUCTURE

    # Callbacks and pass LR to optimizer
    if CYCLICAL:
        optimizer = Lion(learning_rate=LEARNING_RATE)
        print('LR adjustment: cyclical')
        callbacks = [
            ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min'),
            CyclicalLR(base_lr=1e-5, max_lr=6e-4, step_size=2000, mode=CYC_MODE)
        ]
        # using different callbacks for fine-tune
        callbacksFT = [
            ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')
        ]
    elif SCHEDULE:
        optimizer = Lion(learning_rate=lr_schedule)
        print('LR adjustment: Schedule')
        callbacks = [
            ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')
        ]
        callbacksFT = callbacks
    else:
        optimizer = Lion(learning_rate=LEARNING_RATE)
        print('LR adjustment: Reduce on Plateau')
        callbacks = [
            ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min'),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        callbacksFT = callbacks

    # Compile the model with the selected optimizer
    model.compile(optimizer=optimizer,
                loss=Huber(),
                metrics=['mae'])

    # Train the model
    print("Training the model...")
    trainhist = model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS, verbose=2, callbacks=callbacks)

    # Fine-tuning
    print(f'Fine-tuning the model for {FINE_TUNE_EPOCHS} epochs...')
    model = tf.keras.models.load_model('best_model.keras')

    # Recompile for fine-tuning
    model.compile(optimizer=Lion(learning_rate=0.000045),
                loss= Huber(),
                metrics=['mae'])

    # Continue training with fine-tuning
    tunehist = model.fit(train_dataset, epochs=FINE_TUNE_EPOCHS, verbose=2, validation_data=validation_dataset, callbacks=callbacksFT)

    print("Model training and fine-tuning completed successfully.")

    if visualize:
        feature_map(model, './CANS-REGMASKSPLIT/TEST/coke00003.jpg')

    save_model_info(model,trainhist,LEARNING_RATE,EPOCHS)
    save_model_info(model,tunehist,LEARNING_RATE,FINE_TUNE_EPOCHS,"tune_history.csv", "tune_models.csv")

if __name__ == "__main__":
    main()