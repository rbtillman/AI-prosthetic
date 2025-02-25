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
import tensorflow as tf
import keras
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Lambda, Dense, InputLayer, Dropout, Flatten, Reshape, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint

try:
    # import feature map script if present
    from feature_map import feature_map
    visualize = True
    print("Feature map script will be run following training")
except:
    print("No feature map script found.")
    visualize = False

load_time = time.time() - start_time
print(f"\nLibraries imported in {load_time:.2f} seconds.\n")

# Constants
INPUT_SHAPE = (96, 96, 1)
BATCH_SIZE = 32
EPOCHS = 30
FINE_TUNE_EPOCHS = 15 #15
FINE_TUNE_PERCENTAGE = 65

# Paths to stuff
TRAIN_DIR = './CANS-REGMASKSPLIT/TRAIN'  
VALIDATION_DIR = './CANS-REGMASKSPLIT/TEST'  
LABELS_CSV = './CANS-REGMASKSPLIT/angles.csv'  
LEARNING_RATE = 0.0002 # static, consider scheduling?

# Load CSV, create a filename-to-angle dictionary
labels_df = pd.read_csv(LABELS_CSV, header=0, names=['filename', 'angle'])
# print(labels_df.columns)
label_dict = {row['filename']: row['angle'] for _, row in labels_df.iterrows()}

# print(label_dict)


## Changed to work w/ angles.csv
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
        img = tf.image.resize(img, (96, 96)) / 255.0  # normalize
        
        label = tf.numpy_function(get_label, [file_path], tf.float32)
        label.set_shape([])

        return img, label

    dataset = dataset.map(process_path).batch(batch_size)
    return dataset


train_dataset = load_dataset(TRAIN_DIR, BATCH_SIZE)
validation_dataset = load_dataset(VALIDATION_DIR, BATCH_SIZE)

input_layer = keras.Input(shape=INPUT_SHAPE, name='x_input')

# Base CNN block
x = Conv2D(64, (3, 3), activation='relu')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x) # from 128, similar performance
x = MaxPooling2D((2, 2))(x)

# Flatten and feed into dense layers
x = Flatten()(x)
x = Dense(64, activation='leaky_relu')(x)
x = Dense(16, activation='relu')(x)
x = Dropout(0.4)(x)
output_layer = Dense(1, activation='linear')(x)

model = Model(inputs=input_layer, outputs=output_layer)
print(model.summary())

# Compile the model
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss=MeanSquaredError(),
              metrics=['mae'])

# Callbacks
callbacks = [
    ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min'),
]

# Train the model
print("Training the model...")
model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS, verbose=2, callbacks=callbacks)

# Fine-tuning
print(f'Fine-tuning the model for {FINE_TUNE_EPOCHS} epochs...')
model = tf.keras.models.load_model('best_model.keras')

# Recompile for fine-tuning
model.compile(optimizer=Adam(learning_rate=0.000045),
              loss=MeanSquaredError(),
              metrics=['mae'])

# Continue training with fine-tuning
model.fit(train_dataset, epochs=FINE_TUNE_EPOCHS, verbose=2, validation_data=validation_dataset, callbacks=callbacks)

print("Model training and fine-tuning completed successfully.")

if visualize:
    feature_map(model, './CANS-REGMASKSPLIT/TEST/coke00003.jpg')
