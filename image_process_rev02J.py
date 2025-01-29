import time
start_time = time.time()
import math
import requests
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, InputLayer, Dropout, Conv1D, Flatten, Reshape, MaxPooling1D, BatchNormalization,
    Conv2D, GlobalMaxPooling2D, Lambda, GlobalAveragePooling2D)
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from pathlib import Path
load_time = time.time() - start_time
print(f"\nlibraries import completed in{load_time}\n")
# Constants and paths
WEIGHTS_PATH = './transfer-learning-weights/edgeimpulse/MobileNetV2.0_35.96x96.grayscale.bsize_64.lr_0_005.epoch_260.val_loss_3.10.val_accuracy_0.35.hdf5'
ROOT_URL = 'https://cdn.edgeimpulse.com/'
INPUT_SHAPE = (96, 96, 1)
BATCH_SIZE = 32
EPOCHS = 80
LEARNING_RATE = 0.0004
FINE_TUNE_EPOCHS = 15
FINE_TUNE_PERCENTAGE = 65
TRAIN_DIR = './CANS-DATA/TRAIN'  # Update this with your train dataset directory
VALIDATION_DIR = './CANS-DATA/TEST'  # Update this with your validation dataset directory

# Download pretrained weights if not available
p = Path(WEIGHTS_PATH)
if not p.exists():
    print(f"Pretrained weights {WEIGHTS_PATH} unavailable; downloading...")
    if not p.parent.exists():
        p.parent.mkdir(parents=True)
    weights_data = requests.get(ROOT_URL + WEIGHTS_PATH[2:]).content
    with open(WEIGHTS_PATH, 'wb') as f:
        f.write(weights_data)
    print(f"Pretrained weights {WEIGHTS_PATH} downloaded successfully.")

# Load dataset
# train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
#     TRAIN_DIR,
#     image_size=INPUT_SHAPE[:2],
#     batch_size=BATCH_SIZE,
#     label_mode='categorical'
# )

# validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
#     VALIDATION_DIR,
#     image_size=INPUT_SHAPE[:2],
#     batch_size=BATCH_SIZE,
#     label_mode='categorical'
# )
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(96, 96),  # Resize images to 96x96
    batch_size=BATCH_SIZE,
    color_mode='grayscale',  # Ensure images are grayscale
    label_mode='categorical'
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    VALIDATION_DIR,
    image_size=(96, 96),  # Resize validation images to 96x96
    batch_size=BATCH_SIZE,
    color_mode='grayscale',  # Ensure validation images are grayscale
    label_mode='categorical'
)

# Load the base MobileNetV2 model with pretrained weights
base_model = tf.keras.applications.MobileNetV2(
    input_shape=INPUT_SHAPE, alpha=0.35, weights=WEIGHTS_PATH
)

base_model.trainable = False  # Freeze the base model

# Build the final model
model = Sequential()
model.add(InputLayer(input_shape=INPUT_SHAPE, name='x_input'))
last_layer_index = -3
model.add(Model(inputs=base_model.inputs, outputs=base_model.layers[last_layer_index].output))
model.add(Reshape((-1, model.layers[-1].output.shape[3])))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))  # 3 classes (vertical, horizontal, nothing)

# Compile the model
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss=categorical_crossentropy,
              metrics=['accuracy'])

# Callbacks (optional)
callbacks = [
    ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max'),
    #EarlyStopping(patience=3, restore_best_weights=True)
]

# Train the model
print("Training the model...")
model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS, verbose=2, callbacks=callbacks)

# Fine-tuning the model
print(f'Fine-tuning the model for {FINE_TUNE_EPOCHS} epochs...')
model = tf.keras.models.load_model('best_model.h5')

# Unfreeze the base model layers and freeze layers before the fine-tune layer
model_layer_count = len(model.layers)
fine_tune_from = math.ceil(model_layer_count * ((100 - FINE_TUNE_PERCENTAGE) / 100))

model.trainable = True
for layer in model.layers[:fine_tune_from]:
    layer.trainable = False

# Recompile the model after unfreezing
model.compile(optimizer=Adam(learning_rate=0.000045),
              loss=categorical_crossentropy,
              metrics=['accuracy'])

# Continue training with fine-tuning
model.fit(train_dataset, epochs=FINE_TUNE_EPOCHS, verbose=2, validation_data=validation_dataset, callbacks=callbacks)

print("Model training and fine-tuning completed successfully.")
