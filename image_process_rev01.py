import os
import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Load and Preprocess Images
def load_and_preprocess_images(data_dir, img_size=(224, 224)):
    images = []
    labels = []
    
    # Define categories (folders in your directory)
    categories = ['vertical', 'horizontal', 'nothing']
    
    # Loop through the directories
    for label in categories:
        folder_path = os.path.join(data_dir, label)
        for filename in os.listdir(folder_path):
            # Construct image path
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
            
            # Resize image
            img_resized = cv2.resize(img, img_size)
            
            # Normalize image to [0, 1]
            img_normalized = img_resized / 255.0
            
            # Convert to 3D array (needed for feature extraction)
            img_array = np.stack([img_normalized] * 3, axis=-1)  # Replicate grayscale to 3 channels
            
            images.append(img_array)
            labels.append(label)
    
    return np.array(images), np.array(labels)

# 2. Feature Extraction using MobileNetV2
# Load MobileNetV2 model without the top layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

def extract_features(images):
    features = feature_extractor.predict(images)
    return features.reshape(features.shape[0], -1)  # Flatten features for ML models

# 3. Prepare Data
# Define the paths for TRAIN and TEST directories
data_dir_train = 'CANS-DATA/TRAIN'
data_dir_test = 'CANS-DATA/TEST'

# Load and preprocess images from TRAIN and TEST directories
train_images, train_labels = load_and_preprocess_images(data_dir_train)
test_images, test_labels = load_and_preprocess_images(data_dir_test)

# Encode labels (categorical to numeric)
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# Extract features from images
train_features = extract_features(train_images)
test_features = extract_features(test_images)

# Split data into training and validation sets (optional)
X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels_encoded, test_size=0.2, random_state=42)

# Print the shape of the processed data
print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Test data shape: {test_features.shape}")

# 4. Train and Evaluate Model (RandomForest)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Initialize the classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Validate the model
y_val_pred = clf.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Test the model
y_test_pred = clf.predict(test_features)
test_accuracy = accuracy_score(test_labels_encoded, y_test_pred)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
