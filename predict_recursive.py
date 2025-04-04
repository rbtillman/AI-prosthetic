"""
predicts angle for multiple images using some model and saves to csv.

Usage: python predict_recursive.py <model_path> <images_dir> <labels_csv> [<output_csv>]

Written R. Tillman 3.19.2025
Copyright (c) 2025 Tillman. All Rights Reserved.
"""
import os
import sys
import csv
import tensorflow as tf
from feature_map import predict

def AutoPredict(model, images_dir, labels_csv_path, output_csv, verbose=False):
    """
    For each image in images_dir, use the model to predict the angle,
    then write the filename and error to output_csv. Sorts by error, highest first.
    """

    results = []
    # Loop over each image file in the directory.
    for filename in os.listdir(images_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            image_path = os.path.join(images_dir, filename)
            
            result = predict(model, image_path, labels_csv_path, verbose)
            
            if result is None:
                continue
            error = result[2]
            results.append((filename, error))
            print(f"AutoPredict: Processed {filename}: Error = {error:.2f} degrees")

    # Sort the results
    results.sort(key=lambda x: x[1], reverse=True)

    headers = ['filename', 'error']
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for filename, error in results:
            writer.writerow([filename, error])

def main():
    if len(sys.argv) < 4:
        print("Usage: python predict_directory.py <model_path> <images_dir> <labels_csv> [<output_csv>]")
        print("  <model_path>: Path to the saved model file (e.g., best_model.keras)")
        print("  <images_dir>: Directory containing images to predict")
        print("  <labels_csv>: CSV file with measured angles (columns: filename, angle)")
        print("  <output_csv>: (Optional) Output CSV file (default: errors.csv)")
        sys.exit(1)
    
    model_path = sys.argv[1]
    images_dir = sys.argv[2]
    labels_csv = sys.argv[3]
    output_csv = sys.argv[4] if len(sys.argv) > 4 else "errors.csv"

    model = tf.keras.models.load_model(model_path)
    print("MAIN: Model loaded successfully.")
    
    AutoPredict(model, images_dir, labels_csv, output_csv, True)
    print("MAIN: Prediction errors saved to", output_csv)

if __name__ == "__main__":
    main()
