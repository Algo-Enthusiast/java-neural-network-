import os
import json
import dlib
import cv2
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
import zipfile

# Initialize the dlib face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('datasets/face_recognition/shape_predictor_68_face_landmarks.dat')

# Load FFHQ images from the extracted dataset
def load_ffhq_images(image_folder, limit=None):
    images = []
    for idx, filename in enumerate(os.listdir(image_folder)):
        if limit and idx >= limit:
            break
        if filename.endswith(".png") or filename.endswith(".jpg"):
            images.append(os.path.join(image_folder, filename))
    return images

# Path to the FFHQ images folder
image_folder = data_folder
images = load_ffhq_images(image_folder, limit=1000)  # Change limit as needed

# Function to detect facial landmarks
def detect_landmarks(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    landmarks = []

    for face in faces:
        shape = predictor(gray, face)
        landmarks.append([(p.x, p.y) for p in shape.parts()])

    return np.array(landmarks, dtype=np.float32)

# Apply detection to all images
all_landmarks = []
image_paths = []

for image_path in images:
    landmarks = detect_landmarks(image_path)
    if len(landmarks) > 0:
        all_landmarks.append(landmarks)
        image_paths.append(image_path)

# Split the data into training and testing sets
train_paths, test_paths, train_landmarks, test_landmarks = train_test_split(
    image_paths, all_landmarks, test_size=0.2, random_state=42
)

# Save image paths and landmarks to separate files
def save_data(file_name, data):
    with open(file_name, 'w') as f:
        for item in data:
            f.write(f"{item}\n")

# Save the training data
save_data('datasets/face_recognition/trainingData.txt', train_paths)
save_data('datasets/face_recognition/trainingLabels.txt', train_landmarks)

# Save the testing data
save_data('datasets/face_recognition/testingData.txt', test_paths)
save_data('datasets/face_recognition/testingLabels.txt', test_landmarks)

# Print a snippet of the training data to verify
print(train_paths[:5])
print(train_landmarks[:5])
print(test_paths[:5])
print(test_landmarks[:5])
