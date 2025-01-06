import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize pixel values to a range of 0 to 1
train_images = train_images.reshape(-1, 28 * 28).astype('float32') / 255.0
test_images = test_images.reshape(-1, 28 * 28).astype('float32') / 255.0

# One-hot encode labels
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

# Replace 0s with 0.001 in labels
train_labels[train_labels == 0] = 0.001
test_labels[test_labels == 0] = 0.001

# Data augmentation using tf.image
def augment_image(image):
    image = tf.reshape(image, [28, 28, 1])
    image = tf.image.resize_with_crop_or_pad(image, 30, 30)  # Add padding
    image = tf.image.random_crop(image, size=[28, 28, 1])
    image = tf.image.random_brightness(image, max_delta=0.3)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return tf.reshape(image, [28 * 28])

# Apply augmentation to all training images
augmented_images = np.array([augment_image(img).numpy() for img in train_images])
augmented_labels = train_labels

# Combine the original and augmented data
combined_images = np.vstack((train_images, augmented_images))
combined_labels = np.vstack((train_labels, augmented_labels))

# Function to convert numpy arrays to formatted double arrays with trailing commas
def format_array(arr, name, chunk_size=10000):
    total = len(arr)
    formatted = [f"{name} = [\n"]
    
    def format_chunk(start):
        chunk = []
        for i in range(start, min(start + chunk_size, total)):
            chunk.append(" {" + ", ".join(map(str, arr[i])) + "},\n")
            if (i + 1) % (total // 10) == 0:
                print(f"Progress ({name}): {((i + 1) / total) * 100:.2f}%")
        return "".join(chunk)

    with ThreadPoolExecutor(max_workers=13) as executor:
        futures = [executor.submit(format_chunk, start) for start in range(0, total, chunk_size)]
        for future in futures:
            formatted.append(future.result())
    
    formatted.append("};")
    return "".join(formatted)

# Format the data into the specified format
trainingData = format_array(combined_images, 'trainingData')
trainingLabels = format_array(combined_labels, 'trainingLabels')
testingData = format_array(test_images, 'testingData')

# Save the formatted data to text files
for file_name, data in zip(['datasets/digit_recognition/trainingData.txt', 'datasets/digit_recognition/trainingLabels.txt', 'datasets/digit_recognition/testingData.txt'],
                           [trainingData, trainingLabels, testingData]):
    with open(file_name, 'w') as f:
        f.write(data)

# Print a snippet of the formatted data to verify (limited to the first 500 characters)
print(trainingData[:500])
print(trainingLabels[:500])
print(testingData[:500])
