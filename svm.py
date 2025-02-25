import numpy as np
import cv2
import streamlit as st
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
from PIL import Image

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Use a balanced dataset (for speed & accuracy)
X_train, y_train = X_train[:20000], y_train[:20000]
X_test, y_test = X_test[:5000], y_test[:5000]

# CIFAR-10 class names
class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"
]

# Efficient HOG feature extraction
def extract_features(image):
    image = cv2.resize(image, (32, 32))  # Resize to smaller dimensions
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    features = hog(gray, orientations=9, pixels_per_cell=(4, 4), 
                   cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    return features

# Convert images to HOG features (vectorized operation for speed)
X_train_hog = np.array([extract_features(img) for img in X_train])
X_test_hog = np.array([extract_features(img) for img in X_test])

# Normalize features
scaler = StandardScaler()
X_train_hog = scaler.fit_transform(X_train_hog)
X_test_hog = scaler.transform(X_test_hog)

# Apply PCA to reduce dimensions while keeping 98% variance
pca = PCA(n_components=0.98, svd_solver='full')
X_train_pca = pca.fit_transform(X_train_hog)
X_test_pca = pca.transform(X_test_hog)

# Train SVM classifier with optimized settings
svm = SVC(kernel='rbf', C=10, gamma='scale')  # RBF kernel for better accuracy
svm.fit(X_train_pca, y_train.ravel())

# Predict and evaluate
y_pred = svm.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit UI
st.title("🔥 CIFAR-10 Image Classifier using SVM 🔥")
st.write(f"**Model Accuracy:** {accuracy * 100:.2f}%")

# Option to upload an image
uploaded_file = st.file_uploader("Upload an image for prediction:", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Read the image
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Convert to features & predict
    features = extract_features(image).reshape(1, -1)
    features = scaler.transform(features)  # Normalize
    features_pca = pca.transform(features)  # Apply PCA
    prediction = svm.predict(features_pca)[0]

    # Display prediction
    st.image(image, caption=f"Predicted: **{class_names[prediction]}**", use_column_width=True)

# Option to test with a random image from CIFAR-10
if st.button("Test with a Random CIFAR-10 Image"):
    import random
    index = random.randint(0, len(X_test) - 1)
    
    image = X_test[index]
    features = extract_features(image).reshape(1, -1)
    features = scaler.transform(features)  # Normalize
    features_pca = pca.transform(features)  # Apply PCA
    prediction = svm.predict(features_pca)[0]

    # Display prediction
    st.image(image, caption=f"Predicted: **{class_names[prediction]}**", use_column_width=True)
