import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Step 1: Load images and extract features
def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    image = cv2.resize(image, (100, 100))  # You can adjust the resize value to change its resolution
    return image.flatten()  # Convert to 1D feature vector
 
# Define categories
categories = ["Pistol", "Shotgun", "MachineGun", "RPG", "Sniper", "Other"]
dataset = []
labels = []

# Path where gun images are stored (ensure subfolders are named after categories)
base_path = "C:/Users/Damien/OneDrive/Desktop/AIMKNN/Gun Classification KNN/gun_images"
for category in categories:
    folder_path = os.path.join(base_path, category)
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            dataset.append(extract_features(image_path))
            labels.append(category)

# Step 2: Train KNN model
X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3) # You can adjust this value to change its efficiency
knn.fit(X_train, y_train)

# Step 3: Predict a new image
def predict_gun_category(image_path):
    feature = extract_features(image_path).reshape(1, -1)
    prediction = knn.predict(feature)
    return prediction[0]

# Example: Predict a new image
test_image = "C:/Users/Damien/OneDrive/Desktop/AIMKNN/Gun Classification KNN/testing_images/test2.png"
print(f"Predicted Category: {predict_gun_category(test_image)}")