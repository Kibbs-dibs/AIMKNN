#1. Load dataset
#2. Set up KNN
#3. Image classification
#4. Use KNN to classify the images according to Type
#5. Also separate into generation 

# Import required libraries
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# 1. Load MNIST dataset (handwritten digits)
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target

# 2. Preprocess data (normalize pixel values to [0, 1])
X = X / 255.0

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize KNN classifier
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')  # Try n_neighbors=3, 7, etc.

# 5. Train the model
knn.fit(X_train, y_train)

# 6. Predict on test data
y_pred = knn.predict(X_test)

# 7. Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# 8. Plot confusion matrix (optional)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 9. Test on a sample image (optional)
sample_idx = np.random.randint(0, len(X_test))
sample_img = X_test[sample_idx].reshape(28, 28)
plt.imshow(sample_img, cmap='gray')
plt.title(f"True: {y_test[sample_idx]}, Predicted: {y_pred[sample_idx]}")
plt.show()