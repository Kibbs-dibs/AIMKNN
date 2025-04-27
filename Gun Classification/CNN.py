import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os

# Step 1: Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalization for better performance
])

# Step 2: Load Dataset
dataset_path = "C:/Users/Deme/OneDrive/Desktop/Python Projects/Gun Classification KNN/gun_images/"
train_set = datasets.ImageFolder(root=dataset_path, transform=transform)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

# Step 3: Define CNN Model
class GunClassifierCNN(nn.Module):
    def __init__(self, num_classes):
        super(GunClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model
num_classes = len(train_set.classes)
model = GunClassifierCNN(num_classes)

# Step 4: Define Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 5: Train the CNN Model
num_epochs = 2
for epoch in range(num_epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Step 6: Predict a New Image
from PIL import Image

def predict_gun_category(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Apply same preprocessing as training
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        predicted_class = torch.argmax(outputs).item()
    
    return train_set.classes[predicted_class]

# Example: Predict a new image
test_image = "C:/Users/Deme/OneDrive/Desktop/Python Projects/Gun Classification KNN/testing_images/test1.jpg"
print(f"Predicted Category: {predict_gun_category(test_image)}")