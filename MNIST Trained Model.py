import os
import torch
from torch import nn
from torch.optim import Adam 
from torch.utils.data import DataLoader 
from torchvision import datasets
from torchvision.transforms import ToTensor 
from torchvision.transforms import transforms 
import torch.nn.functional as F 
import torch.optim as optim
from PIL import Image



transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='data', train=False, transform=transform, download=True)

batch_size = 64

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # First convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # Second convolutional layer
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, 10)          # Output layer for 10 classes (digits 0-9)

   ## def __len__ (self):
       ## pass 

      ##  data_dir = 'C:\Users\1294346\OneDrive - The Metropolitan Transportation Authority\Desktop\VisionAI\Virtual Wspce\data\MNIST'
       ## dataset = transform(data_dir)
  
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 64 * 7 * 7)  # Flatten feature maps
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model
model = SimpleCNN()
weight_path = 'model_weights.pth'

if os.path.exists(weight_path):
    print("Loading saved weights...")
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    print("Model loaded and set to evaluation mode.")
else: 
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='data', train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10  # Define the number of training epochs
    # Train the model if weights do not exist
    print("No saved weights found. Training model from scratch...")

    for epoch in range(num_epochs):

        model.train()  # Set model to training mode
        running_loss = 0.0

    for images, labels in train_loader:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), 'model_weights.pth')  # Save model weights


    

'''
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # Save model weights after training
    torch.save(model.state_dict(), weight_path)
    print("Model weights saved.")

'''


##model.load_state_dict(torch.load('model_weights.pth')) # Load model weights when rerunning program 


# can seperate the model into funtions 

# Set model to evaluation mode
model.eval()
correct = 0
total = 0

with torch.no_grad():  # No need to track gradients in evaluation
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the test set: {100 * correct / total:.2f}%')


# Load and preprocess a new image (make sure it’s 28x28 pixels and grayscale)
image = Image.open("three buss.jpg").convert('L')
transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
image = transform(image).unsqueeze(0)  # Add batch dimension

# Predict
model.eval()
with torch.no_grad():
    output = model(image)
    probabilities = F.softmax(output, dim=1)
    confidence, predicted = torch.max(probabilities, 1)
   ## _, predicted = torch.max(output.data, 1)
    print(f'Predicted digit: {predicted.item()} with confidence: {confidence.item():.4f} ')

      # Display confidence levels for each digit
    print("Confidence levels for each digit:")
    for i, prob in enumerate(probabilities[0]):
        print(f"Digit {i}: {prob.item():.4f}")