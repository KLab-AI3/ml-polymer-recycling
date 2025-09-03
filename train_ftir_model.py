"""
Module: train_ftir_model
------------------------
This module trains the Figure2CNN model on preprocessed FTIR (Fourier-transform infrared spectroscopy) data.
It includes data loading, preprocessing, splitting into training and testing sets, and a training loop
with loss and accuracy tracking.

Workflow:
1. Load preprocessed FTIR data using `preprocess_ftir`.
2. Split the data into training and testing sets (80% train, 20% test).
3. Convert the data into PyTorch tensors and wrap them in DataLoaders.
4. Instantiate the Figure2CNN model and define the loss function and optimizer.
5. Train the model over a specified number of epochs, tracking loss and accuracy.
6. Evaluate the model on the test set and report accuracy.

Dependencies:
- os
- numpy
- torch
- sklearn.model_selection (for train-test split)
- preprocess_ftir (for data preprocessing)
- models.figure2_cnn (for the CNN model)

Usage:
Run this script directly to train the Figure2CNN model on the FTIR dataset.
The script outputs training loss and accuracy for each epoch, as well as test set accuracy.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
from torch import optim
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from preprocess_ftir import preprocess_ftir
from models.figure2_cnn import Figure2CNN

# Load preprocessed Raman data
data_dir = os.path.join("datasets", "ftir")
X, y = preprocess_ftir(data_dir)

# Print shape for confirmation
print(f"Total Samples: {X.shape[0]}")
print(f"Feature Shape per Sample: {X.shape[1]}")

# Split into 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,  # for reproductability
    stratify=y,  # balances label distribution (e.g.,0s & 1s)
)

# Confirm Results
print(f"Training Samples: {X_train.shape[0]}")
print(f"Test Samples: {X_test.shape[0]}")

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(
    1
)  # shape (N, 1, 500)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Wrap in TensorDataset
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Instantiate model
model = Figure2CNN(input_length=500)

# Set to evaluation or training mode as needed
model.train()

# Define loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backpropagation + optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Epoch summary
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total * 100

    print(
        f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%"
    )


# -------------------------------
# Step 6.1.5 — Evaluation on Test Set
# -------------------------------

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_acc = correct / total * 100
print(f"\nTest Set Accuracy: {test_acc:.2f}%")
