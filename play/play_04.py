import os
import sys
from io import StringIO

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and prepare data
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
train_dataset = datasets.FashionMNIST(
    "data", train=True, download=True, transform=transform
)
test_dataset = datasets.FashionMNIST("data", train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)


import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)


# Training parameters
params = {
    "epochs": 5,
    "learning_rate": 1e-3,
    "batch_size": 64,
    "optimizer": "SGD",
    "model_type": "MLP",
    "hidden_units": [512, 512],
}

# Define optimizer and loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=params["learning_rate"])



print(f"开始训练...")
print("*" * 50)

job_name = os.getenv("VC_JOB_NAME") or "unknown-job"
namespace = os.getenv("VC_NAMESPACE") or "default"
pod_name = os.getenv("POD_NAME") or os.getenv("HOSTNAME") or ""
pod_uid = os.getenv("POD_UID") or ""


for epoch in range(params["epochs"]):
    model.train()
    train_loss, correct, total = 0.0, 0, 0
    
    print(f"Epoch {epoch+1}/{params['epochs']}")
    print("*" * 50)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Calculate metrics
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # Log batch metrics (every 100 batches)
        if batch_idx % 100 == 0:
            batch_loss = train_loss / (batch_idx + 1)
            batch_acc = 100.0 * correct / total

    # Calculate epoch metrics
    epoch_loss = train_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)

            val_loss += loss.item()
            _, predicted = output.max(1)
            val_total += target.size(0)
            val_correct += predicted.eq(target).sum().item()

    # Calculate and log epoch validation metrics
    val_loss = val_loss / len(test_loader)
    val_acc = 100.0 * val_correct / val_total

    print(
        f"Epoch {epoch+1}/{params['epochs']}, "
        f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
    )

import numpy as np
image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
