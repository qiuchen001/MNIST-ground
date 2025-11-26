import mlflow
import logging
import os
from io import StringIO



mlflow.set_tracking_uri("http://localhost:5002")
# The set_experiment API creates a new experiment if it doesn't exist.
mlflow.set_experiment("Deep Learning Experiment New")

# IMPORTANT: Enable system metrics monitoring
mlflow.config.enable_system_metrics_logging()
mlflow.config.set_system_metrics_sampling_interval(1)


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


with mlflow.start_run() as run:
    logger.info("启动实时日志捕获...")
    logger.info("训练开始：Epoch 1")

    experiment_id = run.info.experiment_id
    logger.info(f"当前实验ID: {experiment_id}")
    print(f"当前实验ID: {experiment_id}")

    run_id = run.info.run_id
    logger.info(f"当前运行ID: {run_id}")

    job_name = os.getenv("VC_JOB_NAME") or "unknown-job"
    namespace = os.getenv("VC_NAMESPACE") or "default"
    pod_name = os.getenv("POD_NAME") or os.getenv("HOSTNAME") or ""
    pod_uid = os.getenv("POD_UID") or ""

    mlflow.set_tag("vc.job_name", job_name)
    mlflow.set_tag("vc.namespace", namespace)
    mlflow.set_tag("vc.pod_name", pod_name)
    mlflow.set_tag("vc.pod_uid", pod_uid)

    # Log training parameters
    mlflow.log_params(params)

    for epoch in range(params["epochs"]):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

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
                mlflow.log_metrics(
                    {"batch_loss": batch_loss, "batch_accuracy": batch_acc},
                    step=epoch * len(train_loader) + batch_idx,
                )

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

        # Log epoch metrics
        mlflow.log_metrics(
            {
                "train_loss": epoch_loss,
                "train_accuracy": epoch_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            },
            step=epoch,
        )
        # Log checkpoint at the end of each epoch
        mlflow.pytorch.log_model(model, name=f"checkpoint_{epoch}")

        print(
            f"Epoch {epoch+1}/{params['epochs']}, "
            f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

    # Log the final trained model
    model_info = mlflow.pytorch.log_model(model, name="final_model")


    import mlflow
    import numpy as np

    image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    mlflow.log_image(image, "dogs/step/3/image.png")
    mlflow.log_image(image, "dogs/step/4/image.png")
