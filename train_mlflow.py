import os
import torch
from torch import nn, optim, utils
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid, save_image
import argparse
import mlflow

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        in_size = x.size(0)
        out = self.conv1(x)
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)
        out = self.conv2(out)
        out = F.relu(out)
        out = out.view(in_size, -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out

def log_images(loader, num_images=16, epoch=0):
    images_logged = 0
    batch_images = []
    for images, _ in loader:
        for i in range(images.shape[0]):
            if images_logged < num_images:
                batch_images.append(images[i])
                images_logged += 1
            else:
                break
        if images_logged >= num_images:
            break
    if batch_images:
        grid = make_grid(torch.stack(batch_images), nrow=4, normalize=True)
        tmp_dir = os.path.join("artifacts_tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        img_path = os.path.join(tmp_dir, f"mnist_preview_epoch_{epoch}.png")
        save_image(grid, img_path)
        mlflow.log_artifact(img_path, artifact_path="images")

def train(model, device, train_dataloader, optimizer, criterion, epoch, num_epochs):
    model.train()
    for iter, (inputs, labels) in enumerate(train_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if iter % 100 == 0:
            print('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs, iter + 1, len(train_dataloader), loss.item()))
            global_step = (epoch - 1) * len(train_dataloader) + iter
            mlflow.log_metric("train/loss", loss.item(), step=global_step)

def test(model, device, val_dataloader, epoch):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f'Epoch {epoch}, Test Accuracy: {accuracy:.4f}')
    mlflow.log_metric("val/accuracy", accuracy, step=epoch)
    return accuracy

def get_args():
    parser = argparse.ArgumentParser(description="Volcano MNIST Training (MLflow)")
    parser.add_argument('--data-dir', type=str, default='/data/mnist')
    parser.add_argument('--output-dir', type=str, default='/data/checkpoints')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--experiment-name', type=str, default='Volcano-CNN')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    try:
        use_mps = torch.backends.mps.is_available()
    except AttributeError:
        use_mps = False
    if torch.cuda.is_available():
        device = "cuda"
    elif use_mps:
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    tracking_root = os.path.abspath(os.path.join(args.output_dir, "mlruns"))
    mlflow.set_tracking_uri("file://" + tracking_root)
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=args.experiment_name):
        mlflow.log_params({
            "model": "ConvNet",
            "optim": "Adam",
            "lr": args.lr,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "device": device,
            "data_dir": args.data_dir,
            "output_dir": args.output_dir,
        })
        dataset = MNIST(args.data_dir, train=True, download=True, transform=ToTensor())
        train_dataset, val_dataset = utils.data.random_split(dataset, [55000, 5000])
        train_dataloader = utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader = utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)
        log_images(train_dataloader, 16, 0)
        model = ConvNet()
        model.to(torch.device(device))
        print(model)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        best_acc = 0.0
        for epoch in range(1, args.num_epochs + 1):
            mlflow.log_metric("train/epoch", epoch, step=epoch)
            train(model, device, train_dataloader, optimizer, criterion, epoch, args.num_epochs)
            if epoch % 2 == 0:
                acc = test(model, device, val_dataloader, epoch)
                if acc > best_acc:
                    best_acc = acc
        save_path = os.path.join(args.output_dir, 'latest_checkpoint.pth')
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        mlflow.log_artifact(save_path, artifact_path="checkpoints")