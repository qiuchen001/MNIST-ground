import torch
from torch import nn, optim, utils
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter

def get_device():
    try:
        use_mps = torch.backends.mps.is_available()
    except AttributeError:
        use_mps = False
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif use_mps:
        return torch.device("mps")
    return torch.device("cpu")

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
    def forward(self, x):
        return self.net(x)


def train_one_epoch(model, loader, optimizer, criterion, writer, device, epoch, global_step):
    model.train()
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        writer.add_scalar("train/loss", loss.item(), global_step)
        global_step += 1
    return global_step

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            preds = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return correct / total

def main():
    device = get_device()
    writer = SummaryWriter(log_dir="./tensorboard_logs")
    transform = ToTensor()
    train_ds = FakeData(size=6000, image_size=(1, 28, 28), num_classes=10, transform=transform)
    val_ds = FakeData(size=1000, image_size=(1, 28, 28), num_classes=10, transform=transform)
    train_loader = utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = utils.data.DataLoader(val_ds, batch_size=256, shuffle=False)
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    global_step = 0
    num_epochs = 3
    for epoch in range(1, num_epochs + 1):
        global_step = train_one_epoch(model, train_loader, optimizer, criterion, writer, device, epoch, global_step)
        acc = evaluate(model, val_loader, device)
        writer.add_scalar("val/accuracy", acc, epoch)

    samples, _ = next(iter(val_loader))
    writer.add_image("samples/input", samples[0], global_step=num_epochs, dataformats="CHW")

    writer.flush()
    writer.close()

if __name__ == "__main__":
    main()