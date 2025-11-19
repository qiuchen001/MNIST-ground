import os
import torch
from torch import nn, optim, utils
import torch.nn.functional as F
import torchvision
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import swanlab
import argparse # 1. 导入 argparse

# CNN网络构建 (与您原来的一致)
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1,28x28
        self.conv1 = nn.Conv2d(1, 10, 5)  # 10, 24x24
        self.conv2 = nn.Conv2d(10, 20, 3)  # 128, 10x10
        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        in_size = x.size(0)
        out = self.conv1(x)  # 24
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)  # 12
        out = self.conv2(out)  # 10
        out = F.relu(out)
        out = out.view(in_size, -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out

# 捕获并可视化前20张图像 (与您原来的一致)
def log_images(loader, num_images=16):
    images_logged = 0
    logged_images = []
    for images, labels in loader:
        for i in range(images.shape[0]):
            if images_logged < num_images:
                logged_images.append(swanlab.Image(images[i], caption=f"Label: {labels[i]}"))
                images_logged += 1
            else:
                break
        if images_logged >= num_images:
            break
    swanlab.log({"MNIST-Preview": logged_images})
    

def train(model, device, train_dataloader, optimizer, criterion, epoch, num_epochs):
    model.train()
    for iter, (inputs, labels) in enumerate(train_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if iter % 100 == 0: # 减少打印频率
            print('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs, iter + 1, len(train_dataloader), loss.item()))
            swanlab.log({"train/loss": loss.item()}) # 调整log频率

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
    swanlab.log({"val/accuracy": accuracy}, step=epoch)

# 2. 定义参数解析
def get_args():
    parser = argparse.ArgumentParser(description="Volcano MNIST Training")
    
    # 路径参数 (!!重要!!)
    parser.add_argument('--data-dir', type=str, default='/data/mnist',
                        help='Path to MNIST dataset (will be downloaded here)')
    parser.add_argument('--output-dir', type=str, default='/data/checkpoints',
                        help='Path to save model checkpoints')
    
    # 超参数
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='Number of training epochs')
                        
    # SwanLab 参数
    parser.add_argument('--project-name', type=str, default='MNIST-example',
                        help='SwanLab project name')
    parser.add_argument('--experiment-name', type=str, default='Volcano-CNN',
                        help='SwanLab experiment name')
                        
    return parser.parse_args()

if __name__ == "__main__":
    
    # 3. 解析参数
    args = get_args()

    # 检测设备 (与您原来的一致)
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

    # 4. 使用 args 中的参数初始化 swanlab
    settings = swanlab.Settings(metadata_collect=False, collect_hardware=False, collect_runtime=False, requirements_collect=False, conda_collect=False, hardware_monitor=False)
    run = swanlab.init(
        project=args.project_name,
        experiment_name=args.experiment_name,
        config={
            "model": "ConvNet", # (您的模型是 ConvNet, 不是 ResNet18)
            "optim": "Adam",
            "lr": args.lr,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "device": device,
            "data_dir": args.data_dir,
            "output_dir": args.output_dir
        },
        mode="local",
        settings=settings
    )

    # 确保数据和输出目录存在
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # 5. 使用 args.data_dir 设置数据集路径
    dataset = MNIST(args.data_dir, train=True, download=True, transform=ToTensor())
    train_dataset, val_dataset = utils.data.random_split(dataset, [55000, 5000])

    train_dataloader = utils.data.DataLoader(train_dataset, batch_size=run.config.batch_size, shuffle=True)
    val_dataloader = utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # （可选）看一下数据集的前16张图像
    log_images(train_dataloader, 16)

    # 初始化模型
    model = ConvNet()
    model.to(torch.device(device))
    print(model)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=run.config.lr)

    # 开始训练和测试循环
    for epoch in range(1, run.config.num_epochs + 1):
        swanlab.log({"train/epoch": epoch}, step=epoch)
        train(model, device, train_dataloader, optimizer, criterion, epoch, run.config.num_epochs)
        if epoch % 2 == 0:  
            test(model, device, val_dataloader, epoch)

    # 6. 使用 args.output_dir 保存模型
    save_path = os.path.join(args.output_dir, 'latest_checkpoint.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
