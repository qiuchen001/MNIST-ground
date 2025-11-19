import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid, save_image
import argparse
import mlflow
import mlflow.pytorch
from lightning.pytorch.callbacks import Callback
import lightning as L
from lightning.pytorch.loggers import MLFlowLogger

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

class LitConvNet(L.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = ConvNet()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        preds = outputs.argmax(dim=1)
        acc = (preds == labels).float().mean()
        self.log('val/accuracy', acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

def log_images(loader, out_dir, num_images=16, epoch=0):
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
        os.makedirs(out_dir, exist_ok=True)
        img_path = os.path.join(out_dir, f"mnist_preview_epoch_{epoch}.png")
    save_image(grid, img_path)
        # Artifact logging moved into MLflow callback to avoid run conflicts

def get_args():
    p = argparse.ArgumentParser(description='MNIST Lightning + MLflow')
    p.add_argument('--data-dir', type=str, default='/data/mnist')
    p.add_argument('--output-dir', type=str, default='/data/checkpoints')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--num-epochs', type=int, default=10)
    p.add_argument('--experiment-name', type=str, default='Volcano-CNN-Lightning')
    return p.parse_args()

if __name__ == '__main__':
    args = get_args()
    L.seed_everything(42, workers=True)
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    tracking_root = os.path.abspath(os.path.join(args.output_dir, 'mlruns'))
    mlflow.set_tracking_uri('file://' + tracking_root)
    logger = MLFlowLogger(experiment_name=args.experiment_name, tracking_uri='file://' + tracking_root)
    dataset = MNIST(args.data_dir, train=True, download=True, transform=ToTensor())
    train_ds, val_ds = random_split(dataset, [55000, 5000])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    lit = LitConvNet(lr=args.lr)
    class MLflowArtifactsCallback(Callback):
        def __init__(self, train_loader, output_dir):
            super().__init__()
            self.train_loader = train_loader
            self.output_dir = output_dir

        def on_fit_start(self, trainer, pl_module):
            run_id = getattr(trainer.logger, 'run_id', None)
            if run_id is None:
                return
            img_dir = os.path.join(self.output_dir, 'lightning_images')
            log_images(self.train_loader, img_dir, 16, 0)
            from mlflow.tracking import MlflowClient
            client = MlflowClient(tracking_uri=trainer.logger._tracking_uri)
            img_path = os.path.join(img_dir, 'mnist_preview_epoch_0.png')
            if os.path.isfile(img_path):
                client.log_artifact(run_id, img_path, artifact_path='images')

        def on_fit_end(self, trainer, pl_module):
            run_id = getattr(trainer.logger, 'run_id', None)
            if run_id is None:
                return
            save_path = os.path.join(self.output_dir, 'latest_checkpoint.pth')
            torch.save(pl_module.model.state_dict(), save_path)
            from mlflow.tracking import MlflowClient
            client = MlflowClient(tracking_uri=trainer.logger._tracking_uri)
            client.log_artifact(run_id, save_path, artifact_path='checkpoints')

    callbacks = [MLflowArtifactsCallback(train_loader, args.output_dir)]
    trainer = L.Trainer(max_epochs=args.num_epochs, logger=logger, accelerator='auto', devices='auto', callbacks=callbacks)
    logger.log_hyperparams({'lr': args.lr, 'batch_size': args.batch_size, 'num_epochs': args.num_epochs})
    trainer.fit(lit, train_loader, val_loader)