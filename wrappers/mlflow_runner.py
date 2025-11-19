import os
import re
import sys
import subprocess
import argparse
import mlflow

loss_re = re.compile(r"^Epoch \[(\d+)/(\d+)\], Iteration \[(\d+)/(\d+)\], Loss: ([0-9]*\.?[0-9]+)")
acc_re = re.compile(r"^Epoch (\d+), Test Accuracy: ([0-9]*\.?[0-9]+)")
save_re = re.compile(r"^Model saved to (.+)$")

def get_args():
    p = argparse.ArgumentParser(description="Run train.py and log to MLflow without modifying it")
    p.add_argument('--script', type=str, default='train.py')
    p.add_argument('--data-dir', type=str, default='/data/mnist')
    p.add_argument('--output-dir', type=str, default='/data/checkpoints')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--num-epochs', type=int, default=10)
    p.add_argument('--project-name', type=str, default='MNIST-example')
    p.add_argument('--experiment-name', type=str, default='Volcano-CNN')
    p.add_argument('--mlflow-experiment', type=str, default='Volcano-CNN-MLflow')
    return p.parse_args()

def main():
    args = get_args()
    tracking_root = os.path.abspath(os.path.join(args.output_dir, "mlruns"))
    mlflow.set_tracking_uri("file://" + tracking_root)
    mlflow.set_experiment(args.mlflow_experiment)
    cmd = [sys.executable, args.script,
           '--data-dir', args.data_dir,
           '--output-dir', args.output_dir,
           '--lr', str(args.lr),
           '--batch-size', str(args.batch_size),
           '--num-epochs', str(args.num_epochs),
           '--project-name', args.project_name,
           '--experiment-name', args.experiment_name]
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    with mlflow.start_run(run_name=args.mlflow_experiment):
        mlflow.log_params({
            "model": "ConvNet",
            "optim": "Adam",
            "lr": args.lr,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "data_dir": args.data_dir,
            "output_dir": args.output_dir,
            "delegate": "stdout-parser"
        })
        last_epoch = None
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in proc.stdout:
                line = line.rstrip() if line else ""
                if not line:
                    continue
                m = loss_re.match(line)
                if m:
                    epoch = int(m.group(1))
                    total_epochs = int(m.group(2))
                    iteration = int(m.group(3))
                    total_iters = int(m.group(4))
                    loss_val = float(m.group(5))
                    global_step = (epoch - 1) * total_iters + (iteration - 1)
                    mlflow.log_metric("train/loss", loss_val, step=global_step)
                    if last_epoch != epoch:
                        mlflow.log_metric("train/epoch", epoch, step=epoch)
                        last_epoch = epoch
                    continue
                m = acc_re.match(line)
                if m:
                    epoch = int(m.group(1))
                    acc = float(m.group(2))
                    mlflow.log_metric("val/accuracy", acc, step=epoch)
                    continue
                m = save_re.match(line)
                if m:
                    path = m.group(1).strip()
                    if os.path.isfile(path):
                        mlflow.log_artifact(path, artifact_path="checkpoints")
                    continue
            proc.wait()
        finally:
            if proc.poll() is None:
                proc.terminate()

if __name__ == '__main__':
    main()