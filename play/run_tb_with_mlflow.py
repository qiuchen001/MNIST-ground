import os
import sys
import argparse
import importlib.util
import runpy
from pathlib import Path
import types

def load_and_patch_writer(module_path, tracking_uri, experiment, run_name, artifact_path):
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from wrappers.tensorboard_mlflow_bridge import MLFWriter
    fake_tb = types.ModuleType("torch.utils.tensorboard")
    def factory(*args, **kwargs):
        return MLFWriter(*args, tracking_uri=tracking_uri, experiment=experiment, run_name=run_name, artifact_path=artifact_path, **kwargs)
    fake_tb.SummaryWriter = factory
    sys.modules["torch.utils.tensorboard"] = fake_tb
    runpy.run_path(module_path, run_name="__main__")
    return True

def main():
    p = argparse.ArgumentParser(description="Run a TensorBoard script with MLflow bridge without modifying it")
    p.add_argument("script", type=str, help="Path to the original TB script")
    p.add_argument("--tracking-uri", type=str, default=os.environ.get("MLFLOW_TRACKING_URI", None))
    p.add_argument("--experiment", type=str, default="TensorBoard Sync")
    p.add_argument("--run-name", type=str, default="tb-bridge")
    p.add_argument("--artifact-path", type=str, default="tensorboard_logs")
    args = p.parse_args()

    load_and_patch_writer(args.script, args.tracking_uri, args.experiment, args.run_name, args.artifact_path)

if __name__ == "__main__":
    main()
