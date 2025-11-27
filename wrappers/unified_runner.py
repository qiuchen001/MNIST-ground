import os
import sys
import argparse
import mlflow
import runpy
import types
from pathlib import Path

# Add repo root to path to import wrappers
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from wrappers.stream_logging import StdoutStderrInterceptor
from wrappers.tensorboard_mlflow_bridge import MLFWriter

def patch_tensorboard(tracking_uri, experiment, run_name, artifact_path):
    fake_tb = types.ModuleType("torch.utils.tensorboard")
    def factory(*args, **kwargs):
        # MLFWriter will use the active run started by unified_runner
        return MLFWriter(*args, tracking_uri=tracking_uri, experiment=experiment, run_name=run_name, artifact_path=artifact_path, **kwargs)
    fake_tb.SummaryWriter = factory
    sys.modules["torch.utils.tensorboard"] = fake_tb

def main():
    p = argparse.ArgumentParser(description="Unified Runner: Syncs TensorBoard and Stdout/Stderr to MLflow")
    p.add_argument("script", type=str, help="Path to the python script to run")
    p.add_argument("--tracking-uri", type=str, default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5002"))
    p.add_argument("--experiment", type=str, default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "Unified-Experiment"))
    p.add_argument("--run-name", type=str, default=os.environ.get("MLFLOW_RUN_NAME", "Unified-Run"))
    p.add_argument("--artifact-path", type=str, default="tensorboard_logs")
    
    # Parse known args to separate runner args from script args
    args, script_args = p.parse_known_args()
    
    # Setup MLflow
    mlflow.set_tracking_uri(args.tracking_uri)
    
    # Handle deleted experiment
    try:
        mlflow.set_experiment(args.experiment)
    except Exception:
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            exp = client.get_experiment_by_name(args.experiment)
            if exp and getattr(exp, 'lifecycle_stage', '') == 'deleted':
                client.restore_experiment(exp.experiment_id)
                mlflow.set_experiment(args.experiment)
            else:
                raise
        except Exception:
            print(f"Warning: Could not set or restore experiment {args.experiment}")

    # Start MLflow Run
    mlflow.enable_system_metrics_logging()
    with mlflow.start_run(run_name=args.run_name) as run:
        print(f"Started MLflow Run: {run.info.run_id}")
        
        # Patch TensorBoard
        patch_tensorboard(args.tracking_uri, args.experiment, args.run_name, args.artifact_path)
        
        # Setup Stdout/Stderr Interceptor
        # We use attach_logging=True to capture logging module output as well
        with StdoutStderrInterceptor(base_path="logs/structured", attach_logging=True):
            # Prepare sys.argv for the target script
            sys.argv = [args.script] + script_args
            
            # Execute script
            print(f"Executing script: {args.script}")
            try:
                runpy.run_path(args.script, run_name="__main__")
            except SystemExit as e:
                if e.code != 0:
                    print(f"Script exited with code {e.code}")
                    sys.exit(e.code)
            except Exception as e:
                print(f"Script failed with error: {e}")
                raise

if __name__ == "__main__":
    main()
