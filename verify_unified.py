import mlflow
import os

mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5002'))
runs = mlflow.search_runs(experiment_names=["Unified-Experiment"], order_by=["start_time desc"], max_results=1)

for i, run in runs.iterrows():
    print(f"Run ID: {run.run_id}")
    try:
        client = mlflow.tracking.MlflowClient()
        local_path = client.download_artifacts(run.run_id, "logs/real_time_logs.log", dst_path=".")
        size = os.path.getsize(local_path)
        print(f"Log file size: {size} bytes")
        if size > 0:
            print("SUCCESS: Log file is not empty.")
            with open(local_path, "r") as f:
                print("Log content preview:")
                print(f.read()[:200])
        else:
            print("FAILURE: Log file is empty.")
    except Exception as e:
        print(f"Error checking log file: {e}")
