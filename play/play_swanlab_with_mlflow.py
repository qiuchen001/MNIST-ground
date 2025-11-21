import mlflow
import swanlab
import os

class SyncCallback:
    def __init__(self):
        self.step = 0

    def log_metrics(self, metrics):
        swanlab.log(metrics, step=self.step)
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=self.step)
        self.step += 1

# 使用示例
callback = SyncCallback()

mlflow.set_tracking_uri("http://localhost:5002")
# mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5002"))
mlflow.set_experiment("SwanLab-MLflow-Sync")
with mlflow.start_run(run_name="custom-sync-demo"):
    # settings = swanlab.Settings(metadata_collect=False, collect_hardware=False, collect_runtime=False, requirements_collect=False, conda_collect=False, hardware_monitor=False)
    # swanlab.init(project="my-project", experiment_name="custom-sync", mode="local", settings=settings)
    # swanlab.init(project="my-project", experiment_name="custom-sync",settings=settings)
    swanlab.init(project="my-project", experiment_name="custom-sync")
    for epoch in range(5):
        loss = 0.5 / (epoch + 1)
        accuracy = 0.6 + epoch * 0.1
        callback.log_metrics({"loss": loss, "accuracy": accuracy})
    swanlab.finish()