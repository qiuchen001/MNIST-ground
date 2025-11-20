import os
import math
import mlflow
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

class MLFWriter(SummaryWriter):
    def __init__(self, *args, tracking_uri=None, experiment="TensorBoard Sync", run_name="tb-bridge", artifact_path="tensorboard_logs", **kwargs):
        self._artifact_path = artifact_path
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        self._run = mlflow.start_run(run_name=run_name)
        super().__init__(*args, **kwargs)

    def add_scalar(self, tag, scalar_value, global_step=None, *args, **kwargs):
        super().add_scalar(tag, scalar_value, global_step, *args, **kwargs)
        step = int(global_step) if global_step is not None else 0
        if mlflow.active_run() is not None and scalar_value is not None:
            v = float(scalar_value)
            if not math.isnan(v):
                mlflow.log_metric(tag, v, step=step)

    def add_image(self, tag, img_tensor, global_step=None, *args, **kwargs):
        super().add_image(tag, img_tensor, global_step, *args, **kwargs)
        step = int(global_step) if global_step is not None else 0
        dataformats = kwargs.get("dataformats", "CHW")
        if mlflow.active_run() is not None and img_tensor is not None:
            if isinstance(img_tensor, np.ndarray):
                arr = img_tensor
            else:
                try:
                    import torch
                    t = img_tensor.detach().cpu().numpy() if hasattr(img_tensor, "detach") else np.array(img_tensor)
                except Exception:
                    t = np.array(img_tensor)
                arr = t
            if dataformats.upper() == "CHW" and arr.ndim == 3:
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0.0, 1.0)
                arr = (arr * 255.0).astype(np.uint8)
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = arr.squeeze(2)
                img = Image.fromarray(arr, mode="L")
            elif arr.ndim == 2:
                img = Image.fromarray(arr, mode="L")
            else:
                img = Image.fromarray(arr, mode="RGB")
            try:
                mlflow.log_image(img, artifact_file=f"images/{tag}/{step}.png")
            except Exception:
                tmp = os.path.join(self.log_dir, f"{tag}_{step}.png")
                img.save(tmp)
                mlflow.log_artifact(tmp, artifact_path=f"images/{tag}")

    def close(self):
        try:
            super().close()
            if mlflow.active_run() is not None:
                mlflow.log_artifacts(self.log_dir, artifact_path=self._artifact_path)
        finally:
            try:
                mlflow.end_run()
            except Exception:
                pass