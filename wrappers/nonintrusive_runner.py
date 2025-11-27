import os
import sys
import time
import shlex
import threading
import subprocess
import argparse
import mlflow

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from wrappers.stream_logging import ChunkedUploader

def get_args():
    p = argparse.ArgumentParser(description='Non-intrusive MLflow runner: intercept stdout/stderr and upload chunks')
    p.add_argument('--script', type=str, required=True, help='Path to python script to run')
    return p.parse_args()

def main():
    args = get_args()
    mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5002'))
    exp_name = os.environ.get('MLFLOW_EXPERIMENT_NAME', 'NonIntrusive-Run-02')
    try:
        mlflow.set_experiment(exp_name)
    except Exception:
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            exp = client.get_experiment_by_name(exp_name)
            if exp and getattr(exp, 'lifecycle_stage', '') == 'deleted':
                client.restore_experiment(exp.experiment_id)
                mlflow.set_experiment(exp_name)
            else:
                fallback_exp = os.environ.get('MLFLOW_EXPERIMENT_FALLBACK', 'Default')
                try:
                    mlflow.set_experiment(fallback_exp)
                except Exception:
                    alt_name = f"{exp_name}-{int(time.time())}"
                    mlflow.set_experiment(alt_name)
        except Exception:
            fallback_exp = os.environ.get('MLFLOW_EXPERIMENT_FALLBACK', 'Default')
            try:
                mlflow.set_experiment(fallback_exp)
            except Exception:
                alt_name = f"{exp_name}-{int(time.time())}"
                mlflow.set_experiment(alt_name)

    cmd = [sys.executable, args.script]
    uploader = ChunkedUploader(base_path='logs/structured', max_lines=int(os.environ.get('LOG_CHUNK_LINES', '50')), max_interval=float(os.environ.get('LOG_CHUNK_INTERVAL', '1.0')))

    def reader(pipe, name):
        try:
            for line in iter(pipe.readline, ''):
                if not line:
                    continue
                txt = line.rstrip('\n').rstrip('\r')
                if not txt:
                    continue
                uploader.append({'ts': time.time(), 'stream': name, 'text': txt})
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    env = dict(os.environ)
    env['PYTHONUNBUFFERED'] = env.get('PYTHONUNBUFFERED', '1')
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, env=env)

    attached_run = {'id': None}
    import re
    run_id_patterns = [
        re.compile(r"当前运行ID\s*[:：]\s*([0-9a-f\-]{8,})", re.IGNORECASE),
        re.compile(r"run_id\s*[:：]\s*([0-9a-f\-]{8,})", re.IGNORECASE),
        re.compile(r"Run ID\s*[:：]\s*([0-9a-f\-]{8,})", re.IGNORECASE),
    ]
    attach_timeout = float(os.environ.get('MLFLOW_ATTACH_TIMEOUT', '0.2'))
    fallback_run_name = os.environ.get('MLFLOW_RUN_NAME', 'NonIntrusive-Run')

    def watchdog():
        try:
            time.sleep(attach_timeout)
            if attached_run['id'] is None and mlflow.active_run() is None:
                try:
                    r = mlflow.start_run(run_name=fallback_run_name)
                    mlflow.log_params({'delegate': 'nonintrusive_runner', 'script': args.script})
                    attached_run['id'] = r.info.run_id
                except Exception:
                    pass
        except Exception:
            pass

    def reader(pipe, name):
        try:
            for line in iter(pipe.readline, ''):
                if not line:
                    continue
                txt = line.rstrip('\n').rstrip('\r')
                if not txt:
                    continue
                for pat in run_id_patterns:
                    m = pat.search(txt)
                    if m:
                        rid = m.group(1)
                        if attached_run['id'] is None:
                            try:
                                if mlflow.active_run() is not None:
                                    try:
                                        mlflow.end_run()
                                    except Exception:
                                        pass
                                mlflow.start_run(run_id=rid)
                                mlflow.log_params({'delegate': 'nonintrusive_runner', 'script': args.script})
                                attached_run['id'] = rid
                            except Exception:
                                pass
                        break
                uploader.append({'ts': time.time(), 'stream': name, 'text': txt})
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    t_out = threading.Thread(target=reader, args=(proc.stdout, 'stdout'), daemon=True)
    t_err = threading.Thread(target=reader, args=(proc.stderr, 'stderr'), daemon=True)
    t_watch = threading.Thread(target=watchdog, daemon=True)
    t_out.start(); t_err.start()
    t_watch.start()
    try:
        rc = proc.wait()
    finally:
        try:
            t_watch.join(timeout=max(0.5, attach_timeout))
        except Exception:
            pass
        uploader.close()
        try:
            import mlflow as _mlflow
            if attached_run['id'] is not None and _mlflow.active_run() is not None:
                _mlflow.end_run()
        except Exception:
            pass
    t_out.join(timeout=1.0)
    t_err.join(timeout=1.0)

if __name__ == '__main__':
    main()
