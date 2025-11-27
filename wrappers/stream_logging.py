import sys
import time
import json
import logging
import mlflow
import tempfile
import os

class MLflowLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        from io import StringIO
        self.buffer = StringIO()
        self.on_emit = None

    def emit(self, record):
        msg = self.format(record)
        self.buffer.write(msg + "\n")
        if self.on_emit is not None:
            try:
                self.on_emit({"ts": time.time(), "stream": "log", "level": record.levelname, "text": msg})
            except Exception:
                pass

class TeeStream:
    def __init__(self, original, name, on_line):
        self.original = original
        self.name = name
        self.on_line = on_line
        self.buf = ""

    def write(self, s):
        # DEBUG
        try:
            sys.__stderr__.write(f"DEBUG: TeeStream write '{s}'\n")
        except:
            pass
        r = self.original.write(s)
        try:
            self.original.flush()
        except Exception:
            pass
        self.buf += s
        # sys.__stderr__.write(f"DEBUG: buf '{self.buf}'\n")
        while True:
            nl_pos = self.buf.find("\n")
            cr_pos = self.buf.find("\r")
            cut_pos = -1
            if nl_pos != -1 and cr_pos != -1:
                cut_pos = min(nl_pos, cr_pos)
            elif nl_pos != -1:
                cut_pos = nl_pos
            elif cr_pos != -1:
                cut_pos = cr_pos
import sys
import time
import json
import logging
import mlflow
import tempfile
import os

class MLflowLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        from io import StringIO
        self.buffer = StringIO()
        self.on_emit = None

    def emit(self, record):
        msg = self.format(record)
        self.buffer.write(msg + "\n")
        if self.on_emit is not None:
            try:
                self.on_emit({"ts": time.time(), "stream": "log", "level": record.levelname, "text": msg})
            except Exception:
                pass

class TeeStream:
    def __init__(self, original, name, on_line):
        self.original = original
        self.name = name
        self.on_line = on_line
        self.buf = ""

    def write(self, s):
        r = self.original.write(s)
        try:
            self.original.flush()
        except Exception:
            pass
        self.buf += s
        while True:
            nl_pos = self.buf.find("\n")
            cr_pos = self.buf.find("\r")
            cut_pos = -1
            if nl_pos != -1 and cr_pos != -1:
                cut_pos = min(nl_pos, cr_pos)
            elif nl_pos != -1:
                cut_pos = nl_pos
            elif cr_pos != -1:
                cut_pos = cr_pos
            else:
                break
            line = self.buf[:cut_pos]
            self.buf = self.buf[cut_pos + 1:]
            self.on_line({"ts": time.time(), "stream": self.name, "text": line})
        return r if isinstance(r, int) else len(s)

    def isatty(self):
        try:
            return self.original.isatty()
        except Exception:
            return False

    def fileno(self):
        try:
            return self.original.fileno()
        except Exception:
            raise

import threading

class ChunkedUploader:
    def __init__(self, base_path="logs/structured", max_lines=50, max_interval=1.0):
        self.base_path = base_path
        self.max_lines = max_lines
        self.max_interval = max_interval
        self.buf = []
        self.last_flush = time.time()
        self.part = 0
        self._last_text = ""
        self._session = []
        self.lock = threading.Lock()

    def append(self, entry):
        with self.lock:
            if isinstance(entry.get("text"), str):
                if entry["text"] == self._last_text:
                    return
                self._last_text = entry["text"]
            self.buf.append(entry)
            try:
                self._session.append(entry)
            except Exception:
                pass
            now = time.time()
            if len(self.buf) >= self.max_lines or (now - self.last_flush) >= self.max_interval:
                self._flush_unsafe()

    def _flush_unsafe(self):
        if not self.buf:
            return
        try:
            import mlflow as _mlflow
            if _mlflow.active_run() is None:
                return
        except Exception:
            return
        name = f"part_{self.part:06d}.json"
        try:
            tmp_dir = tempfile.mkdtemp(prefix="mlflow_logs_")
            tmp_path = os.path.join(tmp_dir, name)
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.buf, f, ensure_ascii=False)
            mlflow.log_artifact(tmp_path, artifact_path=self.base_path)
        finally:
            try:
                os.remove(tmp_path)
                os.rmdir(tmp_dir)
            except Exception:
                pass
            self.buf.clear()
            self.last_flush = time.time()
            self.part += 1

    def flush(self):
        with self.lock:
            self._flush_unsafe()

    def close(self):
        self.flush()
        try:
            import mlflow as _mlflow
            if self._session:
                txt = "\n".join([str(e.get("text", "")) for e in self._session if isinstance(e.get("text"), str)])
                if txt:
                    import tempfile as _tempfile
                    import os as _os
                    d = _tempfile.mkdtemp(prefix="mlflow_text_")
                    p = _os.path.join(d, "real_time_logs.log")
                    with open(p, "w", encoding="utf-8") as f:
                        f.write(txt)
                    try:
                        _mlflow.log_artifact(p, artifact_path="logs")
                    finally:
                        try:
                            _os.remove(p)
                            _os.rmdir(d)
                        except Exception:
                            pass
        except Exception:
            pass

class StdoutStderrInterceptor:
    def __init__(self, base_path="logs/structured", max_lines=50, max_interval=1.0, attach_logging=True):
        self.base_path = base_path
        self.uploader = ChunkedUploader(base_path=self.base_path, max_lines=max_lines, max_interval=max_interval)
        self.attach_logging = attach_logging
        self.handler = None
        self._orig_stdout = None
        self._orig_stderr = None
        self._patched_handlers = []

    def __enter__(self):
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = TeeStream(self._orig_stdout, "stdout", self.uploader.append)
        sys.stderr = TeeStream(self._orig_stderr, "stderr", self.uploader.append)
        if self.attach_logging:
            self.handler = MLflowLogHandler()
            self.handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.handler.on_emit = self.uploader.append
            logging.getLogger().addHandler(self.handler)
            try:
                self._patch_existing_stream_handlers()
            except Exception:
                pass
        return self

    def __exit__(self, exc_type, exc, tb):
        if isinstance(sys.stdout, TeeStream):
            if sys.stdout.buf:
                self.uploader.append({"ts": time.time(), "stream": "stdout", "text": sys.stdout.buf})
        if isinstance(sys.stderr, TeeStream):
            if sys.stderr.buf:
                self.uploader.append({"ts": time.time(), "stream": "stderr", "text": sys.stderr.buf})
        self.uploader.close()
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        if self.attach_logging and self.handler is not None:
            try:
                logging.getLogger().removeHandler(self.handler)
            except Exception:
                pass
            # Removed redundant log_text call which was overwriting the logs uploaded by uploader.close()
            # try:
            #     mlflow.log_text(self.handler.buffer.getvalue(), artifact_file="logs/real_time_logs.log")
            # except Exception:
            #     pass
        for h, orig_stream in self._patched_handlers:
            try:
                h.stream = orig_stream
            except Exception:
                pass

    def _patch_existing_stream_handlers(self):
        root = logging.getLogger()
        def patch_handlers(handlers):
            for h in handlers:
                import logging as _logging
                if isinstance(h, _logging.StreamHandler):
                    orig = h.stream
                    try:
                        name = "stderr" if orig is sys.__stderr__ or orig is self._orig_stderr else "stdout"
                        tee = TeeStream(orig, name, self.uploader.append)
                        h.stream = tee
                        self._patched_handlers.append((h, orig))
                    except Exception:
                        pass
        patch_handlers(getattr(root, 'handlers', []))
        try:
            logger_dict = root.manager.loggerDict
            for _name, _logger in logger_dict.items():
                try:
                    if hasattr(_logger, 'handlers'):
                        patch_handlers(_logger.handlers)
                except Exception:
                    continue
        except Exception:
            pass

def intercept_stdout_stderr_mlflow(base_path="logs/structured", max_lines=50, max_interval=1.0, attach_logging=True):
    return StdoutStderrInterceptor(base_path=base_path, max_lines=max_lines, max_interval=max_interval, attach_logging=attach_logging)
