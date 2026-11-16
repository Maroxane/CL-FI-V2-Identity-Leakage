"""Shared helpers: configuration, paths, job state and logging."""
import json
import logging
import os
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
TRAITS = ["O", "C", "E", "A", "N"]
TRAIT_KEYS = {"O": "openness", "C": "conscientiousness", "E": "extraversion",
              "A": "agreeableness", "N": "neuroticism"}


def load_config(path=None):
    path = Path(path or ROOT / "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k, v in cfg["paths"].items():
        p = Path(v)
        cfg["paths"][k] = str(p if p.is_absolute() else (ROOT / p).resolve())
        if k not in ("dataset_root", "mupta_root", "official_archives"):
            Path(cfg["paths"][k]).mkdir(parents=True, exist_ok=True)
    return cfg


def get_logger(name, cfg=None, logfile="job.log"):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if cfg is not None:
        fh = logging.FileHandler(Path(cfg["paths"]["logs"]) / logfile, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


class State:
    """Small JSON file read by the monitoring web page (runs/state.json)."""

    def __init__(self, cfg):
        self.path = Path(cfg["paths"]["runs"]) / "state.json"

    def read(self):
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def update(self, **kw):
        s = self.read()
        s.update(kw)
        s["updated"] = time.time()
        # Windows: the dashboard, an antivirus or OneDrive may hold state.json open for a few ms ->
        # retry, and never let a progress-file hiccup kill a training job
        txt = json.dumps(s, indent=1)
        tmp = self.path.with_name(f"state.{os.getpid()}.tmp")
        for i in range(20):
            try:
                tmp.write_text(txt, encoding="utf-8")
                os.replace(tmp, self.path)
                return s
            except OSError:
                time.sleep(0.05 * (i + 1))
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return s


class Progress:
    """Progress + ETA reporting for long loops, written into State."""

    def __init__(self, state, stage, total, every=5.0):
        self.state, self.stage, self.total = state, stage, max(int(total), 1)
        self.t0, self.last, self.every, self.done = time.time(), 0.0, every, 0

    def step(self, n=1, msg=None, force=False):
        self.done += n
        now = time.time()
        if force or now - self.last > self.every or self.done >= self.total:
            el = now - self.t0
            rate = self.done / el if el > 0 else 0
            eta = (self.total - self.done) / rate if rate > 0 else None
            self.state.update(stage=self.stage, done=self.done, total=self.total,
                              eta_s=eta, rate=rate, message=msg or "")
            self.last = now


def write_pid(cfg, name):
    d = Path(cfg["paths"]["runs"]) / "pids"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.pid").write_text(str(os.getpid()))


def remove_pid(cfg, name):
    p = Path(cfg["paths"]["runs"]) / "pids" / f"{name}.pid"
    if p.exists():
        p.unlink()
