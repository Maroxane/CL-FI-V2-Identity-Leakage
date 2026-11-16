"""Training monitor - http://127.0.0.1:8001"""
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import psutil
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sgmbf.common import ROOT, State, load_config, write_pid  # noqa: E402

cfg = load_config()
state = State(cfg)
RUNS, RES, LOGS, CACHE = (Path(cfg["paths"][k]) for k in ("runs", "results", "logs", "cache"))
PIDS = RUNS / "pids"
app = FastAPI(title="SG-MBF monitor")
write_pid(cfg, "monitor")

STEPS = {
    # every training step first runs the (resumable) feature extraction: instant when the cache is complete
    "download": ["download"], "extract": ["extract"],
    "main": ["extract", "main"], "fusion": ["extract", "fusion"], "ablation": ["extract", "ablation"],
    "robustness": ["extract", "robustness"], "fairness": ["extract", "fairness"],
    "mupta_extract": ["mupta_extract"], "mupta_cross": ["mupta_extract", "mupta_cross"],
    "mupta": ["mupta_extract", "mupta_cross"],
    "gate_select": ["extract", "gate_select"], "gate_final": ["extract", "gate_final"],
    "identity": ["identity"], "leakage": ["leakage"], "audit": ["identity", "leakage"],
    "phase2": ["phase2"], "control": ["control"],
    "train_all": ["extract", "main", "fusion", "ablation", "robustness", "fairness"],
    "pipeline": ["extract", "main", "fusion", "ablation", "robustness", "fairness"],
}


def _pid(name):
    p = PIDS / f"{name}.pid"
    try:
        pid = int(p.read_text())
        return pid if psutil.pid_exists(pid) else None
    except Exception:
        return None


def job_running():
    return _pid("pipeline") or _pid("job")


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "static" / "monitor.html").read_text(encoding="utf-8")


@app.get("/api/status")
def status():
    s = state.read()
    s["running"] = bool(job_running())
    s["now"] = time.time()
    return s


@app.get("/api/gpu")
def gpu():
    q = "name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    try:
        flags = 0x08000000 if os.name == "nt" else 0
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5, creationflags=flags).stdout.strip()
        n, u, mu, mt, t, pw = [x.strip() for x in out.splitlines()[0].split(",")]
        return {"ok": True, "name": n, "util": float(u), "mem_used": float(mu), "mem_total": float(mt),
                "temp": float(t), "power": float(pw) if pw not in ("[N/A]", "") else None}
    except Exception as ex:
        return {"ok": False, "error": repr(ex)}


@app.get("/api/runs")
def runs():
    out = []
    active = state.read().get("current_run") if job_running() else None
    for meta in RUNS.glob("*/*/meta.json"):
        try:
            m = json.loads(meta.read_text())
        except Exception:
            continue
        test = m.get("test", {}).get("mean", {})
        out.append({"group": meta.parent.parent.name, "run": meta.parent.name, "model": m.get("model"),
                    "label": m.get("label"), "seed": m.get("seed"), "params": m.get("params"),
                    "epochs": m.get("epochs_run"), "best_epoch": m.get("best_epoch"),
                    "best_val_loss": m.get("best_val_loss"), "test_acc": test.get("acc"), "test_ccc": test.get("ccc"),
                    "started": m.get("started"), "finished": m.get("finished"),
                    "active": meta.parent.name == active and not m.get("finished")})
    return sorted(out, key=lambda r: r.get("started") or 0, reverse=True)


@app.get("/api/run/{group}/{run}")
def run_metrics(group: str, run: str):
    p = RUNS / group / run / "metrics.jsonl"
    if not p.exists():
        raise HTTPException(404)
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    meta = json.loads((p.parent / "meta.json").read_text()) if (p.parent / "meta.json").exists() else {}
    return {"metrics": rows, "meta": meta}


@app.get("/api/log", response_class=PlainTextResponse)
def log(n: int = 200):
    out = []
    for name in ("job.log",):
        p = LOGS / name
        if p.exists():
            with open(p, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 200_000))
                out += f.read().decode("utf-8", "ignore").splitlines()[-n:]
    return "\n".join(out[-n:])


@app.get("/api/cache")
def cache():
    meta = json.loads((CACHE / "meta.json").read_text()) if (CACHE / "meta.json").exists() else {}
    counts = {m: len(list((CACHE / m).glob("*.npz"))) if (CACHE / m).exists() else 0 for m in ("text", "audio", "visual")}
    audit = json.loads((RES / "split_audit.json").read_text()) if (RES / "split_audit.json").exists() else {}
    meta.pop("empath_categories", None)
    return {"counts": counts, "meta": meta, "audit": audit}


@app.get("/api/diagnostic")
def diagnostic():
    d = {"python": sys.version.split()[0]}
    try:
        import torch
        d.update(torch=torch.__version__, cuda=torch.cuda.is_available(),
                 gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                 vram_gb=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1) if torch.cuda.is_available() else None)
    except Exception as ex:
        d["torch_error"] = repr(ex)
    root = Path(cfg["paths"]["dataset_root"])
    d["dataset_root"] = str(root)
    d["dataset_exists"] = root.exists()
    if root.exists():
        d["videos"] = sum(1 for _ in root.rglob("*.mp4"))
        d["annotation_files"] = [p.name for p in root.rglob("annotation_*.pkl")]
        d["transcription_files"] = [p.name for p in root.rglob("transcription_*.pkl")]
        d["folders"] = dict(Counter(p.parent.parent.name for p in root.rglob("*.mp4")))
    return d


@app.get("/api/results")
def results():
    t = RES / "tables.md"
    files = sorted(p.name for p in RES.glob("*.csv")) + sorted(p.name for p in RES.glob("*.json"))
    return {"tables_md": t.read_text(encoding="utf-8") if t.exists() else "", "files": files}


@app.get("/api/results/file/{name}")
def result_file(name: str):
    p = (RES / name).resolve()
    if p.parent != RES.resolve() or not p.exists():
        raise HTTPException(404)
    return FileResponse(p, filename=name)


class Job(BaseModel):
    action: str


@app.post("/api/job/start")
def start(job: Job):
    if job_running():
        raise HTTPException(409, "A job is already running. Stop it first.")
    if job.action not in STEPS:
        raise HTTPException(400, "unknown action")
    flags = (0x00000200 | 0x08000000) if os.name == "nt" else 0  # new process group, no console window
    console = open(LOGS / "job_console.log", "ab")
    state.update(status="starting", job=job.action, stage="Starting " + job.action, error=None,
                 done=0, total=0, eta_s=None, current_run=None, epoch=None)
    subprocess.Popen([sys.executable, "-m", "sgmbf.pipeline", "--steps", ",".join(STEPS[job.action])],
                     cwd=str(ROOT), stdout=console, stderr=subprocess.STDOUT, creationflags=flags)
    return {"ok": True}


@app.post("/api/job/stop")
def stop():
    killed = 0
    for name in ("pipeline", "job"):
        pid = _pid(name)
        if pid:
            try:
                parent = psutil.Process(pid)
                for c in parent.children(recursive=True):
                    c.kill()
                    killed += 1
                parent.kill()
                killed += 1
            except psutil.NoSuchProcess:
                pass
            (PIDS / f"{name}.pid").unlink(missing_ok=True)
    state.update(status="stopped", stage="Stopped by user")
    return {"ok": True, "killed": killed}
