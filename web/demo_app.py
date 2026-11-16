"""Model test interface - http://127.0.0.1:8002"""
import json
import shutil
import sys
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sgmbf.common import TRAITS, load_config, write_pid  # noqa: E402
from sgmbf.infer import Predictor  # noqa: E402

cfg = load_config()
write_pid(cfg, "demo")
app = FastAPI(title="SG-MBF demo")
pred = Predictor(cfg)
progress = {"msg": ""}
_info = {"data": None}
_lock = threading.Lock()


def dataset():
    """Lazy dataset scan (labels + video paths) for ground truth and test-clip playback."""
    with _lock:
        if _info["data"] is None:
            from sgmbf.data import scan
            try:
                _info["data"] = scan(cfg)
            except Exception:
                _info["data"] = {"videos": {}, "labels": {}, "transcripts": {}}
        return _info["data"]


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "static" / "demo.html").read_text(encoding="utf-8")


@app.get("/api/models")
def models():
    runs = Path(cfg["paths"]["runs"]) / "main"
    ck = sorted(p.parent.name for p in runs.glob("SG-MBF__TAV__s*/model.pt"))
    metas = []
    for c in ck:
        m = json.loads((runs / c / "meta.json").read_text())
        metas.append({"run": c, "seed": m["seed"], "test_acc": m.get("test", {}).get("mean", {}).get("acc"),
                      "split_kind": m.get("split_kind")})
    return {"checkpoints": metas, "loaded": len(pred.models)}


@app.get("/api/test_clips")
def test_clips():
    p = Path(cfg["paths"]["results"]) / "test_clips.json"
    clips = json.loads(p.read_text()) if p.exists() else []
    labels = dataset()["labels"]
    return [{"clip": c, "truth": {t: labels[c][t] for t in TRAITS} if c in labels else None} for c in clips]


@app.get("/api/video/{clip}")
def video(clip: str):
    v = dataset()["videos"].get(clip)
    if not v:
        raise HTTPException(404)
    return FileResponse(v, media_type="video/mp4")


@app.get("/api/progress")
def get_progress():
    return progress


class ClipReq(BaseModel):
    clip: str


@app.post("/api/predict_clip")
async def predict_clip(req: ClipReq):
    try:
        out = await run_in_threadpool(pred.predict, None, req.clip)
    except Exception as ex:
        raise HTTPException(400, str(ex))
    lab = dataset()["labels"].get(req.clip)
    out["truth"] = {t: lab[t] for t in TRAITS} if lab else None
    return out


@app.post("/api/predict")
async def predict_video(file: UploadFile = File(...), transcript: str = Form("")):
    tmp = Path(tempfile.mkdtemp()) / (Path(file.filename or "video.mp4").name)
    with open(tmp, "wb") as f:
        shutil.copyfileobj(file.file, f)
    clip = tmp.name
    lab = dataset()["labels"].get(clip)
    if not transcript and clip in dataset()["transcripts"]:
        transcript = dataset()["transcripts"][clip]

    def log(m):
        progress["msg"] = m

    try:
        log("Loading models...")
        out = await run_in_threadpool(pred.predict, str(tmp), None, transcript or None, log)
    except Exception as ex:
        raise HTTPException(400, str(ex))
    finally:
        progress["msg"] = ""
        shutil.rmtree(tmp.parent, ignore_errors=True)
    out["truth"] = {t: lab[t] for t in TRAITS} if lab else None
    return out


@app.post("/api/reload")
def reload_models():
    return {"loaded": pred.load_models()}


@app.post("/api/release")
def release():
    pred.release()
    return {"ok": True}
