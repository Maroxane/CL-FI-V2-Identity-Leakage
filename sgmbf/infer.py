"""End-to-end prediction for one video (used by the test web interface)."""
import json
import threading
import time
from pathlib import Path

import numpy as np
import torch

from . import descriptors as D
from .common import TRAITS
from .features import Standardizer, assemble, clip_blocks
from .models import build


class Predictor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.text = self.audio = self.emo = self.asr = None
        self.models = []
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------- checkpoints
    def available_runs(self):
        runs = Path(self.cfg["paths"]["runs"])
        return sorted(str(p.parent.relative_to(runs)) for p in runs.glob("*/*/model.pt"))

    def load_models(self, group="main", label="TAV"):
        runs = Path(self.cfg["paths"]["runs"]) / group
        self.models = []
        for p in sorted(runs.glob(f"SG-MBF__{label}__s*/model.pt")):
            ck = torch.load(p, map_location="cpu", weights_only=False)
            m = build(ck["name"], ck["dims"], self.cfg, ck.get("training", {}).get("gate_variant", "sigmoid"))
            m.load_state_dict(ck["state_dict"])
            self.models.append((p.parent.name, m.to(self.dev).eval(), Standardizer.from_state(ck["standardizer"]), ck["spec"]))
        return len(self.models)

    def release(self):
        with self.lock:
            self.text = self.audio = self.emo = self.asr = None
            from .extractors import free_gpu
            free_gpu()

    # -------------------------------------------------------------- features
    def _features(self, video, transcript=None, log=lambda *_: None):
        from .extractors import (AudioEncoder, EmotionEncoder, TextEncoder, Transcriber, visual_worker)
        from .media import load_audio
        e = self.cfg["extraction"]
        sr = e["sample_rate"]
        t0 = time.time()
        wav = load_audio(video, sr)
        if not transcript:
            log("Transcription (Whisper)...")
            if self.asr is None:
                self.asr = Transcriber(self.cfg)
            transcript = self.asr(wav)
        log("Text features...")
        if self.text is None:
            self.text = TextEncoder(self.cfg)
        cls, sb, emp = self.text([transcript])
        log("Audio features...")
        if self.audio is None:
            self.audio = AudioEncoder(self.cfg)
        vgg, ge = self.audio(wav)
        log("Visual features (MediaPipe, Emo-AffectNet)...")
        r = visual_worker(("demo", video, e["visual_fps"], e["emo_frames"], e["face_margin"]))
        if not r["ok"]:
            raise RuntimeError("Visual extraction failed: " + r["error"])
        if self.emo is None:
            self.emo = EmotionEncoder(self.cfg)
        emo = self.emo(r["crops"])
        b = {"bert": cls[0], "sbert": sb[0], "empath": emp[0], "vggish": vgg, "egemaps": ge,
             "geo": D.aggregate_geometry(r["geo"]), "pose": D.aggregate_pose(r["pose"], r["fps"]),
             "emo": D.aggregate_emotion(emo, self.emo.dim)}
        info = {"transcript": transcript, "n_frames": int(len(r["geo"])), "faces_used": int(len(r["crops"])),
                "face_rate": float(np.mean(~np.isnan(r["geo"][:, 0]))) if len(r["geo"]) else 0.0,
                "duration_s": float(wav.size / sr), "extraction_s": time.time() - t0,
                "expression_backbone": self.emo.name}
        return b, info

    # -------------------------------------------------------------- prediction
    def predict(self, video=None, clip=None, transcript=None, log=lambda *_: None):
        with self.lock:
            if not self.models and not self.load_models():
                raise RuntimeError("No trained SG-MBF checkpoint found (runs/main). Train the model first.")
            info = {}
            if clip is not None:
                b = clip_blocks(self.cfg["paths"]["cache"], clip)
                if b is None:
                    raise RuntimeError(f"{clip}: features not cached")
                tp = Path(self.cfg["paths"]["cache"]) / "transcripts.json"
                tr = json.loads(tp.read_text(encoding="utf-8")) if tp.exists() else {}
                info = {"transcript": tr.get(clip, ""), "source": "cached features"}
            else:
                b, info = self._features(video, transcript, log)
                info["source"] = "live extraction"
            blocks = {k: v[None].astype(np.float32) for k, v in b.items()}
            preds, gates, norms = [], [], []
            for name, m, std, spec in self.models:
                X = std(assemble(blocks, spec))
                X = {k: torch.as_tensor(v, device=self.dev) for k, v in X.items()}
                with torch.no_grad():
                    p, ex = m(X)
                preds.append(p[0].float().cpu().numpy())
                if "gates" in ex:
                    gates.append(ex["gates"][0].float().cpu().numpy())
                    norms.append(ex["norms"][0].float().cpu().numpy())
            P = np.clip(np.stack(preds), 0, 1)
            out = {"traits": {t: {"mean": float(P[:, j].mean()), "sd": float(P[:, j].std())} for j, t in enumerate(TRAITS)},
                   "n_models": len(self.models), **info}
            if gates:
                G, Nn = np.stack(gates), np.stack(norms)
                mods = self.models[0][1].mods
                out["gates"] = {m: float(G[:, i].mean()) for i, m in enumerate(mods)}
                contrib = (G * Nn).mean(0)
                out["contribution"] = {m: float(contrib[i] / contrib.sum()) for i, m in enumerate(mods)}
            return out
