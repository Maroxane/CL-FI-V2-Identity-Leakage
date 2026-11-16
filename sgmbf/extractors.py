"""Frozen feature extractors (Section 3.2-3.4 of the article), tuned for a 6-12 GB RTX 3060.

One model family is kept on the GPU at a time; everything runs under torch.inference_mode.
"""
import gc

import numpy as np
import torch

from . import descriptors as D
from .media import sample_frames

EMO_MEAN = torch.tensor([91.4953, 103.8827, 131.0912]).view(1, 3, 1, 1)  # BGR means of Emo-AffectNet


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def device_of(cfg):
    return "cuda" if (cfg["extraction"]["device"] == "cuda" and torch.cuda.is_available()) else "cpu"


# ------------------------------------------------------------------ text
def _expose_torch_cuda_dlls():
    """On Windows, let CTranslate2 (faster-whisper) find cuBLAS/cuDNN DLLs shipped with PyTorch."""
    import os
    if os.name == "nt":
        lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        try:
            os.add_dll_directory(lib)
        except Exception:
            pass
        os.environ["PATH"] = lib + os.pathsep + os.environ.get("PATH", "")


class Transcriber:
    def __init__(self, cfg):
        _expose_torch_cuda_dlls()
        e = cfg["extraction"]
        self.e, self.lang = e, e.get("whisper_language")
        self._load(device_of(cfg))

    def _load(self, dev):
        from faster_whisper import WhisperModel
        self.dev = dev
        try:
            self.model = WhisperModel(self.e["whisper_model"], device=dev,
                                      compute_type=self.e["whisper_compute_type"] if dev == "cuda" else "int8")
        except Exception:
            if dev == "cpu":
                raise
            self._load("cpu")

    def __call__(self, wav):
        if wav.size < 1600:
            return ""
        try:
            segs, _ = self.model.transcribe(wav, language=self.lang, beam_size=5, vad_filter=True)
            return " ".join(s.text.strip() for s in segs).strip()
        except RuntimeError as ex:  # e.g. missing cuBLAS/cuDNN DLL -> CPU fallback
            if self.dev == "cuda" and ("cublas" in str(ex).lower() or "cudnn" in str(ex).lower() or "cuda" in str(ex).lower()):
                self._load("cpu")
                return self(wav)
            raise


class TextEncoder:
    """BERT [CLS] (768) + Sentence-BERT (768) + Empath normalised categories (194 in current Empath)."""

    def __init__(self, cfg):
        from transformers import AutoModel, AutoTokenizer
        from sentence_transformers import SentenceTransformer
        from empath import Empath
        e = cfg["extraction"]
        self.dev = device_of(cfg)
        self.fp16 = e["fp16"] and self.dev == "cuda"
        self.max_tokens = e["max_tokens"]
        self.tok = AutoTokenizer.from_pretrained(e["bert_model"])
        self.bert = AutoModel.from_pretrained(e["bert_model"]).to(self.dev).eval()
        if self.fp16:
            self.bert.half()
        self.sbert = SentenceTransformer(e["sbert_model"], device=self.dev)
        self.empath = Empath()
        self.cats = sorted(self.empath.cats.keys())

    def empath_vec(self, text):
        if len(text.split()) == 0:
            return np.zeros(len(self.cats), np.float32)
        d = self.empath.analyze(text, normalize=True) or {}
        v = np.array([d.get(c, 0.0) for c in self.cats], np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    @torch.inference_mode()
    def __call__(self, texts):
        texts = [t if t.strip() else "." for t in texts]
        enc = self.tok(texts, padding=True, truncation=True, max_length=self.max_tokens,
                       return_tensors="pt").to(self.dev)
        cls = self.bert(**enc).last_hidden_state[:, 0].float().cpu().numpy()
        sb = self.sbert.encode(texts, batch_size=len(texts), convert_to_numpy=True,
                               show_progress_bar=False).astype(np.float32)
        emp = np.stack([self.empath_vec(t if t != "." else "") for t in texts])
        return cls, sb, emp


# ------------------------------------------------------------------ audio
class AudioEncoder:
    """VGGish clip-mean embedding (128) + eGeMAPSv02 functionals (88)."""

    def __init__(self, cfg):
        import opensmile
        self.dev = device_of(cfg)
        self.sr = cfg["extraction"]["sample_rate"]
        self.vgg = torch.hub.load("harritaylor/torchvggish", "vggish", trust_repo=True)
        self.vgg.postprocess = False
        self.vgg.preprocess = True
        self.vgg.device = torch.device(self.dev)
        self.vgg.to(self.dev).eval()
        self.smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                                     feature_level=opensmile.FeatureLevel.Functionals)

    @torch.inference_mode()
    def __call__(self, wav):
        if wav.size < self.sr:  # VGGish needs >= 0.96 s
            wav = np.pad(wav, (0, self.sr - wav.size))
        emb = self.vgg(wav, self.sr)
        emb = emb.reshape(-1, 128).float().mean(0).cpu().numpy()
        ge = self.smile.process_signal(wav, self.sr).values.reshape(-1)[:88].astype(np.float32)
        return emb.astype(np.float32), np.nan_to_num(ge)


# ------------------------------------------------------------------ visual (CPU worker, one process each)
def _crop(frame, lm, margin):
    h, w = frame.shape[:2]
    x0, y0 = lm[:, 0].min() * w, lm[:, 1].min() * h
    x1, y1 = lm[:, 0].max() * w, lm[:, 1].max() * h
    s = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    a, b = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
    c, d = int(min(w, cx + s / 2)), int(min(h, cy + s / 2))
    if c - a < 10 or d - b < 10:
        return None
    import cv2
    return cv2.resize(frame[b:d, a:c], (224, 224), interpolation=cv2.INTER_AREA)


def visual_worker(args):
    """Runs in a separate process: MediaPipe Face Mesh + Pose on frames sampled at `fps`."""
    clip, path, fps, n_emo, margin = args[:5]
    emo_only = len(args) > 5 and args[5]
    try:
        import mediapipe as mp
        frames, dur, fps = sample_frames(path, fps=fps, return_fps=True)
        if emo_only:  # refresh the expression embeddings only: face mesh on n_emo frames, no pose
            sel = np.unique(np.linspace(0, len(frames) - 1, min(n_emo * 2, len(frames))).round().astype(int))
            crops = []
            with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                                 min_detection_confidence=0.5) as fm:
                for i in sel:
                    r = fm.process(frames[i])
                    if r.multi_face_landmarks:
                        lm = np.array([[p.x, p.y, p.z] for p in r.multi_face_landmarks[0].landmark], np.float32)
                        c = _crop(frames[i], lm, margin)
                        if c is not None:
                            crops.append(c)
            if len(crops) > n_emo:
                crops = [crops[int(round(j))] for j in np.linspace(0, len(crops) - 1, n_emo)]
            crops = np.stack(crops).astype(np.uint8) if crops else np.zeros((0, 224, 224, 3), np.uint8)
            return {"clip": clip, "ok": True, "crops": crops, "emo_only": True}
        geo = np.full((len(frames), D.N_GEO), np.nan, np.float32)
        pose = np.full((len(frames), D.N_POSE), np.nan, np.float32)
        face_lm = [None] * len(frames)
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                             min_detection_confidence=0.5) as fm, \
                mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1) as ps:
            for i, fr in enumerate(frames):
                r = fm.process(fr)
                if r.multi_face_landmarks:
                    lm = np.array([[p.x, p.y, p.z] for p in r.multi_face_landmarks[0].landmark], np.float32)
                    geo[i] = D.face_geometry(lm)
                    face_lm[i] = lm
                q = ps.process(fr)
                if q.pose_landmarks:
                    pl = np.array([[p.x, p.y, p.z, p.visibility] for p in q.pose_landmarks.landmark], np.float32)
                    pose[i] = D.pose_vector(pl)
        idx = [i for i, l in enumerate(face_lm) if l is not None]
        if idx:
            sel = [idx[int(round(j))] for j in np.linspace(0, len(idx) - 1, min(n_emo, len(idx)))]
            crops = [c for c in (_crop(frames[i], face_lm[i], margin) for i in sel) if c is not None]
        else:
            crops = []
        crops = np.stack(crops).astype(np.uint8) if crops else np.zeros((0, 224, 224, 3), np.uint8)
        return {"clip": clip, "ok": True, "geo": geo, "pose": pose, "crops": crops, "fps": fps, "dur": dur}
    except Exception as ex:  # keep the pool alive; the clip is logged and skipped
        return {"clip": clip, "ok": False, "error": repr(ex)}


class EmotionEncoder:
    """Emo-AffectNet (ResNet-50, Ryumina et al. 2022) penultimate 512-d embeddings.

    Falls back to an ImageNet ResNet-50 (2048-d) if the Emo-AffectNet weights cannot be loaded;
    the backbone actually used is stored in the cache metadata and reported in the dashboard.
    """

    def __init__(self, cfg):
        e = cfg["extraction"]
        self.dev = device_of(cfg)
        self.fp16 = e["fp16"] and self.dev == "cuda"
        self.batch = e["emo_batch"]
        try:
            from huggingface_hub import hf_hub_download
            from .emo_affectnet import load_emo_affectnet
            p = hf_hub_download(e["emo_repo"], e["emo_file"])
            self.model = load_emo_affectnet(p, self.dev)  # the HF release is a state_dict, not TorchScript
            self.model.extract_features(torch.zeros(1, 3, 224, 224, device=self.dev))
            self.name, self.dim, self.kind = "Emo-AffectNet ResNet-50", 512, "emo"
        except Exception as ex:
            import torchvision
            m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
            m.fc = torch.nn.Identity()
            self.model = m.to(self.dev).eval()
            self.name, self.dim, self.kind = f"ImageNet ResNet-50 fallback ({type(ex).__name__})", 2048, "imagenet"

    @torch.inference_mode()
    def __call__(self, crops):
        if len(crops) == 0:
            return np.zeros((0, self.dim), np.float32)
        out = []
        for i in range(0, len(crops), self.batch):
            x = torch.from_numpy(crops[i:i + self.batch]).to(self.dev).permute(0, 3, 1, 2).float()
            if self.kind == "emo":
                x = x.flip(1) - EMO_MEAN.to(self.dev)  # RGB -> BGR, mean subtraction
                with torch.autocast("cuda", enabled=self.fp16):
                    f = self.model.extract_features(x)
            else:
                x = (x / 255 - torch.tensor([0.485, 0.456, 0.406], device=self.dev).view(1, 3, 1, 1)) / \
                    torch.tensor([0.229, 0.224, 0.225], device=self.dev).view(1, 3, 1, 1)
                with torch.autocast("cuda", enabled=self.fp16):
                    f = self.model(x)
            out.append(f.float().reshape(len(x), -1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)
