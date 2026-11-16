"""Load cached per-clip features into matrices, one block per descriptor family."""
import hashlib
from pathlib import Path

import numpy as np

from . import descriptors as D

MODALITY_BLOCKS = {"t": ["bert", "sbert", "empath"], "a": ["vggish", "egemaps"], "v": ["geo", "emo", "pose"]}
BLOCK_FILE = {"bert": "text", "sbert": "text", "empath": "text", "vggish": "audio", "egemaps": "audio",
              "geo": "visual", "emo": "visual", "pose": "visual"}


def clip_blocks(cache, clip, frame_drop=0.0, rng=None):
    """All blocks for one clip, or None if a modality is missing from the cache."""
    out = {}
    try:
        t = np.load(Path(cache) / "text" / f"{clip}.npz")
        a = np.load(Path(cache) / "audio" / f"{clip}.npz")
        v = np.load(Path(cache) / "visual" / f"{clip}.npz")
    except (FileNotFoundError, OSError):
        return None
    out["bert"], out["sbert"], out["empath"] = (t["bert"].astype(np.float32), t["sbert"].astype(np.float32),
                                                t["empath"].astype(np.float32))
    out["vggish"], out["egemaps"] = a["vggish"].astype(np.float32), a["egemaps"].astype(np.float32)
    geo, pose, emo = v["geo"], v["pose"], v["emo"].astype(np.float32)
    fps = float(v["fps"])
    if frame_drop > 0:
        rng = rng or np.random.default_rng(0)
        geo, pose, emo = D.drop_frames(geo, frame_drop, rng), D.drop_frames(pose, frame_drop, rng), \
            D.drop_frames(emo, frame_drop, rng)
    edim = emo.shape[1] if emo.ndim == 2 and emo.shape[1] else 512
    out["geo"] = D.aggregate_geometry(geo)
    out["pose"] = D.aggregate_pose(pose, fps)
    out["emo"] = D.aggregate_emotion(emo, edim)
    return out


def load_blocks(cfg, clips, frame_drop=0.0, seed=0, use_disk_cache=True):
    """Returns (blocks: dict name -> (N, d) array, kept_clips)."""
    cache = cfg["paths"]["cache"]
    key = hashlib.md5(("|".join(clips) + f"{frame_drop}{seed}").encode()).hexdigest()[:12]
    f = Path(cache) / f"blocks_{key}.npz"
    if use_disk_cache and f.exists():
        z = np.load(f, allow_pickle=True)
        return {k: z[k] for k in z.files if k != "_clips"}, list(z["_clips"])
    rng = np.random.default_rng(seed)
    rows, kept = [], []
    for c in clips:
        b = clip_blocks(cache, c, frame_drop, rng)
        if b is not None:
            rows.append(b)
            kept.append(c)
    if not rows:
        raise RuntimeError("No clip has all three modalities cached (text, audio, visual) in "
                           f"{cache}. Run step 1 'Extraire les features' first (python -m sgmbf.extract).")
    blocks = {k: np.stack([r[k] for r in rows]).astype(np.float32) for k in rows[0]}
    for k in blocks:
        blocks[k] = np.nan_to_num(blocks[k], nan=0.0, posinf=0.0, neginf=0.0)
    if use_disk_cache:
        np.savez(f, _clips=np.array(kept), **blocks)
    return blocks, kept


def assemble(blocks, spec):
    """spec: {'t': [...block names], 'a': [...], 'v': [...]} -> {modality: (N, D_m)}; empty lists dropped."""
    return {m: np.concatenate([blocks[b] for b in bl], 1) for m, bl in spec.items() if bl}


class Standardizer:
    """Per-feature z-scoring with training-set statistics (Empath and BERT included)."""

    def fit(self, X):
        self.mu = {m: x.mean(0) for m, x in X.items()}
        # (near-)constant features are only centred, never divided by a tiny SD
        self.sd = {m: np.where(x.std(0) > 1e-4, x.std(0), 1.0).astype(np.float32) for m, x in X.items()}
        return self

    def __call__(self, X):
        # clipping protects against out-of-range inputs (corrupted modality, unusual video)
        return {m: np.clip((x - self.mu[m]) / self.sd[m], -8, 8).astype(np.float32) for m, x in X.items()}

    def state(self):
        return {"mu": {m: v.tolist() for m, v in self.mu.items()}, "sd": {m: v.tolist() for m, v in self.sd.items()}}

    @classmethod
    def from_state(cls, s):
        o = cls()
        o.mu = {m: np.array(v, np.float32) for m, v in s["mu"].items()}
        o.sd = {m: np.array(v, np.float32) for m, v in s["sd"].items()}
        return o
