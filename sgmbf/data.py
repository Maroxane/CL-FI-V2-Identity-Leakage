"""Dataset discovery for ChaLearn First Impressions V2: videos, labels, transcripts, splits."""
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import TRAITS, TRAIT_KEYS

SPLIT_FILES = {"train": "training", "val": "validation", "test": "test"}
# codes used in the official eth_gender_annotations_*.csv files (ChaLearn dataset 24, data 58)
GENDER = {"1": "male", "2": "female"}
ETHNICITY = {"1": "Asian", "2": "Caucasian", "3": "African-American"}


def _load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f, encoding="latin1")


def youtube_id(clip):
    """'_uNup91ZYw0.002.mp4' -> '_uNup91ZYw0' (several clips come from one YouTube video)."""
    return clip.rsplit(".", 2)[0] if clip.count(".") >= 2 else clip


def scan(cfg):
    """Return a dict describing everything found under dataset_root."""
    root = Path(cfg["paths"]["dataset_root"])
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {root}")
    roots = [root] + [Path(p) for p in [cfg["paths"].get("official_archives")] if p and Path(p).exists()]
    rg = lambda pat: (p for r in roots for p in r.rglob(pat))
    videos = {}
    for p in rg("*.mp4"):
        videos.setdefault(p.name, str(p))
    labels, transcripts, ann_found, demo_files = {}, {}, {}, {}
    for split, word in SPLIT_FILES.items():
        for p in rg(f"annotation_{word}*.pkl"):
            ann_found[split] = str(p)
            d = _load_pkl(p)
            for t in TRAITS:
                for clip, v in d[TRAIT_KEYS[t]].items():
                    labels.setdefault(clip, {})[t] = float(v)
                    labels[clip].setdefault("_split", split)
        for p in rg(f"transcription_{word}*.pkl"):
            for clip, txt in _load_pkl(p).items():
                transcripts[clip] = txt or ""
        for p in list(rg(f"eth_gender_annotations_{word}*.csv")) + list(rg(f"*gender*{word}*.csv")):
            demo_files[split] = str(p)
    labelled = {c: v for c, v in labels.items() if all(t in v for t in TRAITS)}
    return {"root": ", ".join(str(r) for r in roots), "videos": videos, "labels": labelled, "transcripts": transcripts,
            "annotation_files": ann_found, "demographic_files": demo_files}


def make_splits(cfg, info, logger=None):
    """Official splits when complete, otherwise an internal split grouped by YouTube id."""
    mode = cfg["data"]["split_mode"]
    vids, labels = info["videos"], info["labels"]
    usable = sorted(c for c in labels if c in vids)
    if not usable:
        raise RuntimeError(
            "No labelled video found. Put annotation_training.pkl (and validation/test) "
            "from ChaLearn First Impressions V2 inside the dataset folder.")
    official = {s: [c for c in usable if labels[c]["_split"] == s] for s in SPLIT_FILES}
    complete = all(len(official[s]) > 0 for s in SPLIT_FILES)
    if mode == "official" or (mode == "auto" and complete):
        splits, kind = official, "official"
    else:
        groups = defaultdict(list)
        for c in usable:
            groups[youtube_id(c)].append(c)
        keys = sorted(groups)
        random.Random(cfg["data"]["internal_split_seed"]).shuffle(keys)
        fr = cfg["data"]["internal_split"]
        n = len(keys)
        a, b = int(fr[0] * n), int((fr[0] + fr[1]) * n)
        part = {"train": keys[:a], "val": keys[a:b], "test": keys[b:]}
        splits = {s: sorted(c for k in ks for c in groups[k]) for s, ks in part.items()}
        kind = "internal"
    mx = cfg["data"].get("max_clips")
    if mx:
        rng = random.Random(0)
        for s in splits:
            k = max(4, int(mx * {"train": .7, "val": .15, "test": .15}[s]))
            splits[s] = sorted(rng.sample(splits[s], min(k, len(splits[s]))))
    if logger:
        logger.info("Split mode: %s | train %d, val %d, test %d", kind,
                    *(len(splits[s]) for s in ("train", "val", "test")))
    return splits, kind


def audit(cfg, info, splits, kind):
    """Identity overlap between splits (YouTube id as proxy) and other dataset facts for the article."""
    ids = {s: {youtube_id(c) for c in v} for s, v in splits.items()}
    tr = info["transcripts"]
    empty = sum(1 for s in splits.values() for c in s if len((tr.get(c) or "").split()) < 5) if tr else None
    out = {
        "split_kind": kind,
        "n_videos_found": len(info["videos"]),
        "n_labelled": len(info["labels"]),
        "n_per_split": {s: len(v) for s, v in splits.items()},
        "annotation_files": info["annotation_files"],
        "transcripts_available": bool(tr),
        "clips_with_short_or_empty_transcript": empty,
        "youtube_ids_shared": {
            "train_val": len(ids["train"] & ids["val"]),
            "train_test": len(ids["train"] & ids["test"]),
            "val_test": len(ids["val"] & ids["test"]),
        },
        "demographic_files": info["demographic_files"],
    }
    p = Path(cfg["paths"]["results"]) / "split_audit.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def label_matrix(info, clips):
    return np.array([[info["labels"][c][t] for t in TRAITS] for c in clips], dtype=np.float32)


def load_demographics(info, split="test"):
    """Returns {clip: {'gender': .., 'ethnicity': ..}} from the ChaLearn csv if present."""
    p = info["demographic_files"].get(split)
    if not p:
        return {}
    import csv
    out = {}
    with open(p, newline="", encoding="utf-8", errors="ignore") as f:
        head = f.readline()
        f.seek(0)
        rd = csv.DictReader(f, delimiter=";" if head.count(";") > head.count(",") else ",")
        for row in rd:
            row = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
            clip = row.get("videoname") or row.get("video") or row.get("name")
            if clip:
                g, e = row.get("gender", ""), row.get("ethnicity", "")
                out[clip] = {"gender": GENDER.get(g, g), "ethnicity": ETHNICITY.get(e, e)}
    return out
