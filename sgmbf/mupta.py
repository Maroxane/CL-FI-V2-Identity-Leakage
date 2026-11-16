"""MuPTA corpus (Ryumina et al., Interspeech 2023) for cross-dataset evaluation.

    python -m sgmbf.mupta            # translate transcripts + extract features into cache_mupta/

The same frozen extractors as for First Impressions are used. Russian transcripts are translated to
English (OPUS-MT ru-en) so that the English text branch (BERT, Sentence-BERT, Empath) can be applied.
"""
import copy
import csv
import json
import re
import time
import traceback
from pathlib import Path

import numpy as np

from .common import TRAITS, State, get_logger, load_config, remove_pid, write_pid

VIDEO_EXT = {".mov", ".mp4", ".avi", ".mkv"}
TRAIT_COL = {"O": "openness", "C": "conscientiousness", "E": "extraversion", "A": "agreeableness", "N": "neuroticism"}


def _speaker(path):
    for part in reversed(Path(path).parts):
        m = re.match(r"(speaker_\d+)", part)
        if m:
            return m.group(1)
    return None


def _read_csv(p):
    with open(p, newline="", encoding="utf-8-sig") as f:
        head = f.readline()
        f.seek(0)
        return list(csv.DictReader(f, delimiter=";" if head.count(";") > head.count(",") else ","))


def scan(cfg):
    """Clips (video + audio + Russian transcript) and per-speaker labels."""
    root = Path(cfg["paths"]["mupta_root"])
    # videos: Mupta/video/speaker_XX/{center,left,right}_NN.mov (as distributed) or Mupta/extracted/video/...
    # audio : Mupta/extracted/audio/speaker_XX/<view>_NN.wav (from the audio zips) - identical for the 3 views
    views = [v.lower() for v in (cfg.get("mupta", {}).get("views") or ["center"])]
    places = [d for d in (root / "video", root / "extracted" / "video", root / "extracted" / "audio",
                          root / "audio") if d.exists()]
    if not places:
        raise FileNotFoundError(f"No video/ or extracted/ folder in {root}: run extraire_mupta.bat first")
    videos, audio = {}, {}
    for d in places:
        for p in d.rglob("*"):
            if not p.is_file() or p.stem.split("_")[0].lower() not in views:
                continue
            spk = _speaker(p)
            if not spk:
                continue
            key = f"{spk}_{p.stem}"
            if p.suffix.lower() in VIDEO_EXT:
                videos.setdefault(key, str(p))
            elif p.suffix.lower() == ".wav":
                audio.setdefault(key, str(p))
    meta = root / "metadata"
    labels, subset = {}, {}
    for r in _read_csv(meta / "metadata_train_valid.csv"):
        spk = r["speaker_id"].strip()
        labels[spk] = {t: float(r[c]) for t, c in TRAIT_COL.items()}
        subset[spk] = r.get("subset", "").strip()
    test_rows = _read_csv(meta / "metadata_test.csv") if (meta / "metadata_test.csv").exists() else []
    for r in test_rows:
        subset.setdefault(r["speaker_id"].strip(), "test")
    trans = {}
    if (meta / "transcription.csv").exists():
        for r in _read_csv(meta / "transcription.csv"):
            trans[f"{r['speaker_id'].strip()}_{Path(r['name_video']).stem}"] = r.get("transcription") or ""
    clips = sorted(videos)
    return {"videos": videos, "audio_files": audio, "transcripts_ru": trans, "labels": labels,
            "subset": subset, "clips": clips, "speaker": {c: "_".join(c.split("_")[:2]) for c in clips},
            "test_rows": test_rows}


def load_marian(name):
    """OPUS-MT models ship a pickle checkpoint (pytorch_model.bin) that recent transformers refuse to load
    with torch < 2.6. Order: safetensors on main -> safetensors conversion PR on the Hub -> the .bin read
    with torch.load(weights_only=True) (tensors only, no code execution)."""
    import torch
    from transformers import MarianConfig, MarianMTModel
    for kw in ({}, {"revision": "refs/pr/4"}):
        try:
            return MarianMTModel.from_pretrained(name, use_safetensors=True, **kw)
        except Exception:
            pass
    from huggingface_hub import hf_hub_download
    m = MarianMTModel(MarianConfig.from_pretrained(name))
    sd = torch.load(hf_hub_download(name, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    m.tie_weights()
    bad = [k for k in missing if "embed_positions" not in k and "lm_head" not in k]
    if bad or unexpected:
        raise RuntimeError(f"{name}: checkpoint mismatch (missing {bad[:5]}, unexpected {unexpected[:5]})")
    return m


class Translator:
    def __init__(self, cfg):
        import torch
        from transformers import MarianMTModel, MarianTokenizer
        name = cfg["mupta"]["translate_model"]
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = MarianTokenizer.from_pretrained(name)
        self.model = load_marian(name).to(self.dev).eval()
        if self.dev == "cuda":
            self.model.half()

    def __call__(self, texts, bs=16):
        import torch
        out = []
        for i in range(0, len(texts), bs):
            batch = [t if t.strip() else "." for t in texts[i:i + bs]]
            enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.dev)
            with torch.inference_mode():
                gen = self.model.generate(**enc, max_new_tokens=512, num_beams=4)
            out += [t if s.strip() else "" for s, t in zip(batch, self.tok.batch_decode(gen, skip_special_tokens=True))]
        return out


def translate_all(cfg, info, log, state):
    """Sentence-level ru -> en translation, cached in cache_mupta/transcripts_en.json (resumable)."""
    path = Path(cfg["paths"]["cache_mupta"]) / "transcripts_en.json"
    done = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    todo = [c for c in info["clips"] if c not in done]
    if not todo:
        return done
    if cfg["mupta"]["text_mode"] != "translate":
        done.update({c: "" for c in todo})
        path.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
        return done
    from .common import Progress
    from .extractors import free_gpu
    tr = Translator(cfg)
    pr = Progress(state, "MuPTA: translating transcripts (ru -> en)", len(todo))
    for i, c in enumerate(todo):
        ru = info["transcripts_ru"].get(c, "")
        sents = [s for s in re.split(r"(?<=[.!?…])\s+", ru) if s.strip()]
        done[c] = " ".join(tr(sents)) if sents else ""
        pr.step(msg=c)
        if i % 50 == 0:
            path.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
    path.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
    del tr
    free_gpu()
    return done


def mupta_cfg(cfg):
    c = copy.deepcopy(cfg)
    c["paths"]["cache"] = cfg["paths"]["cache_mupta"]
    c["extraction"]["use_provided_transcripts"] = True
    return c


def main():
    cfg = load_config()
    log = get_logger("mupta", cfg)
    state = State(cfg)
    write_pid(cfg, "job")
    try:
        from .extract import stage_audio, stage_text, stage_visual
        state.update(job="mupta_extract", status="running", stage="MuPTA: scanning", error=None, started=time.time())
        info = scan(cfg)
        mx = cfg["mupta"].get("max_clips_per_speaker")
        if mx:
            keep, cnt = [], {}
            for c in info["clips"]:
                s = info["speaker"][c]
                if cnt.get(s, 0) < mx:
                    keep.append(c)
                    cnt[s] = cnt.get(s, 0) + 1
            info["clips"] = keep
        log.info("MuPTA: %d clips, %d with audio, %d with transcript, %d labelled speakers",
                 len(info["clips"]), sum(c in info["audio_files"] for c in info["clips"]),
                 sum(c in info["transcripts_ru"] for c in info["clips"]), len(info["labels"]))
        en = translate_all(cfg, info, log, state)
        info["transcripts"] = {c: en.get(c, "") for c in info["clips"]}
        mc = mupta_cfg(cfg)
        for st in (stage_text, stage_audio, stage_visual):
            st(mc, info, info["clips"], state, log)
        state.update(status="finished", stage="MuPTA feature extraction finished", finished=time.time())
    except Exception as ex:
        log.error(traceback.format_exc())
        state.update(status="error", error=repr(ex))
        raise
    finally:
        remove_pid(cfg, "job")


if __name__ == "__main__":
    main()
