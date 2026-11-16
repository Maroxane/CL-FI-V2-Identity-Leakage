"""Stage 1 - feature caching (resumable). One npz per clip and modality in cache/{text,audio,visual}/.

    python -m sgmbf.extract [--stages text,audio,visual]
"""
import os
import argparse
import json
import multiprocessing as mp
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .common import Progress, State, get_logger, load_config, remove_pid, write_pid
from .data import audit, make_splits, scan
from .media import load_audio


def cache_dir(cfg, mod):
    p = Path(cfg["paths"]["cache"]) / mod
    p.mkdir(parents=True, exist_ok=True)
    return p


def todo(cfg, mod, clips):
    d = cache_dir(cfg, mod)
    return [c for c in clips if not (d / f"{c}.npz").exists()]


def save_meta(cfg, **kw):
    p = Path(cfg["paths"]["cache"]) / "meta.json"
    m = json.loads(p.read_text()) if p.exists() else {}
    m.update(kw)
    p.write_text(json.dumps(m, indent=1))


# ------------------------------------------------------------------ text
def stage_text(cfg, info, clips, state, log):
    from .extractors import TextEncoder, Transcriber, free_gpu
    clips = todo(cfg, "text", clips)
    if not clips:
        log.info("[text] already cached")
        return
    tpath = Path(cfg["paths"]["cache"]) / "transcripts.json"
    trans = json.loads(tpath.read_text(encoding="utf-8")) if tpath.exists() else {}
    if cfg["extraction"]["use_provided_transcripts"]:
        for c in clips:
            if c in info["transcripts"] and c not in trans:
                trans[c] = info["transcripts"][c]
    missing = [c for c in clips if c not in trans]
    if missing:
        log.info("[text] Whisper transcription of %d clips", len(missing))
        asr = Transcriber(cfg)
        pr = Progress(state, "Transcription (Whisper)", len(missing))
        sr = cfg["extraction"]["sample_rate"]
        with ThreadPoolExecutor(cfg["extraction"]["audio_workers"]) as ex:
            for i, (c, wav) in enumerate(zip(missing, ex.map(lambda c: load_audio(info["videos"][c], sr), missing))):
                trans[c] = asr(wav)
                pr.step(msg=c)
                if i % 50 == 0:
                    tpath.write_text(json.dumps(trans), encoding="utf-8")
        tpath.write_text(json.dumps(trans), encoding="utf-8")
        del asr
        free_gpu()
    enc = TextEncoder(cfg)
    save_meta(cfg, text_dims={"bert": 768, "sbert": enc.sbert.get_sentence_embedding_dimension(),
                              "empath": len(enc.cats)}, empath_categories=enc.cats)
    bs = cfg["extraction"]["text_batch_size"]
    pr = Progress(state, "Text features (BERT, S-BERT, Empath)", len(clips))
    d = cache_dir(cfg, "text")
    for i in range(0, len(clips), bs):
        batch = clips[i:i + bs]
        cls, sb, emp = enc([trans.get(c, "") for c in batch])
        for j, c in enumerate(batch):
            np.savez(d / f"{c}.npz", bert=cls[j].astype(np.float16), sbert=sb[j].astype(np.float16),
                     empath=emp[j], n_words=len(trans.get(c, "").split()))
        pr.step(len(batch), msg=batch[-1])
    del enc
    free_gpu()


# ------------------------------------------------------------------ audio
def stage_audio(cfg, info, clips, state, log):
    from .extractors import AudioEncoder, free_gpu
    clips = todo(cfg, "audio", clips)
    if not clips:
        log.info("[audio] already cached")
        return
    enc = AudioEncoder(cfg)
    sr = cfg["extraction"]["sample_rate"]
    d = cache_dir(cfg, "audio")
    pr = Progress(state, "Audio features (VGGish, eGeMAPS)", len(clips))
    with ThreadPoolExecutor(cfg["extraction"]["audio_workers"]) as ex:
        src = lambda c: info.get("audio_files", {}).get(c) or info["videos"][c]
        for c, wav in zip(clips, ex.map(lambda c: load_audio(src(c), sr), clips)):
            try:
                vgg, ge = enc(wav)
                np.savez(d / f"{c}.npz", vggish=vgg, egemaps=ge, dur=wav.size / sr)
            except Exception as e:
                log.warning("[audio] %s failed: %r", c, e)
            pr.step(msg=c)
    del enc
    free_gpu()


# ------------------------------------------------------------------ visual
def stage_visual(cfg, info, clips, state, log):
    from .extractors import EmotionEncoder, free_gpu, visual_worker
    d = cache_dir(cfg, "visual")
    new = todo(cfg, "visual", clips)
    e = cfg["extraction"]
    emo = EmotionEncoder(cfg)
    # clips cached with another expression backbone (e.g. the ImageNet fallback): embeddings refreshed only
    stale = []
    for c in sorted(set(clips) - set(new)):
        try:
            with np.load(d / f"{c}.npz") as z:
                if z["emo"].shape[1] != emo.dim:
                    stale.append(c)
        except Exception:
            new.append(c)
    if not new and not stale:
        log.info("[visual] already cached")
        return
    log.info("[visual] expression backbone: %s | %d new clip(s), %d clip(s) to refresh", emo.name, len(new), len(stale))
    for old in Path(cfg["paths"]["cache"]).glob("blocks_*.npz"):  # aggregated blocks are now outdated
        old.unlink()
    save_meta(cfg, emotion_backbone=emo.name, emotion_dim=emo.dim, visual_fps=e["visual_fps"])
    for label, group, only in (("Expression embeddings (Emo-AffectNet refresh)", stale, True),
                               ("Visual features (MediaPipe, Emo-AffectNet)", new, False)):
        if not group:
            continue
        pr = Progress(state, label, len(group))
        jobs = [(c, info["videos"][c], e["visual_fps"], e["emo_frames"], e["face_margin"], only) for c in group]
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=max(1, e["visual_workers"]), maxtasksperchild=200) as pool:
            for r in pool.imap_unordered(visual_worker, jobs, chunksize=1):
                if r["ok"]:
                    emb = emo(r["crops"]).astype(np.float16)
                    f = d / f"{r['clip']}.npz"
                    if r.get("emo_only"):
                        with np.load(f) as z:
                            old = {k: z[k] for k in z.files}
                        old.update(emo=emb, n_faces=len(r["crops"]))
                        np.savez(str(f) + ".tmp.npz", **old)
                        os.replace(str(f) + ".tmp.npz", f)
                    else:
                        np.savez(f, geo=r["geo"], pose=r["pose"], emo=emb,
                                 fps=r["fps"], dur=r["dur"], n_faces=len(r["crops"]))
                else:
                    log.warning("[visual] %s failed: %s", r["clip"], r["error"])
                pr.step(msg=r["clip"])
    del emo
    free_gpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="text,audio,visual")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("extract", cfg)
    state = State(cfg)
    write_pid(cfg, "job")
    try:
        state.update(job="extract", status="running", stage="Scanning dataset", started=time.time(), error=None)
        info = scan(cfg)
        splits, kind = make_splits(cfg, info, log)
        a = audit(cfg, info, splits, kind)
        state.update(split_kind=kind, n_per_split=a["n_per_split"])
        clips = sorted(set(sum(splits.values(), [])))
        for st in args.stages.split(","):
            {"text": stage_text, "audio": stage_audio, "visual": stage_visual}[st.strip()](cfg, info, clips, state, log)
        state.update(status="finished", stage="Feature extraction finished", finished=time.time())
        log.info("Feature extraction finished")
    except Exception as ex:
        log.error(traceback.format_exc())
        state.update(status="error", error=repr(ex))
        raise
    finally:
        remove_pid(cfg, "job")


if __name__ == "__main__":
    main()
