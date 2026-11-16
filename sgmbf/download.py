"""Download the missing parts of ChaLearn First Impressions V2 from the Hugging Face mirror
(yeray142/first-impressions-v2) and convert its metadata into the files expected by SG-MBF.

    python -m sgmbf.download [--splits train,validation,test] [--workers 8]

- Only clips that are not already somewhere under Chalearn/ are downloaded (resumable).
- Non-video files (metadata.csv / .jsonl / .parquet, .pkl, .csv) are always fetched.
- If the metadata contain Big Five scores, transcriptions or gender/ethnicity, they are written as
  annotation_<split>.pkl, transcription_<split>.pkl and eth_gender_annotations_<split>.csv
  (existing official files are never overwritten).
"""
import argparse
import csv
import os
import pickle
import shutil
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .common import Progress, State, get_logger, load_config, remove_pid, write_pid

REPO = "yeray142/first-impressions-v2"
SPLIT_WORD = {"train": "training", "validation": "validation", "test": "test"}
TRAIT_COLS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism", "interview"]


def _read_table(path):
    import pandas as pd
    p = str(path)
    if p.endswith(".parquet"):
        return pd.read_parquet(p)
    if p.endswith(".jsonl") or p.endswith(".json"):
        return pd.read_json(p, lines=p.endswith(".jsonl"))
    return pd.read_csv(p, sep=None, engine="python")


def _clip_name(row, cols):
    for c in ("file_name", "filename", "video", "videoname", "video_name", "path", "name"):
        if c in cols and isinstance(row[cols[c]], str) and row[cols[c]].endswith(".mp4"):
            return os.path.basename(row[cols[c]])
    return None


def convert_metadata(root, log):
    """Turn every metadata table found under root/hf into the ChaLearn file formats."""
    hf = root / "hf"
    written = []
    for split, word in SPLIT_WORD.items():
        tables = [p for p in hf.rglob("*") if p.is_file() and p.suffix in (".csv", ".jsonl", ".json", ".parquet")
                  and (split in p.parts or split in p.name)]
        ann = {t: {} for t in TRAIT_COLS}
        trans, demo = {}, {}
        for t in tables:
            try:
                df = _read_table(t)
            except Exception as ex:
                log.warning("cannot read %s: %r", t, ex)
                continue
            cols = {c.lower().strip(): c for c in df.columns}
            for _, row in df.iterrows():
                clip = _clip_name(row, cols)
                if not clip:
                    continue
                for tr in TRAIT_COLS:
                    if tr in cols and row[cols[tr]] == row[cols[tr]]:
                        ann[tr][clip] = float(row[cols[tr]])
                for k in ("transcription", "transcript", "text"):
                    if k in cols and isinstance(row[cols[k]], str):
                        trans[clip] = row[cols[k]]
                        break
                g = row[cols["gender"]] if "gender" in cols else None
                e = row[cols["ethnicity"]] if "ethnicity" in cols else None
                if g is not None or e is not None:
                    demo[clip] = (g, e)
        n_lab = len(ann["openness"])
        targets = [
            (root / f"annotation_{word}.pkl", ann if n_lab and all(ann[t] for t in TRAIT_COLS[:5]) else None),
            (root / f"transcription_{word}.pkl", trans or None),
        ]
        for path, obj in targets:
            if obj and not any(root.rglob(path.name)):
                with open(path, "wb") as f:
                    pickle.dump(obj, f, protocol=2)
                written.append(path.name)
        dname = f"eth_gender_annotations_{'dev' if split == 'validation' else split}.csv"
        if demo and not any(root.rglob(dname)):
            with open(root / dname, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f, delimiter=";")
                w.writerow(["VideoName", "Ethnicity", "Gender"])
                for c, (g, e) in demo.items():
                    w.writerow([c, e, g])
            written.append(dname)
        log.info("[%s] metadata: %d Big Five labels, %d transcriptions, %d demographic rows",
                 split, n_lab, len(trans), len(demo))
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,validation,test")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config()
    log = get_logger("download", cfg)
    state = State(cfg)
    write_pid(cfg, "job")
    try:
        from huggingface_hub import HfApi, hf_hub_download
        root = Path(cfg["paths"]["dataset_root"])
        root.mkdir(parents=True, exist_ok=True)
        state.update(job="download", status="running", stage="Listing the Hugging Face repository", error=None,
                     started=time.time(), done=0, total=0)
        files = HfApi().list_repo_files(REPO, repo_type="dataset")
        splits = [s.strip() for s in args.splits.split(",")]
        have = {p.name for p in root.rglob("*.mp4")}
        videos = [f for f in files if f.endswith(".mp4") and f.split("/")[0] in splits and os.path.basename(f) not in have]
        others = [f for f in files if not f.endswith(".mp4") and not f.startswith(".git")]
        per = {s: sum(1 for f in files if f.endswith(".mp4") and f.split("/")[0] == s) for s in splits}
        log.info("Repository: %s | clips per split %s | already present %d | to download %d | metadata files %s",
                 REPO, per, len(have), len(videos), others)
        free = shutil.disk_usage(root).free / 2**30
        need = len(videos) * 3.0 / 1024
        log.info("Free disk space %.1f GB, estimated need %.1f GB", free, need)
        if free < need + 2:
            raise RuntimeError(f"Not enough disk space: {free:.1f} GB free, about {need:.1f} GB needed")
        hf_dir = root / "hf"
        for f in others:
            hf_hub_download(REPO, f, repo_type="dataset", local_dir=hf_dir)
        pr = Progress(state, "Downloading missing clips (Hugging Face)", len(videos))
        failed = []
        with ThreadPoolExecutor(args.workers) as ex:
            futs = {ex.submit(hf_hub_download, REPO, f, repo_type="dataset", local_dir=hf_dir): f for f in videos}
            for fu in as_completed(futs):
                try:
                    fu.result()
                except Exception as e:
                    failed.append(futs[fu])
                    log.warning("failed %s: %r", futs[fu], e)
                pr.step(msg=os.path.basename(futs[fu]))
        written = convert_metadata(root, log)
        names = {p.name for p in root.rglob("*")}
        missing = [n for n in ("annotation_validation.pkl", "annotation_test.pkl") if n not in names]
        msg = f"Download finished: {len(videos) - len(failed)} clips, {len(failed)} failures; new files: {written or 'none'}"
        if missing:
            msg += (". STILL MISSING: " + ", ".join(missing) +
                    " - these Big Five labels are not in the Hugging Face mirror; get them from the official "
                    "ChaLearn First Impressions V2 page (chalearnlap.cvc.uab.es) and copy them into Chalearn/.")
        log.info(msg)
        state.update(status="finished", stage=msg, finished=time.time())
    except Exception as ex:
        log.error(traceback.format_exc())
        state.update(status="error", error=repr(ex))
        raise
    finally:
        remove_pid(cfg, "job")


if __name__ == "__main__":
    main()
