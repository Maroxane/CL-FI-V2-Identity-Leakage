"""Gather ChaLearn First Impressions V2 into ONE clean folder and delete everything else.

    python -m sgmbf.organize_chalearn [--yes]

Final layout (paths.dataset_root, default SGMBF/Chalearn):
    Chalearn/
      annotation_training.pkl  annotation_validation.pkl  annotation_test.pkl
      transcription_training.pkl  transcription_validation.pkl  transcription_test.pkl
      (eth_gender_annotations_*.csv if present)
      train/  6000 .mp4      val/  2000 .mp4      test/  2000 .mp4

Videos are MOVED (no copy, no extra disk space). Nothing is deleted unless all 10,000 labelled videos are
found and the six .pkl files are in place; then the leftovers (hf/, official/, train-1..6/, duplicates,
unlabelled clips) and the whole 'Chalearn V2' folder (zips) are deleted.
"""
import pickle
import shutil
import sys
from pathlib import Path

from .common import ROOT, load_config

SPLITS = {"training": "train", "validation": "val", "test": "test"}


def _labels(pkl):
    with open(pkl, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    return set(next(iter(d.values())).keys())


def main():
    cfg = load_config()
    root = Path(cfg["paths"]["dataset_root"])
    v2 = Path(cfg["paths"].get("official_archives") or ROOT / "Chalearn V2")
    roots = [r for r in (root, v2) if r.exists()]
    yes = "--yes" in sys.argv

    # 1. metadata: official copies from Chalearn V2 take precedence
    meta = {}
    for r in reversed(roots):  # V2 last -> wins
        for pat in ("annotation_*.pkl", "transcription_*.pkl", "eth_gender_annotations_*.csv"):
            for p in r.rglob(pat):
                if "official" not in p.parts or p.parent == r:
                    meta[p.name] = p
    need = [f"{k}_{s}.pkl" for k in ("annotation", "transcription") for s in SPLITS]
    missing = [n for n in need if n not in meta]
    if missing:
        sys.exit(f"STOP: missing metadata files {missing} - nothing changed.")

    # 2. where is every labelled video?
    videos = {}
    for r in roots:
        for p in r.rglob("*.mp4"):
            if p.stat().st_size > 0:
                videos.setdefault(p.name, p)
    plan, report = [], []
    for s, folder in SPLITS.items():
        want = _labels(meta[f"annotation_{s}.pkl"])
        have = [v for v in want if v in videos]
        report.append(f"  {s:<10} {len(have):>5} / {len(want)}")
        plan += [(videos[v], root / folder / v) for v in have]
        if len(have) < len(want):
            print("\n".join(report))
            lost = sorted(set(want) - set(have))[:10]
            sys.exit(f"STOP: {len(want) - len(have)} {s} video(s) missing (e.g. {lost}) - nothing deleted.\n"
                     "Run extraire_archives.bat first.")
    print("Labelled videos found:\n" + "\n".join(report))

    keep_dirs = {root / f for f in SPLITS.values()}
    to_delete = [p for p in root.iterdir() if p.is_dir() and p not in keep_dirs]
    print("\nWill be deleted after moving the videos:")
    for p in to_delete:
        print("  ", p)
    if v2.exists() and v2 != root:
        print("  ", v2, "(archives .zip + copies of the .pkl, which are first copied into", root, ")")
    if not yes and input("\nType OUI to continue: ").strip().upper() != "OUI":
        sys.exit("Cancelled - nothing changed.")

    # 3. metadata into root
    for name, p in meta.items():
        if p.parent != root:
            shutil.copy2(p, root / name)
    # 4. move videos
    moved = 0
    for src, dst in plan:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src != dst:
            if dst.exists():
                dst.unlink()
            shutil.move(str(src), dst)
            moved += 1
    print(f"{moved} video(s) moved into {[str(d) for d in sorted(keep_dirs)]}")
    # 5. verify, then delete
    for s, folder in SPLITS.items():
        n = len(list((root / folder).glob("*.mp4")))
        want = len(_labels(root / f"annotation_{s}.pkl"))
        if n != want:
            sys.exit(f"STOP: {folder}/ holds {n} videos instead of {want} - nothing deleted.")
    for p in to_delete:
        shutil.rmtree(p, ignore_errors=True)
    for p in root.iterdir():  # stray files (old zips, .done markers, unlabelled clips)
        if p.is_file() and p.name not in meta:
            p.unlink()
    if v2.exists() and v2 != root:
        shutil.rmtree(v2, ignore_errors=True)
    free = shutil.disk_usage(root).free / 2**30
    print(f"\nDone. Single dataset folder: {root}")
    for f in SPLITS.values():
        print(f"  {f:<6} {len(list((root / f).glob('*.mp4')))} videos")
    print(f"  metadata: {sorted(p.name for p in root.glob('*.*'))}")
    print(f"Free disk space: {free:.0f} GB")


if __name__ == "__main__":
    main()
