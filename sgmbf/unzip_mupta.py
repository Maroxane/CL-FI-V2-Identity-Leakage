"""Extract the MuPTA corpus archives (audio/speaker_XX.zip, video/speaker_XX.zip).

    python -m sgmbf.unzip_mupta [--audio-only] [--shared] [--list]

- Output: Mupta/extracted/audio/speaker_XX/..., Mupta/extracted/video/speaker_XX/...
- Each archive is extracted once (a '.done' marker is written); the zips are kept.
- Shared.zip (54 GB) is only inspected, not extracted, unless --shared is given: it appears to bundle
  the same per-speaker archives, so extracting it would duplicate about 50 GB.
- Free disk space is checked before every archive.
"""
import argparse
import shutil
import zipfile
from collections import Counter
from pathlib import Path

from .common import load_config


def summarize(z):
    with zipfile.ZipFile(z) as f:
        infos = f.infolist()
    ext = Counter(Path(i.filename).suffix.lower() for i in infos if not i.is_dir())
    size = sum(i.file_size for i in infos) / 2**30
    tops = Counter(i.filename.split("/")[0] for i in infos)
    return len(infos), size, ext, tops


def extract(z, out, log=print):
    done = z.with_suffix(".zip.done")
    if done.exists():
        log(f"  {z.name}: already extracted")
        return True
    n, size, ext, _ = summarize(z)
    free = shutil.disk_usage(z.parent).free / 2**30
    if free < size + 2:
        log(f"  {z.name}: NOT ENOUGH DISK SPACE ({free:.1f} GB free, {size:.1f} GB needed). Stopping.")
        return False
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z) as f:
        f.extractall(out)
    done.write_text("ok")
    log(f"  {z.name}: {n} files, {size:.2f} GB {dict(ext)} -> {out}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-only", action="store_true")
    ap.add_argument("--shared", action="store_true", help="also extract Shared.zip")
    ap.add_argument("--list", action="store_true", help="only list archive contents")
    args = ap.parse_args()
    cfg = load_config()
    root = Path(cfg["paths"]["mupta_root"])
    dest = root / "extracted"
    print(f"MuPTA folder: {root}")
    shared = root / "Shared.zip"
    if shared.exists():
        n, size, ext, tops = summarize(shared)
        print(f"Shared.zip: {n} entries, {size:.1f} GB uncompressed, types {dict(ext)}, top folders {dict(tops)}")
        if args.shared and not args.list:
            extract(shared, dest / "shared")
    kinds = ["audio"] if args.audio_only else ["audio", "video"]
    for kind in kinds:
        zips = sorted((root / kind).glob("*.zip"))
        print(f"\n{kind}: {len(zips)} archive(s)")
        for z in zips:
            if args.list:
                n, size, ext, _ = summarize(z)
                print(f"  {z.name}: {n} files, {size:.2f} GB, {dict(ext)}")
            elif not extract(z, dest / kind / z.stem):
                return
    print("\nDone.")


if __name__ == "__main__":
    main()
