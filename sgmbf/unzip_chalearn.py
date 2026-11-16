"""Extract the official ChaLearn First Impressions V2 archives (password-protected, often nested).

    python -m sgmbf.unzip_chalearn [folder ...]

Archives are searched in the dataset folder (config paths.dataset_root) and in paths.official_archives
(default 'Chalearn V2'). Everything is extracted into <dataset_root>/official/<archive name>/ with the
passwords published on the ChaLearn pages; nested zips (e.g. test80_01.zip) are extracted too.
Videos already present anywhere in the dataset folder are skipped, so nothing is duplicated.
Archives already processed are marked with a '.done' file.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from .common import ROOT, load_config

def _passwords():
    """Archive keys published on the ChaLearn download pages (not redistributed here): one per line in
    chalearn_passwords.txt next to config.yaml, or in the CHALEARN_PASSWORDS environment variable (';'-separated)."""
    import os
    from .common import ROOT
    keys = [k for k in os.environ.get("CHALEARN_PASSWORDS", "").split(";") if k]
    f = ROOT / "chalearn_passwords.txt"
    if f.exists():
        keys += [l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [None] + [k.encode() for k in keys]


PASSWORDS = _passwords()


def _pwd_for(f):
    """First password that opens the first encrypted member (None if not encrypted)."""
    enc = [i for i in f.infolist() if i.flag_bits & 0x1 and not i.is_dir()]
    if not enc:
        return None
    for pw in PASSWORDS[1:]:
        try:
            with f.open(enc[0], pwd=pw) as h:
                h.read(1)
            return pw
        except RuntimeError:
            continue
    raise RuntimeError("no known password")


def _seven_zip():
    """7-Zip decrypts ZipCrypto ~50x faster than Python's pure-Python implementation."""
    for c in [shutil.which("7z"), shutil.which("7za"),
              os.path.expandvars(r"%ProgramFiles%\7-Zip\7z.exe"), os.path.expandvars(r"%ProgramFiles(x86)%\7-Zip\7z.exe")]:
        if c and Path(c).exists():
            return c
    return None


SEVEN = _seven_zip()


def extract(z, out, have, log=print):
    done = Path(str(z) + ".done")
    if done.exists():
        return []
    with zipfile.ZipFile(z) as f:
        try:
            pw = _pwd_for(f)
        except RuntimeError:
            log(f"  {z.name}: no known password worked - skipped")
            return []
        except NotImplementedError:
            log(f"  {z.name}: encryption not supported by Python, extract it with 7-Zip")
            return []
        members = [i for i in f.infolist() if not i.is_dir()
                   and not (i.filename.lower().endswith(".mp4") and Path(i.filename).name in have)]
        need = sum(i.file_size for i in members) / 2**30
        free = shutil.disk_usage(z.parent).free / 2**30
        if free < need + 2:
            raise SystemExit(f"  {z.name}: NOT ENOUGH DISK SPACE ({free:.1f} GB free, {need:.1f} GB needed)")
        skipped = len(f.infolist()) - len(members)
        if SEVEN and pw and members:
            out.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as lf:
                lf.write("\n".join(i.filename for i in members))
            r = subprocess.run([SEVEN, "x", f"-p{pw.decode()}", f"-o{out}", "-y", "-aoa", "-bso0", "-bsp0",
                                str(z), f"@{lf.name}"])
            os.unlink(lf.name)
            if r.returncode != 0:
                raise zipfile.BadZipFile(f"7-Zip failed on {z.name}")
        else:
            for i in members:
                f.extract(i, out, pwd=pw)
        for i in members:
            if i.filename.lower().endswith(".mp4"):
                have.add(Path(i.filename).name)
    done.write_text("ok")
    log(f"  {z.name}: {len(members)} file(s) extracted -> {out}" + (f", {skipped} already present" if skipped else ""))
    return sorted(out.rglob("*.zip"))


def main():
    cfg = load_config()
    root = Path(cfg["paths"]["dataset_root"])
    extra = cfg["paths"].get("official_archives") or str(ROOT / "Chalearn V2")
    folders = [Path(a) for a in sys.argv[1:]] or [root, Path(extra)]
    have = {p.name for d in folders if d.exists() for p in d.rglob("*.mp4")}
    todo = [z for d in folders if d.exists() for z in sorted(d.rglob("*.zip"))]
    # first-level bundles (train-1.zip, val-1.zip, test-1e.zip, *-annotation*.zip ...) are skipped when they
    # were already unpacked by hand, i.e. when their folder already holds the annotation files
    bundle = lambda z: re.match(r"^(train|val|test)-", z.name) and any(z.parent.glob("annotation_*.pkl"))
    skipped = [z.name for z in todo if bundle(z)]
    todo = [z for z in todo if not bundle(z)]
    for name in skipped:  # bundle copies left by an older run: duplicates, removed
        old = root / "official" / Path(name).stem
        if old.exists():
            shutil.rmtree(old, ignore_errors=True)
            print(f"  removed duplicate folder {old}")
    todo = [z for z in todo if z.exists()]
    if skipped:
        print(f"{len(skipped)} first-level archive(s) already unpacked by hand, skipped: {', '.join(skipped)}")
    # parents first; nested leftovers of an interrupted run are recreated by their parent anyway
    todo.sort(key=lambda z: str(z).startswith(str(root / "official")))
    print("7-Zip:", SEVEN or "not found (Python decryption, slow - install 7-Zip from https://www.7-zip.org to go ~50x faster)")
    print(f"{len(todo)} archive(s) found in {[str(d) for d in folders if d.exists()]}; {len(have)} videos already present")
    seen = set()
    while todo:
        z = todo.pop(0)
        if z in seen:
            continue
        seen.add(z)
        nested = str(z).startswith(str(root / "official"))
        dest = z.parent / z.stem if nested else root / "official" / z.stem
        try:
            todo += [n for n in extract(z, dest, have) if n not in seen]
        except zipfile.BadZipFile:
            if nested:  # truncated leftover of an interrupted run: removed, its parent archive recreates it
                print(f"  {z.name}: incomplete leftover from an interrupted run - removed")
                z.unlink(missing_ok=True)
            else:
                print(f"  {z.name}: NOT A VALID ZIP (incomplete download?) - skipped, re-download it")
            continue
        if str(z).startswith(str(root / "official")):  # nested copy already unpacked: free the space
            z.unlink(missing_ok=True)
            Path(str(z) + ".done").unlink(missing_ok=True)
    print("\nDone.")
    ex = [d for d in folders if d.exists()]
    print("Videos available:", len({p.name for d in ex for p in d.rglob("*.mp4")}))
    print("Annotation files:", sorted({p.name for d in ex for p in d.rglob("annotation_*.pkl")}))
    print("Transcription files:", sorted({p.name for d in ex for p in d.rglob("transcription_*.pkl")}))


if __name__ == "__main__":
    main()
