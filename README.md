# Identity leakage in ChaLearn First Impressions V2

Code, splits and results for the paper *"Identity Leakage in ChaLearn First Impressions V2: What Multimodal Apparent
Personality Recognition Benchmarks Actually Measure"* (under review).
Repository: https://github.com/Maroxane/CL-FI-V2-Identity-Leakage

The official partition of First Impressions V2 was drawn over clips, so clips of the same person sit in training and
in test: 84.0% of the test clips share their YouTube video with training clips. This repository quantifies the effect,
provides verified identity-disjoint splits and re-evaluates 14 models on them.

## Main results (mean CCC over the five traits)

| Model | Official split | Random clip-level splits (10 draws) | Identity-disjoint splits (10 draws) | Same model: seen / unseen persons (5 draws) |
|---|---|---|---|---|
| Same-video lookup (no input) | 0.601 | 0.604 | 0.000 | 0.669 / 0.000 |
| Text only | 0.171 | 0.184 | 0.162 | 0.182 / 0.165 |
| Audio only | 0.521 | 0.525 | 0.408 | 0.489 / 0.404 |
| Visual only | 0.600 | 0.603 | 0.493 | 0.593 / 0.478 |
| Audio + visual | 0.640 | 0.656 | 0.574 | 0.652 / 0.565 |
| GMU (best three-modality fusion) | 0.628 | 0.638 | 0.571 | 0.641 / 0.561 |
| SG-MBF (shared gate) | 0.621 | 0.629 | 0.565 | 0.628 / 0.554 |
| Same model without the gate | 0.618 | 0.630 | 0.566 | 0.623 / 0.553 |

Full tables: `results/leakage.md`, `results/control.md`, `results/phase2.md`.

## Splits and identity table (`splits/`)

| File | Content |
|---|---|
| `identities.json` | clip name -> anonymous identity label (2,876 identities; face + voice clustering, manually checked) |
| `identities_single_link.json` | first grouping (connected components, over-merged), kept for the sensitivity analysis |
| `S0R_k.json` (k = 0..9) | random clip-level splits (leaky, same construction as the official split) |
| `S1_k.json` (k = 0..2) | video-disjoint splits |
| `S2A_k.json` (k = 0..9) | **identity-disjoint splits (recommended)** |
| `S2_k.json` (k = 0..2) | identity-disjoint splits built from the first grouping |
| `S3_k.json` (k = 0..4) | control: `S2A_k` with 600 training clips held out as a "seen person" test set |

Each file holds `{"splits": {"train": [...], "val": [...], "test": [...]}}` with 6,000 / 2,000 / 2,000 clip names
(5,400 / 2,000 / 2,600 for `S3_k`). No video, image or embedding is distributed.

## Repository layout

The Python package is named `sgmbf` after the shared-gate fusion model (SG-MBF) from which the study started;
the model is one of the fourteen evaluated in the paper.

```
sgmbf/            Python package
  extract.py      frozen encoders -> one .npz per clip (text, audio, visual)
  models.py       the eight fusion strategies (incl. the shared-gate model SG-MBF and its gate-free controls)
  identity.py     face (FaceNet) + voice (WavLM-SV) embeddings, identity grouping, split construction
  leakage.py      audit: 14 models x 5 seeds on official, random, video-disjoint and identity-disjoint splits
  control.py      same-model control: seen vs unseen persons
  phase2.py       robustness, fairness, representation probe and MuPTA transfer with the audit models
  experiments.py  earlier experiments on the official split (main, fusion, ablation, robustness, fairness)
  mupta.py, cross.py   MuPTA feature extraction and cross-corpus evaluation
web/              dashboard started by start.bat (port 8001) and demo interface (port 8002)
launch.py         starts and stops the two web servers (used by start.bat / stop.bat)
config.yaml       every hyper-parameter and path
splits/           identity table and all splits
results/          result tables of the paper (leakage, control, phase2) and the gate-variant selection
```

## Installation

Tested on Windows 11, Python 3.11, PyTorch 2.5.1 + CUDA 12.4, NVIDIA RTX 3060 Laptop (6 GB).

**Windows:** double-click `start.bat`. It creates `.venv`, installs PyTorch and the requirements, then opens the
dashboard at http://127.0.0.1:8001, where each step of the pipeline (feature extraction, identities, leakage audit,
control, secondary analyses) is one button. `stop.bat` stops everything; finished steps are cached and resumed.

**Any OS:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install "pyarrow==17.0.0" --no-deps --force-reinstall   # mediapipe 0.10.14 needs numpy<2
```

## Data (not included)

* **ChaLearn First Impressions V2** — request access at https://chalearnlap.cvc.uab.cat/dataset/24/description/
  (CC BY-NC 4.0). Place videos and the official files in `Chalearn/`:
  `annotation_{training,validation,test}.pkl`, `transcription_{training,validation,test}.pkl`,
  `eth_gender_annotations_{dev,test}.csv`. `python -m sgmbf.organize_chalearn` checks that
  all 10,000 labelled clips are present and arranges them into `train/`, `val/`, `test/`.
* **MuPTA** (Ryumina et al., Interspeech 2023) — obtained from its authors; place it in `Mupta/`
  (`video/`, `audio/`, `metadata/`). Only the central camera is used (`mupta.views`).


## Reproducing the paper

```bash
python -m sgmbf.extract        # features (cached, resumable; about one day on an RTX 3060 laptop)
python -m sgmbf.identity       # identities and splits (skip to reuse the released splits/)
python -m sgmbf.leakage        # audit (results/leakage.md, results/leakage/summary.csv)
python -m sgmbf.control        # same-model control (results/control.md)
python -m sgmbf.phase2         # secondary analyses (results/phase2.md)
```

`sgmbf.identity` needs `facenet-pytorch` (install with `--no-deps`) and downloads `microsoft/wavlm-base-plus-sv`.
The audit trains 2,240 small models on cached features; finished runs are reused.

## Citation

```bibtex
@article{echhaimi2026leakage,
  title   = {Identity Leakage in ChaLearn First Impressions V2: What Multimodal Apparent Personality Recognition Benchmarks Actually Measure},
  author  = {Echhaimi, Marouane and Lekdioui, Khadija and Chaabi, Youness and Aziz, Oussama and Boujiha, Tarik},
  year    = {2026},
  note    = {Under review}
}
```

Please also cite the corpus (Ponce-López et al., 2016; Escalante et al., 2022); the perceived gender and ethnicity
annotations were provided by H. Kaya and A. A. Salah.

## License

Code: MIT (see `LICENSE`). Datasets remain under the terms of their respective owners.
