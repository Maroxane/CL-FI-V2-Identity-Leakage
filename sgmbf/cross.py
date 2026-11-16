"""Cross-dataset evaluation: models trained on ChaLearn First Impressions V2, tested on MuPTA.

    python -m sgmbf.cross

Settings (labels are per speaker, so predictions are averaged over each speaker's clips):
  zero-shot                    FI models and FI feature statistics, no MuPTA label used
  zero-shot, no text           same, text branch removed (gate set to 0 / input zeroed)
  zero-shot, MuPTA z-scoring   features re-standardised with MuPTA statistics (unsupervised)
  recalibrated (LOSO)          per-trait linear recalibration fitted on the other speakers
  few-shot head (LOSO)         prediction head fine-tuned on the other 23 speakers (proj. + gate frozen)
Outputs: results/cross_mupta.csv, results/mupta_gates.csv, results/mupta_metadata_test_predictions.csv
"""
import copy
import csv
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .common import TRAITS, Progress, State, get_logger, load_config, remove_pid, write_pid
from .features import MODALITY_BLOCKS, Standardizer, assemble, load_blocks
from .models import build
from .mupta import mupta_cfg, scan
from .train import predict, set_seed, to_dev

FULL = {m: list(b) for m, b in MODALITY_BLOCKS.items()}
FUSION_RUNS = {"SG-MBF": ("main", "SG-MBF"), "Early concat": ("fusion", "Early_concat"),
               "Late average": ("fusion", "Late_average"), "Softmax attention": ("fusion", "Softmax_attention"),
               "GMU": ("fusion", "GMU"), "Token transformer": ("fusion", "Token_transformer"),
               "Ungated sum": ("fusion", "Ungated_sum"), "Ungated sum + LN": ("fusion", "Ungated_sum_p_LN")}


# ------------------------------------------------------------------ metrics at speaker level
def speaker_means(pred, clips, spk_of):
    spk = sorted({spk_of[c] for c in clips})
    idx = {s: [i for i, c in enumerate(clips) if spk_of[c] == s] for s in spk}
    return spk, np.stack([pred[idx[s]].mean(0) for s in spk])


def trait_metrics(y, p):
    out = {}
    for j, t in enumerate(TRAITS):
        a, b = y[:, j], p[:, j]
        r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
        cov = np.mean((a - a.mean()) * (b - b.mean()))
        ccc = float(2 * cov / (a.var() + b.var() + (a.mean() - b.mean()) ** 2 + 1e-12))
        out[t] = {"r": r, "ccc": ccc, "mae": float(np.mean(np.abs(a - b)))}
    out["mean"] = {k: float(np.mean([out[t][k] for t in TRAITS])) for k in ("r", "ccc", "mae")}
    return out


def boot_mean_ccc(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        vals.append(trait_metrics(y[i], p[i])["mean"]["ccc"])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ------------------------------------------------------------------ helpers
def load_fi_models(cfg, name):
    group, prefix = FUSION_RUNS[name]
    out = []
    for p in sorted((Path(cfg["paths"]["runs"]) / group).glob(f"{prefix}__TAV__s*/model.pt")):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = build(ck["name"], ck["dims"], cfg, ck.get("training", {}).get("gate_variant", "sigmoid"))
        m.load_state_dict(ck["state_dict"])
        out.append((int(p.parent.name.split("__s")[-1]), m, Standardizer.from_state(ck["standardizer"])))
    return out


def run_model(m, X, dev, mask=None):
    m = m.to(dev).eval()
    mm = None if mask is None else torch.as_tensor(np.tile(mask, (len(next(iter(X.values()))), 1)), device=dev)
    return predict(m, to_dev(X, dev), modality_mask=mm)


def finetune_head(model, X, y, cfg, seed, dev):
    """Fine-tune only the prediction head (projections and gate frozen)."""
    set_seed(seed)
    m = copy.deepcopy(model).to(dev)
    for p in m.parameters():
        p.requires_grad = False
    for p in m.head.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(m.head.parameters(), lr=cfg["mupta"]["fewshot_lr"], weight_decay=cfg["training"]["weight_decay"])
    Xd, yd = to_dev(X, dev), torch.as_tensor(y, device=dev)
    n = len(y)
    for _ in range(cfg["mupta"]["fewshot_epochs"]):
        m.train()
        for mod in list(m.proj.values()):
            mod.eval()  # keep BatchNorm statistics of the frozen projections
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, 64):
            idx = perm[i:i + 64]
            if len(idx) < 2:
                continue
            p, _ = m({k: v[idx] for k, v in Xd.items()})
            loss = F.mse_loss(p.float(), yd[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return m.eval()


# ------------------------------------------------------------------ main
def main():
    cfg = load_config()
    log = get_logger("cross", cfg)
    state = State(cfg)
    write_pid(cfg, "job")
    try:
        state.update(job="cross_mupta", status="running", stage="Cross-dataset: loading MuPTA features",
                     error=None, started=time.time())
        res = Path(cfg["paths"]["results"])
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        info = scan(cfg)
        mc = mupta_cfg(cfg)
        blocks, clips = load_blocks(mc, info["clips"])
        dur = np.array([float(np.load(Path(mc["paths"]["cache"]) / "audio" / f"{c}.npz")["dur"]) for c in clips])
        keep = dur >= cfg["mupta"]["min_duration_s"]
        blocks = {k: v[keep] for k, v in blocks.items()}
        clips = [c for c, k in zip(clips, keep) if k]
        dur = dur[keep]
        spk_of = info["speaker"]
        inv = cfg["mupta"]["invert_neuroticism"]

        def lab(s):
            v = np.array([info["labels"][s][t] for t in TRAITS], np.float32)
            if inv:
                v[4] = 1 - v[4]
            return v

        labelled = np.array([spk_of[c] in info["labels"] for c in clips])
        L_clips = [c for c, l in zip(clips, labelled) if l]
        L_blocks = {k: v[labelled] for k, v in blocks.items()}
        y_clip = np.stack([lab(spk_of[c]) for c in L_clips])
        spont = dur[labelled] >= cfg["mupta"]["spontaneous_min_s"]
        speakers = sorted({spk_of[c] for c in L_clips})
        log.info("MuPTA: %d clips kept (>= %.1f s), %d labelled clips, %d labelled speakers",
                 len(clips), cfg["mupta"]["min_duration_s"], len(L_clips), len(speakers))
        rows, gate_rows = [], []

        def record(setting, model, seed, pred, subset_mask=None, sub="all"):
            cl = [c for c, k in zip(L_clips, subset_mask if subset_mask is not None else np.ones(len(L_clips), bool)) if k]
            P = pred if subset_mask is None else pred[subset_mask]
            Yc = y_clip if subset_mask is None else y_clip[subset_mask]
            spk, Ps = speaker_means(P, cl, spk_of)
            Ys = np.stack([lab(s) for s in spk])
            for level, (yy, pp) in (("speaker", (Ys, Ps)), ("clip", (Yc, P))):
                m = trait_metrics(yy, pp)
                for t in TRAITS + ["mean"]:
                    rows.append([setting, model, seed, level, sub, t, m[t]["r"], m[t]["ccc"], m[t]["mae"], len(yy)])
            return Ps, spk

        # ---- zero-shot for every fusion strategy trained on FI
        pr = Progress(state, "Cross-dataset: zero-shot", len(FUSION_RUNS))
        sgmbf = load_fi_models(cfg, "SG-MBF")
        if not sgmbf:
            raise RuntimeError("No SG-MBF model trained on First Impressions (runs/main). Run step 2 first.")
        mupta_std = Standardizer().fit(assemble(blocks, FULL))  # unsupervised: all MuPTA clips, no labels
        zs_speaker_preds = {}
        for name in FUSION_RUNS:
            for seed, m, std in (sgmbf if name == "SG-MBF" else load_fi_models(cfg, name)):
                X = std(assemble(L_blocks, FULL))
                p, g, nrm = run_model(m, X, dev)
                Ps, spk = record("zero-shot", name, seed, p)
                record("zero-shot", name, seed, p, spont, "spontaneous (>= 4 s)")
                if name == "SG-MBF":
                    zs_speaker_preds[seed] = (Ps, spk)
                    gate_rows.append(["MuPTA zero-shot", seed, *g.mean(0), *(g * nrm).mean(0)])
                    pt, _, _ = run_model(m, X, dev, mask=np.array([0, 1, 1], np.float32))
                    record("zero-shot, no text", name, seed, pt)
                    pz, gz, _ = run_model(m, mupta_std(assemble(L_blocks, FULL)), dev)
                    record("zero-shot, MuPTA z-scoring", name, seed, pz)
            pr.step(msg=name)

        # FI test gates for comparison
        for seed, _, _ in sgmbf:
            rd = Path(cfg["paths"]["runs"]) / "main" / f"SG-MBF__TAV__s{seed}"
            if (rd / "test_gates.npy").exists():
                g, nrm = np.load(rd / "test_gates.npy"), np.load(rd / "test_norms.npy")
                gate_rows.append(["FI V2 test", seed, *g.mean(0), *(g * nrm).mean(0)])

        # ---- LOSO linear recalibration of zero-shot speaker predictions
        for seed, (Ps, spk) in zs_speaker_preds.items():
            Ys = np.stack([lab(s) for s in spk])
            out = np.zeros_like(Ps)
            for i in range(len(spk)):
                tr = np.arange(len(spk)) != i
                for j in range(5):
                    a, b = np.polyfit(Ps[tr, j], Ys[tr, j], 1)
                    out[i, j] = a * Ps[i, j] + b
            m = trait_metrics(Ys, out)
            for t in TRAITS + ["mean"]:
                rows.append(["recalibrated (LOSO)", "SG-MBF", seed, "speaker", "all", t, m[t]["r"], m[t]["ccc"], m[t]["mae"], len(spk)])

        # ---- few-shot: head fine-tuned on the other speakers (leave-one-speaker-out)
        pr = Progress(state, "Cross-dataset: few-shot LOSO fine-tuning", len(sgmbf) * len(speakers))
        spk_arr = np.array([spk_of[c] for c in L_clips])
        for seed, m, std in sgmbf:
            X = std(assemble(L_blocks, FULL))
            pred = np.zeros_like(y_clip)
            for s in speakers:
                tr, te = spk_arr != s, spk_arr == s
                ft = finetune_head(m, {k: v[tr] for k, v in X.items()}, y_clip[tr], cfg, seed, dev)
                pred[te] = run_model(ft, {k: v[te] for k, v in X.items()}, dev)[0]
                pr.step(msg=f"seed {seed} / {s}")
            record("few-shot head (LOSO)", "SG-MBF", seed, pred)

        # ---- write tables
        with open(res / "cross_mupta.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["setting", "model", "seed", "level", "subset", "trait", "r", "ccc", "mae", "n"])
            w.writerows(rows)
        with open(res / "mupta_gates.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["data", "seed", "g_t", "g_a", "g_v", "contrib_t", "contrib_a", "contrib_v"])
            w.writerows(gate_rows)

        # ---- predictions for the confidential MuPTA test speakers (submission file)
        if info["test_rows"]:
            state.update(stage="Cross-dataset: predictions for the MuPTA test set")
            preds = []
            for seed, m, std in sgmbf:
                ft = finetune_head(m, std(assemble(L_blocks, FULL)), y_clip, cfg, seed, dev)
                preds.append(run_model(ft, std(assemble(blocks, FULL)), dev)[0])
            P = dict(zip(clips, np.mean(preds, 0)))
            out = res / "mupta_metadata_test_predictions.csv"
            with open(out, "w", newline="", encoding="utf-8") as f:
                cols = list(info["test_rows"][0].keys())
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for r in info["test_rows"]:
                    key = f"{r['speaker_id'].strip()}_{Path(r['name_file']).stem}"
                    r = dict(r)
                    if key in P:
                        v = np.clip(P[key], 0, 1)
                        for j, t in enumerate(TRAITS):
                            col = {"O": "openness", "C": "conscientiousness", "E": "extraversion",
                                   "A": "agreeableness", "N": "neuroticism"}[t]
                            r[col] = f"{(1 - v[j]) if (t == 'N' and inv) else v[j]:.6f}"
                    w.writerow(r)
            log.info("MuPTA test predictions written to %s", out)

        from .experiments import Ctx, summary
        try:
            summary(Ctx(cfg, log, state))
        except Exception as ex:
            log.warning("summary not refreshed: %r", ex)
        state.update(status="finished", stage="Cross-dataset evaluation on MuPTA finished", finished=time.time())
    except Exception as ex:
        log.error(traceback.format_exc())
        state.update(status="error", error=repr(ex))
        raise
    finally:
        remove_pid(cfg, "job")


def md_lines(res):
    """Markdown table for results/tables.md (speaker level, mean over seeds)."""
    p = Path(res) / "cross_mupta.csv"
    if not p.exists():
        return []
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    key = lambda r: (r["setting"], r["model"], r["subset"])
    groups = {}
    for r in rows:
        if r["level"] == "speaker":
            groups.setdefault(key(r), {}).setdefault(r["trait"], []).append(r)
    L = ["", "## Table 8 - cross-dataset: FI V2 -> MuPTA (speaker level, CCC mean ± SD over seeds; MAE)", "",
         "| Setting | Model | Subset | " + " | ".join(f"CCC {t}" for t in TRAITS) + " | Mean CCC | Mean r | Mean MAE |",
         "|---|---|---|" + "---|" * 8]
    for (s, m, sub), d in groups.items():
        cell = lambda t, k: np.array([float(x[k]) for x in d[t]])
        f = lambda v: f"{v.mean():.3f} ± {v.std(ddof=1) if len(v) > 1 else 0:.3f}"
        L.append(f"| {s} | {m} | {sub} | " + " | ".join(f(cell(t, 'ccc')) for t in TRAITS) +
                 f" | {f(cell('mean', 'ccc'))} | {f(cell('mean', 'r'))} | {cell('mean', 'mae').mean():.3f} |")
    return L


if __name__ == "__main__":
    main()
