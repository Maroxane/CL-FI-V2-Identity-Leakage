"""Stage 2 - all experiments of the article, resumable (finished runs are skipped).

    python -m sgmbf.experiments --plan main|fusion|ablation|robustness|fairness|all
Outputs in results/: seeds.csv, fusion_seeds.csv, components.csv, robustness.csv, gates_test.csv,
fairness.csv, tables.md, published.csv (template) - the inputs of make_results_figures.py.
"""
import argparse
import csv
import json
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .common import ROOT, TRAITS, Progress, State, get_logger, load_config, remove_pid, write_pid
from .data import audit, label_matrix, load_demographics, make_splits, scan
from .features import MODALITY_BLOCKS, Standardizer, assemble, load_blocks
from .metrics import bootstrap_ci, holm, paired_bootstrap, per_trait
from .train import predict, to_dev, train_one

FULL = {m: list(b) for m, b in MODALITY_BLOCKS.items()}
MOD_SUBSETS = {"T+A": "ta", "T+V": "tv", "A+V": "av", "T": "t", "A": "a", "V": "v"}
COMPONENTS = {
    "T: BERT": ("t", ["bert"]), "T: Sentence-BERT": ("t", ["sbert"]), "T: Empath": ("t", ["empath"]),
    "T: BERT + Sentence-BERT": ("t", ["bert", "sbert"]), "T: BERT + Empath": ("t", ["bert", "empath"]),
    "A: VGGish": ("a", ["vggish"]), "A: eGeMAPS": ("a", ["egemaps"]),
    "V: Geometry": ("v", ["geo"]), "V: Emo-AffectNet": ("v", ["emo"]), "V: Pose kinematics": ("v", ["pose"]),
}
BASELINES = ["Early concat", "Late average", "Softmax attention", "GMU", "Token transformer", "Ungated sum",
             "Ungated sum + LN"]


class Ctx:
    def __init__(self, cfg, log, state):
        self.cfg, self.log, self.state = cfg, log, state
        self.info = scan(cfg)
        self.splits, self.kind = make_splits(cfg, self.info, log)
        audit(cfg, self.info, self.splits, self.kind)
        self.blocks, self.clips, self.y = {}, {}, {}
        for s in ("train", "val", "test"):
            b, kept = load_blocks(cfg, self.splits[s])
            if len(kept) < len(self.splits[s]):
                log.warning("%s: %d/%d clips have all modalities cached", s, len(kept), len(self.splits[s]))
            self.blocks[s], self.clips[s] = b, kept
            self.y[s] = label_matrix(self.info, kept)
        self.res = Path(cfg["paths"]["results"])
        self.runs = Path(cfg["paths"]["runs"])
        state.update(split_kind=self.kind, n_used={s: len(self.clips[s]) for s in self.clips})
        (self.res / "test_clips.json").write_text(json.dumps(self.clips["test"]))

    def data(self, spec, test_blocks=None):
        X = {s: assemble(self.blocks[s], spec) for s in ("train", "val", "test")}
        if test_blocks is not None:
            X["test"] = assemble(test_blocks, spec)
        std = Standardizer().fit(X["train"])
        return {s: std(X[s]) for s in X}, std


def run(ctx, group, name, label, spec, seed):
    """Train (or reuse) one run; returns (run_dir, test predictions)."""
    rd = ctx.runs / group / f"{name}__{label}__s{seed}".replace(" ", "_").replace(":", "").replace("+", "p")
    if (rd / "test_pred.npy").exists():
        return rd, np.load(rd / "test_pred.npy")
    X, std = ctx.data(spec)
    ctx.state.update(stage=f"Training {group}: {name} [{label}] seed {seed}")
    model, meta = train_one(ctx.cfg, name, X["train"], ctx.y["train"], X["val"], ctx.y["val"], seed, rd,
                            ctx.state, label, {"group": group, "spec": spec, "split_kind": ctx.kind})
    dev = next(model.parameters()).device
    pred, gates, norms = predict(model, to_dev(X["test"], dev))
    np.save(rd / "test_pred.npy", pred)
    if gates is not None:
        np.save(rd / "test_gates.npy", gates)
    if norms is not None:
        np.save(rd / "test_norms.npy", norms)
    m = per_trait(ctx.y["test"], pred)
    meta.update(test=m)
    (rd / "meta.json").write_text(json.dumps(meta, indent=1))
    torch.save({"state_dict": model.state_dict(), "name": name, "dims": meta["dims"], "spec": spec,
                "standardizer": std.state(), "training": ctx.cfg["training"], "label": label},
               rd / "model.pt")
    ctx.log.info("%s | %s | seed %d | test acc %.4f ccc %.4f | best epoch %d", name, label, seed,
                 m["mean"]["acc"], m["mean"]["ccc"], meta["best_epoch"])
    return rd, pred


SEED_HDR = ["model", "seed", "trait", "acc", "ccc", "r", "r2"]


def write_rows(path, rows, header=None):
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header or SEED_HDR)
        w.writerows(rows)


def drop_models(path, names):
    """Remove previous rows of these models (so re-running a plan never duplicates rows)."""
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([rows[0]] + [r for r in rows[1:] if r and r[0] not in names])


def seed_rows(model_label, seed, y, pred):
    m = per_trait(y, pred)
    return [[model_label, seed, t, m[t]["acc"], m[t]["ccc"], m[t]["r"], m[t]["r2"]] for t in TRAITS]


def reset(path):
    if path.exists():
        path.unlink()


# ------------------------------------------------------------------ plans
def plan_main(ctx):
    seeds = ctx.cfg["training"]["seeds"]
    out = {}
    for s in seeds:
        rd, p = run(ctx, "main", "SG-MBF", "TAV", FULL, s)
        out[s] = (rd, p)
    drop_models(ctx.res / "seeds.csv", {"Ours"})
    for s, (rd, p) in out.items():
        write_rows(ctx.res / "seeds.csv", seed_rows("Ours", s, ctx.y["test"], p),
                   ["model", "seed", "trait", "acc", "ccc", "r", "r2"])
    rd1 = out[seeds[0]][0]
    g, nrm = np.load(rd1 / "test_gates.npy"), np.load(rd1 / "test_norms.npy")
    with open(ctx.res / "gates_test.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id"] + TRAITS + ["g_t", "g_a", "g_v", "norm_t", "norm_a", "norm_v"])
        for i, c in enumerate(ctx.clips["test"]):
            w.writerow([c] + list(ctx.y["test"][i]) + list(g[i]) + list(nrm[i]))
    tpl = ROOT / "results_templates" / "published.csv"
    if tpl.exists() and not (ctx.res / "published.csv").exists():
        shutil.copy(tpl, ctx.res / "published.csv")
    return out


def plan_fusion(ctx):
    seeds = ctx.cfg["training"]["seeds"]
    ours = [run(ctx, "main", "SG-MBF", "TAV", FULL, s)[1] for s in seeds]
    reset(ctx.res / "fusion_seeds.csv")
    for s, p in zip(seeds, ours):
        write_rows(ctx.res / "fusion_seeds.csv", seed_rows("SG-MBF", s, ctx.y["test"], p),
                   ["model", "seed", "trait", "acc", "ccc", "r", "r2"])
    for name in BASELINES:
        for s in seeds:
            _, p = run(ctx, "fusion", name, "TAV", FULL, s)
            write_rows(ctx.res / "fusion_seeds.csv", seed_rows(name, s, ctx.y["test"], p))


def plan_ablation(ctx):
    seeds = ctx.cfg["training"]["seeds"]
    drop_models(ctx.res / "seeds.csv", set(MOD_SUBSETS))
    for lab, mods in MOD_SUBSETS.items():
        spec = {m: FULL[m] for m in mods}
        for s in seeds:
            _, p = run(ctx, "ablation", "SG-MBF", lab, spec, s)
            write_rows(ctx.res / "seeds.csv", seed_rows(lab, s, ctx.y["test"], p))
    reset(ctx.res / "components.csv")
    for lab, (m, bl) in COMPONENTS.items():
        spec = {k: (bl if k == m else FULL[k]) for k in FULL}
        for s in seeds:
            _, p = run(ctx, "components", "SG-MBF", lab, spec, s)
            write_rows(ctx.res / "components.csv", seed_rows(lab, s, ctx.y["test"], p),
                       ["model", "seed", "trait", "acc", "ccc", "r", "r2"])


def _load_main_models(ctx, group="main", prefix="SG-MBF"):
    models = []
    for s in ctx.cfg["training"]["seeds"]:
        rd = ctx.runs / group / f"{prefix}__TAV__s{s}"
        ck = torch.load(rd / "model.pt", map_location="cpu", weights_only=False)
        from .models import build
        m = build(ck["name"], ck["dims"], ctx.cfg, ck.get("training", {}).get("gate_variant", "sigmoid"))
        m.load_state_dict(ck["state_dict"])
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        models.append((s, m.to(dev).eval(), Standardizer.from_state(ck["standardizer"]), dev))
    return models


def _eval_models(ctx, models, test_blocks, mask=None):
    rows = []
    for s, m, std, dev in models:
        X = std(assemble(test_blocks, FULL))
        mm = None if mask is None else torch.as_tensor(np.tile(mask, (len(ctx.y["test"]), 1)),
                                                       device=dev, dtype=torch.float32)
        p, g, _ = predict(m, to_dev(X, dev), modality_mask=mm)
        rows.append((s, per_trait(ctx.y["test"], p)["mean"]["acc"], g.mean(0)))
    return rows


ROBUST_CONTROL = ("fusion", "Ungated_sum_p_LN", "robustness_control.csv")  # same model without the gate


def plan_robustness(ctx):
    """SG-MBF and its no-gate control under the same perturbations: shows whether the gate helps when a
    modality is degraded (the accuracy tables cannot show it on clean data)."""
    from .extractors import AudioEncoder, free_gpu
    from .media import add_noise, load_audio
    ev = ctx.cfg["evaluation"]
    sets = [("robustness.csv", _load_main_models(ctx))]
    g, pre, fname = ROBUST_CONTROL
    if (ctx.runs / g / f"{pre}__TAV__s{ctx.cfg['training']['seeds'][0]}" / "model.pt").exists():
        sets.append((fname, _load_main_models(ctx, g, pre)))
    hdr = ["perturbation", "level", "seed", "acc", "g_t", "g_a", "g_v"]
    for f, _ in sets:
        reset(ctx.res / f)
        write_rows(ctx.res / f, [], hdr)

    def evaluate(kind, level, blocks, mask=None):
        for f, models in sets:
            for s, acc, gm in _eval_models(ctx, models, blocks, mask):
                write_rows(ctx.res / f, [[kind, level, s, acc, *gm]])

    evaluate("clean", 0, ctx.blocks["test"])
    # frame dropping (re-aggregation of cached per-frame descriptors)
    for rate in ev["robustness_frame_drop"]:
        ctx.state.update(stage=f"Robustness: frame drop {rate}")
        b, kept = load_blocks(ctx.cfg, ctx.clips["test"], frame_drop=rate, seed=123)
        assert kept == ctx.clips["test"]
        evaluate("frame_drop", rate, b)
    # modality removal at test time
    for j, m in enumerate("tav"):
        mask = np.ones(3, np.float32)
        mask[j] = 0
        evaluate(f"remove_{m}", 1, ctx.blocks["test"], mask)
    # audio noise: re-extract VGGish + eGeMAPS on noisy audio (cached, so re-running is instant)
    if not ev["robustness_snr_db"]:
        return
    snrs = ev["robustness_snr_db"]
    noisy = Path(ctx.cfg["paths"]["cache"]) / "robustness_audio.npz"
    if noisy.exists():
        z = np.load(noisy)
        vg = {snr: list(z[f"vg_{snr}"]) for snr in snrs}
        ge = {snr: list(z[f"ge_{snr}"]) for snr in snrs}
    else:
        enc = AudioEncoder(ctx.cfg)
        sr = ctx.cfg["extraction"]["sample_rate"]
        babble = load_audio(ev["babble_wav"], sr) if ev.get("babble_wav") else None
        vg = {x: [] for x in snrs}
        ge = {x: [] for x in snrs}
        pr = Progress(ctx.state, "Robustness: noisy audio features", len(ctx.clips["test"]))
        rng = np.random.default_rng(7)
        for c in ctx.clips["test"]:
            wav = load_audio(ctx.info["videos"][c], sr)
            for snr in snrs:
                a, b = enc(add_noise(wav, snr, babble, rng))
                vg[snr].append(a)
                ge[snr].append(b)
            pr.step(msg=c)
        del enc
        free_gpu()
        np.savez(noisy, **{f"vg_{x}": np.stack(vg[x]) for x in snrs}, **{f"ge_{x}": np.stack(ge[x]) for x in snrs})
    for snr in snrs:
        b = dict(ctx.blocks["test"])
        b["vggish"], b["egemaps"] = np.stack(vg[snr]), np.nan_to_num(np.stack(ge[snr]))
        evaluate("audio_snr_db", snr, b)


def _robust_md(ctx):
    import pandas as pd
    a, c = ctx.res / "robustness.csv", ctx.res / ROBUST_CONTROL[2]
    if not a.exists():
        return []
    A = pd.read_csv(a)
    C = pd.read_csv(c) if c.exists() else None
    L = ["", "## Table 7 - robustness (test accuracy %, mean over seeds; gates = SG-MBF mean gates t/a/v)", "",
         "| Perturbation | Level | SG-MBF | Ungated sum + LN | Δ (gate) | gates t / a / v |", "|---|---|---|---|---|---|"]
    for (k, lv), grp in A.groupby(["perturbation", "level"], sort=False):
        acc = 100 * grp["acc"].mean()
        cc = None if C is None else C[(C.perturbation == k) & (C.level == lv)]["acc"]
        cs = "-" if cc is None or cc.empty else f"{100 * cc.mean():.2f}"
        ds = "-" if cc is None or cc.empty else f"{acc - 100 * cc.mean():+.2f}"
        L.append(f"| {k} | {lv} | {acc:.2f} | {cs} | {ds} | "
                 f"{grp['g_t'].mean():.2f} / {grp['g_a'].mean():.2f} / {grp['g_v'].mean():.2f} |")
    return L


def plan_fairness(ctx):
    demo = load_demographics(ctx.info, "test")
    if not demo:
        ctx.log.warning("No eth_gender_annotations_test.csv found: fairness analysis skipped")
        return
    seeds = ctx.cfg["training"]["seeds"]
    P = np.mean([np.load(ctx.runs / "main" / f"SG-MBF__TAV__s{s}" / "test_pred.npy") for s in seeds], 0)
    err = np.abs(P - ctx.y["test"])
    rows = []
    for attr in ("gender", "ethnicity"):
        groups = {}
        for i, c in enumerate(ctx.clips["test"]):
            g = demo.get(c, {}).get(attr)
            if g:
                groups.setdefault(g, []).append(i)
        for g, idx in sorted(groups.items()):
            rows.append([attr, g, len(idx), *err[idx].mean(0), err[idx].mean()])
    reset(ctx.res / "fairness.csv")
    write_rows(ctx.res / "fairness.csv", rows, ["attribute", "group", "n"] + TRAITS + ["mean_mae"])


# ------------------------------------------------------------------ summary tables (Markdown)
def _load_seed_csv(p):
    if not p.exists():
        return {}
    d = {}
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d.setdefault(r["model"], {}).setdefault(int(r["seed"]), {})[r["trait"]] = {k: float(r[k]) for k in ("acc", "ccc", "r", "r2")}
    return d


def _fmt(vals, pct=True):
    v = np.array(vals) * (100 if pct else 1)
    return f"{v.mean():.2f} ± {v.std(ddof=1) if len(v) > 1 else 0:.2f}" if pct else f"{v.mean():.3f} ± {v.std(ddof=1) if len(v) > 1 else 0:.3f}"


def summary(ctx):
    seeds = ctx.cfg["training"]["seeds"]
    y = ctx.y["test"]
    L = [f"# SG-MBF results ({time.strftime('%Y-%m-%d %H:%M')})", "",
         f"Split: **{ctx.kind}** - train {len(ctx.clips['train'])}, val {len(ctx.clips['val'])}, test {len(ctx.clips['test'])} clips."]
    if ctx.kind != "official":
        L.append("> WARNING: internal split - results are NOT comparable with the official ChaLearn test set.")
    trivial = np.tile(ctx.y["train"].mean(0), (len(y), 1))
    mt = per_trait(y, trivial)
    L += ["", "## Table 3 - test results (accuracy in %, mean ± SD over seeds)", "",
          "| Method | O | C | E | A | N | Mean Acc | Mean CCC |", "|---|---|---|---|---|---|---|---|",
          "| Training-set mean predictor | " + " | ".join(f"{100 * mt[t]['acc']:.2f}" for t in TRAITS) +
          f" | {100 * mt['mean']['acc']:.2f} | {mt['mean']['ccc']:.3f} |"]
    fus = _load_seed_csv(ctx.res / "fusion_seeds.csv")
    preds = {}
    for name in ["SG-MBF"] + BASELINES:
        sub = "main" if name == "SG-MBF" else "fusion"
        P = [np.load(ctx.runs / sub / f"{name}__TAV__s{s}".replace(" ", "_").replace("+", "p") / "test_pred.npy")
             for s in seeds if (ctx.runs / sub / f"{name}__TAV__s{s}".replace(" ", "_").replace("+", "p") / "test_pred.npy").exists()]
        if not P:
            continue
        preds[name] = P
        ms = [per_trait(y, p) for p in P]
        cells = [_fmt([m[t]["acc"] for m in ms]) for t in TRAITS]
        L.append(f"| {'**SG-MBF (ours)**' if name == 'SG-MBF' else name} | " + " | ".join(cells) +
                 f" | {_fmt([m['mean']['acc'] for m in ms])} | {_fmt([m['mean']['ccc'] for m in ms], False)} |")
    if "SG-MBF" in preds:
        acc = bootstrap_ci(y, preds["SG-MBF"], "acc", ctx.cfg["evaluation"]["bootstrap"])
        ccc = bootstrap_ci(y, preds["SG-MBF"], "ccc", min(2000, ctx.cfg["evaluation"]["bootstrap"]))
        L += ["", f"SG-MBF mean accuracy {100 * acc[0]:.2f}% (95% CI {100 * acc[1]:.2f}-{100 * acc[2]:.2f}); "
                  f"mean CCC {ccc[0]:.3f} (95% CI {ccc[1]:.3f}-{ccc[2]:.3f})."]
        comps = [n for n in BASELINES if n in preds]
        if comps:
            st = [paired_bootstrap(y, preds["SG-MBF"], preds[n], ctx.cfg["evaluation"]["bootstrap"]) for n in comps]
            adj = holm([s[3] for s in st])
            L += ["", "## Table 6 - fusion strategies (Δ = SG-MBF minus baseline, accuracy points)", "",
                  "| Strategy | Mean Acc | Mean CCC | Δ Acc [95% CI] | p (Holm) | Params |", "|---|---|---|---|---|---|"]
            for n, s_, pa in zip(comps, st, adj):
                ms = [per_trait(y, p)["mean"] for p in preds[n]]
                prm = json.loads((ctx.runs / "fusion" / f"{n}__TAV__s{seeds[0]}".replace(" ", "_").replace("+", "p") / "meta.json").read_text())["params"]
                L.append(f"| {n} | {_fmt([m['acc'] for m in ms])} | {_fmt([m['ccc'] for m in ms], False)} | "
                         f"{100 * s_[0]:+.2f} [{100 * s_[1]:+.2f}, {100 * s_[2]:+.2f}] | {pa:.4f} | {prm:,} |")
            prm = json.loads((ctx.runs / "main" / f"SG-MBF__TAV__s{seeds[0]}" / "meta.json").read_text())["params"]
            ms = [per_trait(y, p)["mean"] for p in preds["SG-MBF"]]
            L.append(f"| **SG-MBF (ours)** | {_fmt([m['acc'] for m in ms])} | {_fmt([m['ccc'] for m in ms], False)} | - | - | {prm:,} |")
    sd = _load_seed_csv(ctx.res / "seeds.csv")
    if len(sd) > 1:
        L += ["", "## Table 4 - modality ablation", "", "| Modalities | Mean Acc | Mean CCC |", "|---|---|---|"]
        for lab in ["Ours"] + list(MOD_SUBSETS):
            if lab in sd:
                v = sd[lab]
                L.append(f"| {'T + A + V (full)' if lab == 'Ours' else lab} | "
                         f"{_fmt([np.mean([v[s][t]['acc'] for t in TRAITS]) for s in v])} | "
                         f"{_fmt([np.mean([v[s][t]['ccc'] for t in TRAITS]) for s in v], False)} |")
    cp = _load_seed_csv(ctx.res / "components.csv")
    if cp:
        L += ["", "## Table 5 - feature components (other branches complete)", "", "| Variant | Mean Acc | Mean CCC |", "|---|---|---|"]
        for lab, v in cp.items():
            L.append(f"| {lab} | {_fmt([np.mean([v[s][t]['acc'] for t in TRAITS]) for s in v])} | "
                     f"{_fmt([np.mean([v[s][t]['ccc'] for t in TRAITS]) for s in v], False)} |")
    L += _robust_md(ctx)
    from .cross import md_lines
    L += md_lines(ctx.res)
    (ctx.res / "tables.md").write_text("\n".join(L), encoding="utf-8")
    ctx.log.info("Summary tables written to results/tables.md")


# ------------------------------------------------------------------ gate variant selection (validation only)
def _val_pred_from_ckpt(ctx, rd):
    """Validation predictions of an already trained run (model.pt), without touching the test set."""
    from .models import build
    ck = torch.load(rd / "model.pt", map_location="cpu", weights_only=False)
    m = build(ck["name"], ck["dims"], ctx.cfg, ck.get("training", {}).get("gate_variant", "sigmoid"))
    m.load_state_dict(ck["state_dict"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    std = Standardizer.from_state(ck["standardizer"])
    p, g, _ = predict(m.to(dev).eval(), to_dev(std(assemble(ctx.blocks["val"], ck["spec"])), dev))
    return p, g


def plan_gate_select(ctx):
    """Trains every gate variant (5 seeds) and ranks them on the VALIDATION set only.

    No test prediction is computed here. The selection rule is fixed in advance: lowest mean validation
    MSE over seeds (the early-stopping criterion); validation accuracy and CCC are reported alongside.
    """
    from .models import GATE_VARIANTS
    seeds = ctx.cfg["training"]["seeds"]
    yv = ctx.y["val"]
    res = {}
    for v in GATE_VARIANTS:
        preds, gates = [], []
        for s in seeds:
            rd = ctx.runs / "gate_select" / f"SG-MBF_{v}__TAV__s{s}"
            if v == "sigmoid" and (ctx.runs / "main" / f"SG-MBF__TAV__s{s}" / "model.pt").exists() \
                    and not (rd / "val_pred.npy").exists():
                p, g = _val_pred_from_ckpt(ctx, ctx.runs / "main" / f"SG-MBF__TAV__s{s}")
                rd.mkdir(parents=True, exist_ok=True)
                np.save(rd / "val_pred.npy", p)
                np.save(rd / "val_gates.npy", g)
            if not (rd / "val_pred.npy").exists():
                X, _ = ctx.data(FULL)
                ctx.state.update(stage=f"Gate selection (validation only): {v} seed {s}")
                model, meta = train_one(ctx.cfg, "SG-MBF", X["train"], ctx.y["train"], X["val"], yv, s, rd,
                                        ctx.state, f"gate:{v}", {"group": "gate_select"}, gate_variant=v)
                dev = next(model.parameters()).device
                p, g, _ = predict(model, to_dev(X["val"], dev))
                np.save(rd / "val_pred.npy", p)
                np.save(rd / "val_gates.npy", g)
                ctx.log.info("gate %s | seed %d | VAL acc %.4f ccc %.4f mse %.5f", v, s,
                             per_trait(yv, p)["mean"]["acc"], per_trait(yv, p)["mean"]["ccc"],
                             float(np.mean((p - yv) ** 2)))
            preds.append(np.load(rd / "val_pred.npy"))
            gates.append(np.load(rd / "val_gates.npy"))
        res[v] = (preds, gates)
    ref = []
    for s in seeds:
        rd = ctx.runs / "fusion" / f"Ungated_sum__TAV__s{s}"
        if (rd / "model.pt").exists():
            ref.append(_val_pred_from_ckpt(ctx, rd)[0])

    def row(preds):
        ms = [per_trait(yv, p)["mean"] for p in preds]
        return {"mse": float(np.mean([np.mean((p - yv) ** 2) for p in preds])),
                "mse_sd": float(np.std([np.mean((p - yv) ** 2) for p in preds])),
                "acc": float(np.mean([m["acc"] for m in ms])), "acc_sd": float(np.std([m["acc"] for m in ms])),
                "ccc": float(np.mean([m["ccc"] for m in ms])), "ccc_sd": float(np.std([m["ccc"] for m in ms]))}

    table = {v: row(p) for v, (p, _) in res.items()}
    best = min(table, key=lambda v: table[v]["mse"])
    out = {"rule": "lowest mean validation MSE over seeds (fixed before looking at the results)",
           "selected": best, "variants": table, "test_set_used": False}
    L = ["# Gate variant selection - VALIDATION set only (the test set is not used)", "",
         f"Selection rule (fixed in advance): {out['rule']}.", "",
         "| Variant | Val MSE | Val Acc (%) | Val CCC | mean gates (t, a, v) |", "|---|---|---|---|---|"]
    for v, r in sorted(table.items(), key=lambda kv: kv[1]["mse"]):
        g = np.mean([x.mean(0) for x in res[v][1]], 0)
        L.append(f"| {'**' + v + '**' if v == best else v} | {r['mse']:.5f} ± {r['mse_sd']:.5f} | "
                 f"{100 * r['acc']:.2f} ± {100 * r['acc_sd']:.2f} | {r['ccc']:.3f} ± {r['ccc_sd']:.3f} | "
                 f"{', '.join(f'{x:.2f}' for x in g)} |")
    if ref:
        rr = row(ref)
        out["ungated_sum_val"] = rr
        L.append(f"| *Ungated sum (reference)* | {rr['mse']:.5f} ± {rr['mse_sd']:.5f} | "
                 f"{100 * rr['acc']:.2f} ± {100 * rr['acc_sd']:.2f} | {rr['ccc']:.3f} ± {rr['ccc_sd']:.3f} | 1, 1, 1 |")
        dl, lo, hi, pv = paired_bootstrap(yv, res[best][0], ref, ctx.cfg["evaluation"]["bootstrap"])
        d = {"delta": dl, "lo": lo, "hi": hi, "p": pv}
        out["selected_vs_ungated_val"] = d
        beats = table[best]["mse"] < rr["mse"]
        out["beats_ungated_on_val"] = bool(beats)
        L += ["", f"Selected variant: **{best}** - {'better' if beats else 'NOT better'} than the ungated sum on "
                  f"validation (MSE {table[best]['mse']:.5f} vs {rr['mse']:.5f}; Δacc {100 * d['delta']:+.2f} pts, "
                  f"95% CI [{100 * d['lo']:+.2f}, {100 * d['hi']:+.2f}], p = {d['p']:.4f}).",
              "", "Next step: '10 · Évaluation finale (test)' evaluates ONLY the selected variant on the test set, once."
              if beats else "The gate brings no gain on validation: report SG-MBF as an analysis paper (option 1)."]
    from .models import GATE_VARIANTS as GV
    L += ["", "Variants:"] + [f"- `{k}`: {t}" for k, t in GV.items()]
    (ctx.res / "gate_selection.json").write_text(json.dumps(out, indent=1))
    (ctx.res / "gate_selection.md").write_text("\n".join(L), encoding="utf-8")
    ctx.log.info("Gate selection: %s (validation MSE %.5f)", best, table[best]["mse"])


def plan_gate_final(ctx):
    """Uses the variant chosen on validation as THE SG-MBF, archives the original runs, re-runs main + fusion
    (only the SG-MBF runs are retrained; the baselines are reused). Ablation, robustness, fairness and the
    cross-dataset steps then use the selected variant."""
    import re
    sel = json.loads((ctx.res / "gate_selection.json").read_text())["selected"]
    cur = ctx.cfg["training"].get("gate_variant", "sigmoid")
    if sel != cur:
        old = ctx.runs / "main"
        arch = ctx.runs / f"archive_gate_{cur}"
        arch.mkdir(parents=True, exist_ok=True)
        for rd in old.glob("SG-MBF__*"):
            shutil.move(str(rd), arch / rd.name)
        for f in ("seeds.csv", "gates_test.csv", "fusion_seeds.csv"):
            if (ctx.res / f).exists():
                shutil.copy(ctx.res / f, arch / f)
        cfgp = ROOT / "config.yaml"
        txt = cfgp.read_text(encoding="utf-8")
        line = f"  gate_variant: {sel}                # chosen on validation (results/gate_selection.md)"
        txt = re.sub(r"(?m)^  gate_variant:.*$", line, txt) if re.search(r"(?m)^  gate_variant:", txt) \
            else re.sub(r"(?m)^training:\s*$", "training:\n" + line, txt, count=1)
        cfgp.write_text(txt, encoding="utf-8")
        ctx.cfg["training"]["gate_variant"] = sel
        ctx.log.info("SG-MBF now uses gate variant '%s'; original runs archived in %s", sel, arch)
    plan_main(ctx)
    plan_fusion(ctx)


PLANS = {"main": plan_main, "fusion": plan_fusion, "ablation": plan_ablation,
         "robustness": plan_robustness, "fairness": plan_fairness,
         "gate_select": plan_gate_select, "gate_final": plan_gate_final}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="main")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("experiments", cfg)
    state = State(cfg)
    write_pid(cfg, "job")
    try:
        state.update(job=f"experiments:{args.plan}", status="running", stage="Loading cached features",
                     started=time.time(), error=None)
        ctx = Ctx(cfg, log, state)
        plans = ["main", "fusion", "ablation", "robustness", "fairness"] if args.plan == "all" else args.plan.split(",")
        for p in plans:
            state.update(plan=p)
            PLANS[p](ctx)
            if p != "gate_select":
                summary(ctx)
        state.update(status="finished", stage=f"Experiments '{args.plan}' finished", finished=time.time())
    except Exception as ex:
        log.error(traceback.format_exc())
        state.update(status="error", error=repr(ex))
        raise
    finally:
        remove_pid(cfg, "job")


if __name__ == "__main__":
    main()
