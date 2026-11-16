"""Leakage audit: the same models evaluated on the official split (S0), on video-disjoint splits (S1_k) and on
identity-disjoint splits (S2_k) built by sgmbf.identity.

    python -m sgmbf.leakage [--splits S0,S1_0,S2_0,...] [--seeds 1,2,3,4,5]

For every split: training-set mean predictor, "same source" lookup predictors (no input at all: the mean
training label of the clips from the same video / the same identity), unimodal T, A, V and the eight fusion
strategies (5 seeds). Model selection uses each split's own validation set; each test set is predicted once
per trained model. Outputs: results/leakage/<split>/seeds.csv, results/leakage/summary.csv and
results/leakage.md.
"""
import argparse
import json
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import ROOT, TRAITS, State, get_logger, load_config, remove_pid, write_pid
from .experiments import BASELINES, FULL, Ctx, reset, run, seed_rows, write_rows
from .data import label_matrix
from .features import load_blocks
from .metrics import bootstrap_ci, paired_bootstrap, per_trait

UNIMODAL = {"T": "t", "A": "a", "V": "v", "T+A": "ta", "T+V": "tv", "A+V": "av"}


class SplitCtx(Ctx):
    """Ctx on a given split; runs and results kept apart per split."""

    def __init__(self, cfg, log, state, base, tag, splits):
        self.cfg, self.log, self.state, self.info, self.kind = cfg, log, state, base.info, tag
        self.splits = splits
        self.blocks, self.clips, self.y = {}, {}, {}
        for s in ("train", "val", "test"):
            b, kept = load_blocks(cfg, splits[s])
            self.blocks[s], self.clips[s] = b, kept
            self.y[s] = label_matrix(self.info, kept)
        self.runs = Path(cfg["paths"]["runs"]) / "leakage" / tag
        self.res = Path(cfg["paths"]["results"]) / "leakage" / tag
        self.res.mkdir(parents=True, exist_ok=True)


def lookup_pred(info, train, test, key):
    """Mean training label of the same group (video or identity); training mean when the group is unseen."""
    y = {c: [info["labels"][c][t] for t in TRAITS] for c in train}
    mean = np.mean(list(y.values()), 0)
    g = defaultdict(list)
    for c in train:
        g[key(c)].append(y[c])
    P = np.array([np.mean(g[key(c)], 0) if key(c) in g else mean for c in test])
    seen = np.array([key(c) in g for c in test])
    return P, seen, mean


def load_splits(cfg, base):
    """S0 official; S0R_k random clip-level (leaky by construction, same sizes: controls for test-set
    composition); S1_k video-disjoint; S2_k identity-disjoint."""
    import random
    out = {"S0": base.splits}
    sp = ROOT / "splits"
    clips = sorted(c for v in base.splits.values() for c in v)
    n_draws = int(cfg.get("leakage", {}).get("draws", 10))
    for k in range(n_draws):
        f = sp / f"S0R_{k}.json"
        if not f.exists() and sp.exists():
            c = clips[:]
            random.Random(200 + k).shuffle(c)
            n = {s: len(base.splits[s]) for s in ("train", "val")}
            sp_k = {"train": sorted(c[:n["train"]]), "val": sorted(c[n["train"]:n["train"] + n["val"]]),
                    "test": sorted(c[n["train"] + n["val"]:])}
            f.write_text(json.dumps({"splits": sp_k}))
        if f.exists():
            out[f.stem] = json.loads(f.read_text())["splits"]
    for f in sorted(list(sp.glob("S[12]_*.json")) + list(sp.glob("S2A_*.json"))):
        out[f.stem] = json.loads(f.read_text())["splits"]
    ident = json.loads((sp / "identities.json").read_text()) if (sp / "identities.json").exists() else {}
    if ident and set(ident) == set(clips):      # extra identity-disjoint draws (same rule and seeds as sgmbf.identity)
        from .identity import group_split
        for k in range(n_draws):
            f = sp / f"S2A_{k}.json"
            if not f.exists():
                f.write_text(json.dumps({"splits": group_split(clips, ident, seed=100 + k)}))
            out[f.stem] = json.loads(f.read_text())["splits"]
    return out, ident


def summarise(cfg, rows, out_md):
    """rows: list of dicts(split, model, acc, acc_lo, acc_hi, ccc, ccc_lo, ccc_hi, sd_acc, sd_ccc)."""
    import pandas as pd
    d = pd.DataFrame(rows)
    d.to_csv(Path(cfg["paths"]["results"]) / "leakage" / "summary.csv", index=False)
    fam = lambda s: {"S0": "S0 official"}.get(s) or ("S0R random clips" if s.startswith("S0R") else
                     "S1 video-disjoint" if s.startswith("S1") else
                     "S2A identity-disjoint (refined)" if s.startswith("S2A") else "S2 identity-disjoint")
    d["family"] = d.split.map(fam)
    g = d.groupby(["model", "family"]).agg(acc=("acc", "mean"), acc_sd=("acc", "std"), ccc=("ccc", "mean"),
                                           ccc_sd=("ccc", "std"), n=("split", "count")).reset_index()
    order = list(dict.fromkeys(d.model))
    fams = ["S0 official", "S0R random clips", "S1 video-disjoint", "S2 identity-disjoint",
            "S2A identity-disjoint (refined)"]
    fams = [f for f in fams if f in set(d.family)]
    L = ["# Leakage audit: official vs video-disjoint vs identity-disjoint splits", "",
         "Mean over split draws (S1, S2: 3 draws; S0R, S2A: up to 10 draws, see summary.csv; S2 = single-link identities, S2A = refined average-link identities; SD across draws in brackets). Each model: 5 seeds per draw.", "",
         "| Model | " + " | ".join(f"{f} Acc" for f in fams) + " | " + " | ".join(f"{f} CCC" for f in fams) +
         " | ΔCCC S0→S2 |", "|---|" + "---|" * (2 * len(fams) + 1)]
    for m in order:
        cells, cc = [], {}
        for metric in ("acc", "ccc"):
            for f in fams:
                r = g[(g.model == m) & (g.family == f)]
                if r.empty:
                    cells.append("-")
                    continue
                v, sd = float(r[metric].iloc[0]), r[metric + "_sd"].iloc[0]
                cc[(metric, f)] = v
                k = 100 if metric == "acc" else 1
                fmt = "{:.2f}" if metric == "acc" else "{:.3f}"
                cells.append(fmt.format(v * k) + ("" if np.isnan(sd) else f" ({fmt.format(sd * k)})"))
        dlt = cc.get(("ccc", "S2 identity-disjoint"), np.nan) - cc.get(("ccc", "S0 official"), np.nan)
        L.append(f"| {m} | " + " | ".join(cells) + f" | {dlt:+.3f} |")
    L += ["", "Per-split detail (95% bootstrap CI over test clips, seeds pooled) in results/leakage/summary.csv."]
    out_md.write_text("\n".join(L), encoding="utf-8")


def identity_probe(cfg, base, ident, out):
    """How much identity each modality encodes: top-1 nearest-neighbour retrieval of a clip of the same
    video (cosine on z-scored features, the query's own clip excluded), on clips whose video has >= 2 clips."""
    import pandas as pd
    from .data import youtube_id
    from .features import assemble
    clips = sorted(c for v in base.splits.values() for c in v)
    b, kept = load_blocks(cfg, clips)
    vid = np.array([youtube_id(c) for c in kept])
    n_per = pd.Series(vid).value_counts()
    q = np.where(pd.Series(vid).map(n_per).values >= 2)[0]
    rows = []
    for name, spec in (("T", {"t": FULL["t"]}), ("A", {"a": FULL["a"]}), ("V", {"v": FULL["v"]}),
                       ("V: face geometry", {"v": ["geo"]}), ("V: Emo-AffectNet", {"v": ["emo"]}),
                       ("V: pose", {"v": ["pose"]}), ("A: VGGish", {"a": ["vggish"]}), ("A: eGeMAPS", {"a": ["egemaps"]})):
        X = next(iter(assemble(b, spec).values())).astype(np.float32)
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
        hit = 0
        for s0 in range(0, len(q), 512):
            qi = q[s0:s0 + 512]
            S = X[qi] @ X.T
            S[np.arange(len(qi)), qi] = -np.inf
            hit += int((vid[S.argmax(1)] == vid[qi]).sum())
        rows.append({"features": name, "same_video_top1": hit / len(q), "n_queries": len(q),
                     "chance": float(np.mean((pd.Series(vid).map(n_per).values[q] - 1) / (len(kept) - 1)))})
    pd.DataFrame(rows).to_csv(out, index=False)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="")
    ap.add_argument("--seeds", default="")
    args = ap.parse_args()
    cfg = load_config()
    if args.seeds:
        cfg["training"]["seeds"] = [int(s) for s in args.seeds.split(",")]
    log = get_logger("leakage", cfg)
    state = State(cfg)
    write_pid(cfg, "job")
    try:
        state.update(job="leakage", status="running", stage="Loading features", error=None, started=time.time())
        base = Ctx(cfg, log, state)
        all_splits, ident = load_splits(cfg, base)
        if len(all_splits) == 1:
            raise RuntimeError("No splits/S1_*.json or S2_*.json: run step 11 (identities) first.")
        tags = args.splits.split(",") if args.splits else list(all_splits)
        state.update(stage="Identity probe (same-video retrieval per modality)")
        root0 = Path(cfg["paths"]["results"]) / "leakage"
        root0.mkdir(parents=True, exist_ok=True)
        log.info("Identity probe: %s", identity_probe(cfg, base, ident, root0 / "identity_probe.csv"))
        seeds = cfg["training"]["seeds"]
        rows = []
        root = Path(cfg["paths"]["results"]) / "leakage"
        root.mkdir(parents=True, exist_ok=True)
        from .data import youtube_id
        for tag in tags:
            ctx = SplitCtx(cfg, log, state, base, tag, all_splits[tag])
            y = ctx.y["test"]
            reset(ctx.res / "seeds.csv")
            # reference predictors without any input
            P_vid, seen_v, mean = lookup_pred(base.info, ctx.clips["train"], ctx.clips["test"], youtube_id)
            preds = {"Training-set mean": [np.tile(mean, (len(y), 1))], "Lookup: same video (no input)": [P_vid]}
            if ident:
                P_id, seen_i, _ = lookup_pred(base.info, ctx.clips["train"], ctx.clips["test"], lambda c: ident.get(c, c))
                preds["Lookup: same identity (no input)"] = [P_id]
                log.info("%s: test clips with a training video %d, with a training identity %d / %d",
                         tag, seen_v.sum(), seen_i.sum(), len(y))
            for lab, mods in UNIMODAL.items():
                spec = {m: FULL[m] for m in mods}
                lab_full = f"Unimodal {lab}" if len(mods) == 1 else f"Bimodal {lab}"
                preds[lab_full] = []
                for s in seeds:
                    _, p = run(ctx, "unimodal", "SG-MBF", lab, spec, s)
                    preds[lab_full].append(p)
                    write_rows(ctx.res / "seeds.csv", seed_rows(lab_full, s, y, p),
                               ["model", "seed", "trait", "acc", "ccc", "r", "r2"])
            for name in BASELINES + ["SG-MBF"]:
                preds[name] = []
                for s in seeds:
                    _, p = run(ctx, "fusion", name, "TAV", FULL, s)
                    preds[name].append(p)
                    write_rows(ctx.res / "seeds.csv", seed_rows(name, s, y, p))
            for name, ps in preds.items():
                acc = bootstrap_ci(y, ps, "acc", n=2000)
                ccc = bootstrap_ci(y, ps, "ccc", n=2000)
                per = [per_trait(y, p)["mean"] for p in ps]
                rows.append({"split": tag, "model": name, "acc": acc[0], "acc_lo": acc[1], "acc_hi": acc[2],
                             "ccc": ccc[0], "ccc_lo": ccc[1], "ccc_hi": ccc[2],
                             "sd_acc_seeds": float(np.std([m["acc"] for m in per])),
                             "sd_ccc_seeds": float(np.std([m["ccc"] for m in per])), "n_test": len(y)})
            # SG-MBF vs every fusion baseline on this split (paired over clips, seeds paired)
            cmp = []
            for name in BASELINES:
                d, lo, hi, p = paired_bootstrap(y, preds["SG-MBF"], preds[name], n=2000)
                cmp.append({"split": tag, "baseline": name, "delta_acc": d, "lo": lo, "hi": hi, "p": p})
            import pandas as pd
            pd.DataFrame(cmp).to_csv(ctx.res / "sgmbf_vs_baselines.csv", index=False)
            summarise(cfg, rows, Path(cfg["paths"]["results"]) / "leakage.md")
            log.info("Leakage audit: split %s done", tag)
        state.update(status="finished", stage="Leakage audit finished (results/leakage.md)", finished=time.time())
    except Exception as ex:
        log.error(traceback.format_exc())
        state.update(status="error", error=repr(ex))
        raise
    finally:
        remove_pid(cfg, "job")


if __name__ == "__main__":
    main()
