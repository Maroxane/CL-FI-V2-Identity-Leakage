"""Control experiment: seen vs unseen persons with the SAME trained model.

    python -m sgmbf.control

The drop from random clip-level splits (S0R) to identity-disjoint splits (S2A) mixes two effects: (a) test
persons are no longer in the training set (leakage proper) and (b) at a fixed number of clips, an
identity-disjoint training set contains fewer distinct persons. This experiment isolates (a).

S3_k is built from S2A_k (k = 0..draws-1): one clip of `seen_videos` randomly chosen training videos (videos
with at least 2 training clips) is moved out of the training set. Each model is trained once on the reduced
training set (validation set unchanged, still identity-disjoint) and evaluated on
  - "seen person"   : the clips moved out (other clips of the same video are in the training set);
  - "unseen person" : the S2A_k test set (no clip of these persons in training).
Outputs: splits/S3_k.json, results/control/summary.csv, results/control.md. Runs in runs/leakage/S3_k.
"""
import json
import random
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import ROOT, State, get_logger, load_config, remove_pid, write_pid
from .data import youtube_id
from .experiments import BASELINES, FULL, Ctx, run
from .leakage import UNIMODAL, SplitCtx, load_splits, lookup_pred
from .metrics import per_trait


def make_s3(src, seed, n_videos):
    by = defaultdict(list)
    for c in src["train"]:
        by[youtube_id(c)].append(c)
    rng = random.Random(seed)
    elig = sorted(v for v, cl in by.items() if len(cl) >= 2)
    rng.shuffle(elig)
    held = sorted(rng.choice(sorted(by[v])) for v in elig[:n_videos])
    hs = set(held)
    return {"splits": {"train": sorted(c for c in src["train"] if c not in hs), "val": sorted(src["val"]),
                       "test": sorted(src["test"]) + held}, "seen_test": held}


def report(rows, md):
    import pandas as pd
    d = pd.DataFrame(rows)
    d.to_csv(md.parent / "control" / "summary.csv", index=False)
    p = d.pivot_table(index=["split", "model"], columns="subset", values=["ccc", "acc"]).reset_index()
    p.columns = ["_".join(x for x in c if x).replace(" person", "") for c in p.columns]
    order = list(dict.fromkeys(d.model))

    def ms(x, k=1.0, f="{:.3f}"):
        x = np.asarray(x, float)
        return f.format(k * x.mean()) + (f" ({f.format(k * x.std(ddof=1))})" if len(x) > 1 else "")

    nd = d.split.nunique()
    L = ["# Control: seen vs unseen persons, same trained model", "",
         f"{nd} draw(s) (S3_k, built from S2A_k). Mean over draws, SD across draws in brackets; each draw is the mean "
         f"of the seeds. Clips per draw: seen {d[d.subset == 'seen person'].n.mean():.0f}, "
         f"unseen {d[d.subset == 'unseen person'].n.mean():.0f}.", "",
         "| Model | CCC seen | CCC unseen | Gap seen - unseen | min / max gap | Acc seen | Acc unseen |", "|---|---|---|---|---|---|---|"]
    for m in order:
        x = p[p.model == m]
        if x.empty:
            continue
        g = (x.ccc_seen - x.ccc_unseen).values
        L.append(f"| {m} | {ms(x.ccc_seen)} | {ms(x.ccc_unseen)} | {ms(g)} | {g.min():+.3f} / {g.max():+.3f} | "
                 f"{ms(x.acc_seen, 100, '{:.2f}')} | {ms(x.acc_unseen, 100, '{:.2f}')} |")
    w = p.pivot(index="split", columns="model")
    pairs = [("GMU", "Unimodal V"), ("Bimodal A+V", "Unimodal V"), ("Bimodal T+V", "Unimodal V"),
             ("Bimodal T+A", "Unimodal A"), ("GMU", "Bimodal A+V"), ("SG-MBF", "Ungated sum + LN")]
    L += ["", "## Contrasts between models (CCC), on seen and on unseen persons", "",
          "| Contrast | Seen persons | Unseen persons | Difference unseen - seen | min / max |", "|---|---|---|---|---|"]
    for a, b in pairs:
        try:
            s = (w[("ccc_seen", a)] - w[("ccc_seen", b)]).values
            u = (w[("ccc_unseen", a)] - w[("ccc_unseen", b)]).values
        except KeyError:
            continue
        L.append(f"| {a} - {b} | {ms(s)} | {ms(u)} | {ms(u - s)} | {(u - s).min():+.3f} / {(u - s).max():+.3f} |")
    md.write_text("\n".join(L), encoding="utf-8")


def main():
    cfg = load_config()
    log = get_logger("control", cfg)
    state = State(cfg)
    write_pid(cfg, "job")
    try:
        state.update(job="control", status="running", stage="Control: loading", error=None, started=time.time())
        base = Ctx(cfg, log, state)
        splits, ident = load_splits(cfg, base)
        cc = cfg.get("control", {})
        n_draws, n_vid = int(cc.get("draws", 5)), int(cc.get("seen_videos", 600))
        sp = ROOT / "splits"
        res = Path(cfg["paths"]["results"])
        (res / "control").mkdir(parents=True, exist_ok=True)
        seeds = cfg["training"]["seeds"]
        models = [("unimodal", "SG-MBF", lab, {m: FULL[m] for m in mods},
                   f"Unimodal {lab}" if len(mods) == 1 else f"Bimodal {lab}") for lab, mods in UNIMODAL.items()]
        models += [("fusion", n, "TAV", FULL, n) for n in BASELINES + ["SG-MBF"]]
        rows = []
        for k in range(n_draws):
            if f"S2A_{k}" not in splits:
                raise RuntimeError(f"splits/S2A_{k}.json missing: run step 12 first.")
            f = sp / f"S3_{k}.json"
            if not f.exists():
                f.write_text(json.dumps(make_s3(splits[f"S2A_{k}"], 300 + k, n_vid)))
            z = json.loads(f.read_text())
            tag = f"S3_{k}"
            ctx = SplitCtx(cfg, log, state, base, tag, z["splits"])
            seen = np.isin(ctx.clips["test"], z["seen_test"])
            tv = {youtube_id(c) for c in ctx.clips["train"]}
            in_train = np.array([youtube_id(c) in tv for c in ctx.clips["test"]])
            if not (in_train == seen).all():
                raise RuntimeError(f"{tag}: seen/unseen clips are not separated by training videos")
            y = ctx.y["test"]
            log.info("%s: train %d, seen-person test %d, unseen-person test %d", tag, len(ctx.clips["train"]),
                     seen.sum(), (~seen).sum())
            P_vid, _, _ = lookup_pred(base.info, ctx.clips["train"], ctx.clips["test"], youtube_id)
            preds = {"Lookup: same video (no input)": [P_vid]}
            for group, name, lab, spec, disp in models:
                preds[disp] = [run(ctx, group, name, lab, spec, s)[1] for s in seeds]
            for disp, ps in preds.items():
                for sub, mk in (("seen person", seen), ("unseen person", ~seen)):
                    m = [per_trait(y[mk], p[mk])["mean"] for p in ps]
                    rows.append({"split": tag, "model": disp, "subset": sub, "n": int(mk.sum()),
                                 "acc": float(np.mean([x["acc"] for x in m])),
                                 "ccc": float(np.mean([x["ccc"] for x in m]))})
            report(rows, res / "control.md")
            log.info("Control: %s done", tag)
        state.update(status="finished", stage="Control finished (results/control.md)", finished=time.time())
    except Exception as ex:
        log.error(traceback.format_exc())
        state.update(status="error", error=repr(ex))
        raise
    finally:
        remove_pid(cfg, "job")


if __name__ == "__main__":
    main()
