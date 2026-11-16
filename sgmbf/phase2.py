"""Phase 2 of the leakage audit: secondary analyses repeated with the models trained by sgmbf.leakage.
No model is trained here: every analysis re-uses runs/leakage/<split>/.

    python -m sgmbf.phase2 [--parts seen,robust,fair,probe,mupta]

Families: S0R_k = random clip-level splits (leaky), S2A_k = identity-disjoint splits (leak-free).
  seen   : S0 and S0R_k, test clips whose YouTube video is in the training set vs clips of unseen videos.
  robust : S2A_k, SG-MBF vs its no-gate control (Ungated sum + LN): modality removal, frame dropping and
           audio noise. Noisy audio features exist only for the official test clips, so audio noise is
           evaluated on the clips of each S2A_k test set that belong to the official test set.
  fair   : S0R_k and S2A_k, mean absolute error per gender / ethnicity group (dev + test annotations).
  probe  : S0R_k and S2A_k, how much identity survives in the learned representation: top-1 same-video
           retrieval among test clips, on the input features and on the last hidden layer.
  mupta  : S0R_k and S2A_k, zero-shot transfer to MuPTA (speaker level).
Outputs: results/phase2/*.csv and results/phase2.md.
"""
import argparse
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .common import TRAITS, State, get_logger, load_config, remove_pid, write_pid
from .data import load_demographics, youtube_id
from .experiments import BASELINES, FULL, Ctx
from .features import Standardizer, assemble, load_blocks
from .leakage import UNIMODAL, SplitCtx, load_splits
from .metrics import per_trait
from .models import build
from .train import predict, to_dev

PARTS = ["seen", "robust", "fair", "probe", "mupta"]


def model_list():
    """(group, name, run label, display label) for the 14 trained models of the audit."""
    out = []
    for lab, mods in UNIMODAL.items():
        out.append(("unimodal", "SG-MBF", lab, f"Unimodal {lab}" if len(mods) == 1 else f"Bimodal {lab}"))
    return out + [("fusion", n, "TAV", n) for n in BASELINES + ["SG-MBF"]]


def rdir(ctx, group, name, label, seed):
    return ctx.runs / group / f"{name}__{label}__s{seed}".replace(" ", "_").replace(":", "").replace("+", "p")


def load_model(ctx, group, name, label, seed, dev):
    p = rdir(ctx, group, name, label, seed) / "model.pt"
    if not p.exists():
        return None
    ck = torch.load(p, map_location="cpu", weights_only=False)
    m = build(ck["name"], ck["dims"], ctx.cfg, ck.get("training", {}).get("gate_variant", "sigmoid"))
    m.load_state_dict(ck["state_dict"])
    return m.to(dev).eval(), Standardizer.from_state(ck["standardizer"]), ck["spec"]


def tags_of(splits, family, n=None):
    t = sorted((s for s in splits if s.rsplit("_", 1)[0] == family), key=lambda s: int(s.rsplit("_", 1)[1]))
    return t if n is None else t[:n]


def family(tag):
    return tag.rsplit("_", 1)[0] if "_" in tag else tag


def top1_same_video(X, vid):
    """Top-1 nearest neighbour (cosine on z-scored columns) is a clip of the same video; queries = clips
    whose video has at least 2 clips in the set. Returns (hit rate, n queries, chance)."""
    vid = np.asarray(vid)
    uniq, inv, cnt = np.unique(vid, return_inverse=True, return_counts=True)
    q = np.where(cnt[inv] >= 2)[0]
    if len(q) == 0:
        return float("nan"), 0, float("nan")
    X = np.asarray(X, np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    hit = 0
    for s in range(0, len(q), 512):
        qi = q[s:s + 512]
        S = X[qi] @ X.T
        S[np.arange(len(qi)), qi] = -np.inf
        hit += int((vid[S.argmax(1)] == vid[qi]).sum())
    return hit / len(q), int(len(q)), float(np.mean((cnt[inv][q] - 1) / (len(vid) - 1)))


# ------------------------------------------------------------------------------------------------ parts
def part_seen(P):
    rows = []
    for tag in ["S0"] + tags_of(P.splits, "S0R"):
        ctx = P.ctx(tag)
        P.state.update(stage=f"Phase 2: seen vs unseen videos ({tag})")
        tv = {youtube_id(c) for c in ctx.clips["train"]}
        seen = np.array([youtube_id(c) in tv for c in ctx.clips["test"]])
        y = ctx.y["test"]
        for group, name, lab, disp in model_list():
            ps = [np.load(f) for f in (rdir(ctx, group, name, lab, s) / "test_pred.npy" for s in P.seeds) if f.exists()]
            if not ps:
                continue
            for sub, mk in (("seen video", seen), ("unseen video", ~seen), ("all", np.ones(len(y), bool))):
                m = [per_trait(y[mk], p[mk])["mean"] for p in ps]
                rows.append({"split": tag, "model": disp, "subset": sub, "n": int(mk.sum()), "seeds": len(ps),
                             "acc": float(np.mean([x["acc"] for x in m])), "ccc": float(np.mean([x["ccc"] for x in m]))})
    return rows


def part_robust(P):
    ev = P.cfg["evaluation"]
    dev = P.dev
    noisy_f = Path(P.cfg["paths"]["cache"]) / "robustness_audio.npz"
    noisy, pos = None, {}
    if noisy_f.exists():
        z = np.load(noisy_f)
        noisy = {int(k.split("_")[1]): (z[k], z["ge_" + k.split("_")[1]]) for k in z.files if k.startswith("vg_")}
        pos = {c: i for i, c in enumerate(P.base.clips["test"])}
        if any(len(v[0]) != len(pos) for v in noisy.values()):
            P.log.warning("robustness_audio.npz does not match the official test set: audio noise skipped")
            noisy = None
    else:
        P.log.warning("No cache/robustness_audio.npz (step 5 never ran): audio noise skipped")
    n_fd = int(P.cfg.get("phase2", {}).get("frame_drop_draws", 3))
    rows = []
    for k, tag in enumerate(tags_of(P.splits, "S2A")):
        ctx = P.ctx(tag)
        y, clips = ctx.y["test"], ctx.clips["test"]
        conds = [("clean", 0, ctx.blocks["test"], None, None)]
        for j, m in enumerate("tav"):
            mask = np.ones(3, np.float32)
            mask[j] = 0
            conds.append((f"remove_{m}", 1, ctx.blocks["test"], mask, None))
        if k < n_fd:
            for rate in ev["robustness_frame_drop"]:
                if rate > 0:
                    P.state.update(stage=f"Phase 2: robustness {tag}, frame drop {rate}")
                    b, kept = load_blocks(P.cfg, clips, frame_drop=rate, seed=123)
                    if kept == clips:
                        conds.append(("frame_drop", rate, b, None, None))
        if noisy:
            idx = np.array([i for i, c in enumerate(clips) if c in pos])
            src = np.array([pos[clips[i]] for i in idx])
            if len(idx) >= 50:
                sub = {kk: v[idx] for kk, v in ctx.blocks["test"].items()}
                conds.append(("clean (official-test subset)", 0, sub, None, idx))
                for snr, (vg, ge) in sorted(noisy.items(), reverse=True):
                    b = dict(sub)
                    b["vggish"], b["egemaps"] = vg[src].astype(np.float32), np.nan_to_num(ge[src]).astype(np.float32)
                    conds.append(("audio_snr_db", snr, b, None, idx))
        P.state.update(stage=f"Phase 2: robustness {tag}")
        for name in ("SG-MBF", "Ungated sum + LN"):
            for s in P.seeds:
                lm = load_model(ctx, "fusion", name, "TAV", s, dev)
                if lm is None:
                    continue
                m, std, spec = lm
                for kind, level, blocks, mask, idx in conds:
                    X = std(assemble(blocks, spec))
                    yy = y if idx is None else y[idx]
                    mm = None if mask is None else torch.as_tensor(np.tile(mask, (len(yy), 1)), device=dev,
                                                                   dtype=torch.float32)
                    with torch.no_grad():
                        p, g, _ = predict(m, to_dev(X, dev), modality_mask=mm)
                    r = per_trait(yy, p)["mean"]
                    gm = g.mean(0) if g is not None else [np.nan] * 3
                    rows.append({"split": tag, "model": name, "perturbation": kind, "level": level, "seed": s,
                                 "n": len(yy), "acc": r["acc"], "ccc": r["ccc"],
                                 "g_t": float(gm[0]), "g_a": float(gm[1]), "g_v": float(gm[2])})
    return rows


def part_fair(P):
    demo = {}
    for split in P.base.info["demographic_files"]:
        demo.update(load_demographics(P.base.info, split))
    if not demo:
        P.log.warning("No eth_gender_annotations_*.csv found: fairness skipped")
        return []
    rows = []
    for tag in tags_of(P.splits, "S0R") + tags_of(P.splits, "S2A"):
        ctx = P.ctx(tag)
        P.state.update(stage=f"Phase 2: fairness ({tag})")
        y, clips = ctx.y["test"], ctx.clips["test"]
        for name in ("SG-MBF", "GMU"):
            ps = [np.load(f) for f in (rdir(ctx, "fusion", name, "TAV", s) / "test_pred.npy" for s in P.seeds) if f.exists()]
            if not ps:
                continue
            err = np.abs(np.mean(ps, 0) - y).mean(1)
            for attr in ("gender", "ethnicity"):
                groups = {}
                for i, c in enumerate(clips):
                    g = demo.get(c, {}).get(attr)
                    if g:
                        groups.setdefault(g, []).append(i)
                for g, idx in sorted(groups.items()):
                    rows.append({"split": tag, "model": name, "attribute": attr, "group": g, "n": len(idx),
                                 "mae": float(err[idx].mean())})
    return rows


def part_probe(P):
    dev = P.dev
    n = int(P.cfg.get("phase2", {}).get("probe_draws", 3))
    todo = [("unimodal", "SG-MBF", "A", "Unimodal A"), ("unimodal", "SG-MBF", "V", "Unimodal V"),
            ("unimodal", "SG-MBF", "A+V", "Bimodal A+V"), ("fusion", "GMU", "TAV", "GMU"),
            ("fusion", "SG-MBF", "TAV", "SG-MBF")]
    rows = []
    for tag in tags_of(P.splits, "S0R", n) + tags_of(P.splits, "S2A", n):
        ctx = P.ctx(tag)
        P.state.update(stage=f"Phase 2: identity probe on learned representations ({tag})")
        vid = [youtube_id(c) for c in ctx.clips["test"]]
        for group, name, lab, disp in todo:
            for s in P.seeds:
                lm = load_model(ctx, group, name, lab, s, dev)
                if lm is None:
                    continue
                m, std, spec = lm
                X = std(assemble(ctx.blocks["test"], spec))
                if s == P.seeds[0]:
                    h, nq, ch = top1_same_video(np.concatenate([X[k] for k in sorted(X)], 1), vid)
                    rows.append({"split": tag, "model": disp, "representation": "input features", "seed": 0,
                                 "top1": h, "n_queries": nq, "chance": ch})
                if not hasattr(m, "head"):
                    continue
                store = []
                hk = m.head.out.register_forward_hook(lambda mod, inp, out: store.append(inp[0].detach().float().cpu()))
                with torch.no_grad():
                    predict(m, to_dev(X, dev))
                hk.remove()
                h, nq, ch = top1_same_video(torch.cat(store).numpy(), vid)
                rows.append({"split": tag, "model": disp, "representation": "last hidden layer", "seed": s,
                             "top1": h, "n_queries": nq, "chance": ch})
    return rows


def part_mupta(P):
    from .cross import speaker_means, trait_metrics
    from .mupta import mupta_cfg, scan as mscan
    cfg, dev = P.cfg, P.dev
    P.state.update(stage="Phase 2: MuPTA zero-shot (loading features)")
    info = mscan(cfg)
    mc = mupta_cfg(cfg)
    blocks, clips = load_blocks(mc, info["clips"])
    dur = np.array([float(np.load(Path(mc["paths"]["cache"]) / "audio" / f"{c}.npz")["dur"]) for c in clips])
    keep = dur >= cfg["mupta"]["min_duration_s"]
    spk_of = info["speaker"]
    keep &= np.array([spk_of[c] in info["labels"] for c in clips])
    blocks = {k: v[keep] for k, v in blocks.items()}
    clips = [c for c, k in zip(clips, keep) if k]
    inv = cfg["mupta"]["invert_neuroticism"]

    def lab(s):
        v = np.array([info["labels"][s][t] for t in TRAITS], np.float32)
        if inv:
            v[4] = 1 - v[4]
        return v

    n = int(cfg.get("phase2", {}).get("mupta_draws", 3))
    rows = []
    for tag in tags_of(P.splits, "S0R", n) + tags_of(P.splits, "S2A", n):
        ctx = P.ctx(tag, light=True)
        P.state.update(stage=f"Phase 2: MuPTA zero-shot ({tag})")
        for name in ("SG-MBF", "GMU", "Ungated sum + LN"):
            for s in P.seeds:
                lm = load_model(ctx, "fusion", name, "TAV", s, dev)
                if lm is None:
                    continue
                m, std, spec = lm
                with torch.no_grad():
                    p, _, _ = predict(m, to_dev(std(assemble(blocks, spec)), dev))
                spk, Ps = speaker_means(p, clips, spk_of)
                r = trait_metrics(np.stack([lab(x) for x in spk]), Ps)["mean"]
                rows.append({"split": tag, "model": name, "seed": s, "n_speakers": len(spk), "ccc": r["ccc"],
                             "r": r["r"], "mae": r["mae"]})
    return rows


# ----------------------------------------------------------------------------------------------- report
def report(out, md):
    import pandas as pd
    L = ["# Phase 2: secondary analyses with the models of the leakage audit", "",
         "S0R = random clip-level splits (leaky); S2A = identity-disjoint splits (leak-free). Values are means over "
         "split draws; SD across draws in brackets (each draw first averaged over seeds)."]

    def ms(x, k=1.0, f="{:.3f}"):
        x = np.asarray(x, float)
        if len(x) == 0:
            return "-"
        return f.format(k * x.mean()) + (f" ({f.format(k * x.std(ddof=1))})" if len(x) > 1 else "")

    def load(name):
        p = out / f"{name}.csv"
        return pd.read_csv(p) if p.exists() and p.stat().st_size > 5 else None

    d = load("seen_unseen")
    if d is not None:
        d = d[d.split != "S0"]
        L += ["", "## Seen vs unseen videos inside the leaky splits (S0R)", "",
              "| Model | CCC seen video | CCC unseen video | Acc seen | Acc unseen | n unseen per draw |", "|---|---|---|---|---|---|"]
        for mname in dict.fromkeys(d.model):
            a = d[(d.model == mname) & (d.subset == "seen video")]
            b = d[(d.model == mname) & (d.subset == "unseen video")]
            L.append(f"| {mname} | {ms(a.ccc)} | {ms(b.ccc)} | {ms(a.acc, 100, '{:.2f}')} | {ms(b.acc, 100, '{:.2f}')} | {b.n.mean():.0f} |")
    d = load("robustness")
    if d is not None:
        g = d.groupby(["split", "model", "perturbation", "level"], sort=False)[["acc", "ccc", "g_t", "g_a", "g_v"]].mean().reset_index()
        L += ["", "## Robustness on identity-disjoint splits (S2A): SG-MBF vs the same model without gate", "",
              "| Perturbation | Level | draws | SG-MBF CCC | Ungated + LN CCC | Δ CCC (gate) | SG-MBF Acc | Ungated + LN Acc | gates t / a / v |",
              "|---|---|---|---|---|---|---|---|---|"]
        for (kind, lv), grp in g.groupby(["perturbation", "level"], sort=False):
            a = grp[grp.model == "SG-MBF"].set_index("split")
            c = grp[grp.model == "Ungated sum + LN"].set_index("split")
            common = [s for s in a.index if s in c.index]
            if not common:
                continue
            a, c = a.loc[common], c.loc[common]
            L.append(f"| {kind} | {lv} | {len(common)} | {ms(a.ccc)} | {ms(c.ccc)} | {ms(a.ccc.values - c.ccc.values)} | "
                     f"{ms(a.acc, 100, '{:.2f}')} | {ms(c.acc, 100, '{:.2f}')} | "
                     f"{a.g_t.mean():.2f} / {a.g_a.mean():.2f} / {a.g_v.mean():.2f} |")
    d = load("fairness")
    if d is not None:
        d["family"] = d.split.map(family)
        L += ["", "## Fairness: mean absolute error x100 per group", "",
              "| Model | Attribute | Group | n per draw | S0R (leaky) | S2A (leak-free) |", "|---|---|---|---|---|---|"]
        for (mname, attr, grp), x in d.groupby(["model", "attribute", "group"], sort=False):
            L.append(f"| {mname} | {attr} | {grp} | {x.n.mean():.0f} | {ms(x[x.family == 'S0R'].mae, 100, '{:.2f}')} | "
                     f"{ms(x[x.family == 'S2A'].mae, 100, '{:.2f}')} |")
    d = load("representation_probe")
    if d is not None:
        d["family"] = d.split.map(family)
        g = d.groupby(["family", "split", "model", "representation"], sort=False)[["top1", "chance", "n_queries"]].mean().reset_index()
        L += ["", "## Identity in the learned representation (top-1 same-video retrieval among test clips, %)", "",
              "| Model | Representation | S0R (leaky) | S2A (leak-free) | chance S0R / S2A | queries S0R / S2A |", "|---|---|---|---|---|---|"]
        for (mname, rep), x in g.groupby(["model", "representation"], sort=False):
            a, b = x[x.family == "S0R"], x[x.family == "S2A"]
            L.append(f"| {mname} | {rep} | {ms(a.top1, 100, '{:.1f}')} | {ms(b.top1, 100, '{:.1f}')} | "
                     f"{100 * a.chance.mean():.2f} / {100 * b.chance.mean():.2f} | {a.n_queries.mean():.0f} / {b.n_queries.mean():.0f} |")
        L += ["", "The two families have different query sets (a random split scatters the clips of a video), so compare "
              "each learned representation with the input features of the same column, not the columns with each other."]
    d = load("mupta_zero_shot")
    if d is not None:
        d["family"] = d.split.map(family)
        g = d.groupby(["family", "split", "model"], sort=False)[["ccc", "r"]].mean().reset_index()
        L += ["", "## Zero-shot transfer to MuPTA (speaker level, mean over traits)", "",
              "| Model | CCC, trained on S0R | CCC, trained on S2A | r, trained on S0R | r, trained on S2A |", "|---|---|---|---|---|"]
        for mname, x in g.groupby("model", sort=False):
            a, b = x[x.family == "S0R"], x[x.family == "S2A"]
            L.append(f"| {mname} | {ms(a.ccc)} | {ms(b.ccc)} | {ms(a.r)} | {ms(b.r)} |")
    md.write_text("\n".join(L), encoding="utf-8")


class Phase:
    def __init__(self, cfg, log, state):
        self.cfg, self.log, self.state = cfg, log, state
        self.base = Ctx(cfg, log, state)
        self.splits, self.ident = load_splits(cfg, self.base)
        self.seeds = cfg["training"]["seeds"]
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self._last = (None, None)

    def ctx(self, tag, light=False):
        """Split context (one kept in memory). light=True: run directory only, no features loaded."""
        if light:
            c = SplitCtx.__new__(SplitCtx)
            c.cfg, c.runs = self.cfg, Path(self.cfg["paths"]["runs"]) / "leakage" / tag
            return c
        if self._last[0] != tag:
            self._last = (tag, SplitCtx(self.cfg, self.log, self.state, self.base, tag, self.splits[tag]))
        return self._last[1]


FILES = {"seen": "seen_unseen", "robust": "robustness", "fair": "fairness", "probe": "representation_probe",
         "mupta": "mupta_zero_shot"}
FUNCS = {"seen": part_seen, "robust": part_robust, "fair": part_fair, "probe": part_probe, "mupta": part_mupta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default=",".join(PARTS))
    args = ap.parse_args()
    cfg = load_config()
    log = get_logger("phase2", cfg)
    state = State(cfg)
    write_pid(cfg, "job")
    try:
        import pandas as pd
        state.update(job="phase2", status="running", stage="Phase 2: loading", error=None, started=time.time())
        P = Phase(cfg, log, state)
        if not tags_of(P.splits, "S2A"):
            raise RuntimeError("No S2A splits: run steps 11 and 12 first.")
        out = Path(cfg["paths"]["results"]) / "phase2"
        out.mkdir(parents=True, exist_ok=True)
        failed = []
        for part in [p.strip() for p in args.parts.split(",") if p.strip()]:
            try:
                rows = FUNCS[part](P)
                if rows:
                    pd.DataFrame(rows).to_csv(out / f"{FILES[part]}.csv", index=False)
                log.info("Phase 2: part '%s' done (%d rows)", part, len(rows))
            except Exception:
                failed.append(part)
                log.error("Phase 2: part '%s' failed\n%s", part, traceback.format_exc())
            report(out, Path(cfg["paths"]["results"]) / "phase2.md")
        if failed:
            raise RuntimeError("Phase 2 parts failed: " + ", ".join(failed) + " (see logs/phase2.log); the other "
                               "parts are in results/phase2.md")
        state.update(status="finished", stage="Phase 2 finished (results/phase2.md)", finished=time.time())
    except Exception as ex:
        log.error(traceback.format_exc())
        state.update(status="error", error=repr(ex))
        raise
    finally:
        remove_pid(cfg, "job")


if __name__ == "__main__":
    main()
