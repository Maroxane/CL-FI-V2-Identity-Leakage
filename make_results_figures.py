"""
make_results_figures.py  -  Figures 4-6 of the revised manuscript, drawn from YOUR re-run results.

Usage
    python make_results_figures.py --results results/ --out figures/

Expected files in --results (templates are provided in results_templates/):
    seeds.csv        model,seed,trait,acc,ccc,r,r2          (trait in O,C,E,A,N; one row per model x seed x trait)
                     model names: "Ours" plus ablation names such as "T+V", "A+V", "T+A", "T", "A", "V"
    published.csv    model,source,O,C,E,A,N,mean            (published test accuracies, 0-1 scale, with source)
    gates_test.csv   clip_id,O,C,E,A,N,g_t,g_a,g_v           (ground-truth labels + gate values per test clip, one seed)
    robustness.csv   perturbation,level,seed,acc,g_t,g_a,g_v (perturbation in {"audio_snr_db","frame_drop"})

Outputs (PDF + SVG vector, 600-dpi PNG): Fig4_results, Fig5_gates, Fig6_robustness.
Nothing is plotted that is not in the CSV files.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRAITS = ["O", "C", "E", "A", "N"]
TNAME = {"O": "Openness", "C": "Conscientiousness", "E": "Extraversion", "A": "Agreeableness", "N": "Neuroticism"}
COL = {"t": "#0072B2", "a": "#D55E00", "v": "#009E73", "ours": "#6A3D9A", "pub": "#9A9A9A"}

plt.rcParams.update({
    "font.family": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 7, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
})


def boot_ci(x, n=10000, seed=0):
    """95% percentile bootstrap CI of the mean over seeds."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return x.mean(), x.mean(), x.mean()
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def save(fig, out, name):
    for ext in ("pdf", "svg"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(out, f"{name}.png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig4(res, out):
    s = pd.read_csv(os.path.join(res, "seeds.csv"))
    pub = pd.read_csv(os.path.join(res, "published.csv"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 2.5), gridspec_kw={"width_ratios": [1.25, 1]})

    # (a) per-trait accuracy of Ours with 95% CI vs published points
    ours = s[s.model == "Ours"]
    x = np.arange(len(TRAITS) + 1)
    labels = TRAITS + ["Mean"]
    vals = [boot_ci(ours[ours.trait == t].acc) for t in TRAITS]
    seed_means = ours.groupby("seed").acc.mean()
    vals.append(boot_ci(seed_means))
    m = np.array([v[0] for v in vals]) * 100
    lo = np.array([v[1] for v in vals]) * 100
    hi = np.array([v[2] for v in vals]) * 100
    a.errorbar(x, m, yerr=[m - lo, hi - m], fmt="o", ms=3.5, color=COL["ours"], capsize=2, lw=0.9,
               label="Ours (mean, 95% CI, 5 seeds)", zorder=3)
    markers = ["s", "^", "D", "v", "P", "X", "<", ">"]
    for i, row in pub.reset_index(drop=True).iterrows():
        y = [row[t] for t in TRAITS] + [row["mean"]]
        y = np.array(y, float) * 100
        a.scatter(x + 0.18, y, s=9, marker=markers[i % len(markers)], color=COL["pub"],
                  edgecolor="none", label=f"{row.model}", zorder=2)
    a.set_xticks(x, labels)
    a.set_ylabel("Accuracy, 1 - MAE (%)")
    a.set_title("(a) Test accuracy per trait", loc="left", fontsize=7, fontweight="bold")
    a.legend(fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)

    # (b) modality ablation: drop vs full model, mean and 95% CI over seeds
    full = s[s.model == "Ours"].groupby("seed").acc.mean()
    rows = []
    for mdl in [mm for mm in s.model.unique() if mm != "Ours"]:
        sub = s[s.model == mdl].groupby("seed").acc.mean()
        common = full.index.intersection(sub.index)
        d = (sub.loc[common] - full.loc[common]) * 100
        rows.append((mdl, *boot_ci(d)))
    rows.sort(key=lambda r: r[1])
    yy = np.arange(len(rows))
    b.barh(yy, [r[1] for r in rows], color="#BBBBBB", height=0.6, zorder=2)
    b.errorbar([r[1] for r in rows], yy, xerr=[[r[1] - r[2] for r in rows], [r[3] - r[1] for r in rows]],
               fmt="none", ecolor="#333333", capsize=2, lw=0.8, zorder=3)
    b.axvline(0, color="#333333", lw=0.6)
    b.set_yticks(yy, [r[0] for r in rows])
    b.set_xlabel("Change in mean accuracy vs. T+A+V (points)")
    b.set_title("(b) Modality ablation", loc="left", fontsize=7, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "Fig4_results")


def fig5(res, out):
    g = pd.read_csv(os.path.join(res, "gates_test.csv"))
    fig, axes = plt.subplots(1, 5, figsize=(7.16, 2.0), sharey=True)
    for ax, t in zip(axes, TRAITS):
        q = pd.qcut(g[t], 3, labels=["Low", "Mid", "High"])
        pos = 0
        ticks = []
        for lvl in ["Low", "Mid", "High"]:
            d = g[q == lvl]
            for k, (gc, c) in enumerate((("g_t", COL["t"]), ("g_a", COL["a"]), ("g_v", COL["v"]))):
                bp = ax.boxplot(d[gc], positions=[pos + k * 0.28], widths=0.22, patch_artist=True,
                                showfliers=False, medianprops=dict(color="black", lw=0.7),
                                whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
                bp["boxes"][0].set(facecolor=c, alpha=0.75, lw=0.5)
            ticks.append(pos + 0.28)
            pos += 1.1
        ax.set_xticks(ticks, ["Low", "Mid", "High"])
        ax.set_title(TNAME[t], fontsize=6.6)
        ax.set_xlabel("Label tertile", fontsize=6)
    axes[0].set_ylabel("Gate value")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=COL["t"], label="text $g_t$"), Patch(color=COL["a"], label="voice $g_a$"),
                        Patch(color=COL["v"], label="visual $g_v$")], ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.07), fontsize=6)
    fig.tight_layout()
    save(fig, out, "Fig5_gates")


def fig6(res, out):
    r = pd.read_csv(os.path.join(res, "robustness.csv"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 2.3))
    spec = {"audio_snr_db": ("Audio SNR (dB)", "g_a", COL["a"]), "frame_drop": ("Frame-drop rate", "g_v", COL["v"])}
    for pert, (lab, gc, c) in spec.items():
        d = r[r.perturbation == pert]
        if d.empty:
            continue
        agg = d.groupby("level").agg(acc=("acc", "mean"), acc_sd=("acc", "std"),
                                     g=(gc, "mean"), g_sd=(gc, "std")).reset_index()
        x = np.arange(len(agg))
        a.errorbar(x, agg.acc * 100, yerr=agg.acc_sd * 100, marker="o", ms=3, color=c, lw=0.9, capsize=2, label=lab)
        b.errorbar(x, agg.g, yerr=agg.g_sd, marker="o", ms=3, color=c, lw=0.9, capsize=2,
                   label=f"{lab}: gate of corrupted modality")
        for ax in (a, b):
            ax.set_xticks(x, agg.level.astype(str))
    a.set_ylabel("Mean accuracy (%)")
    a.set_xlabel("Perturbation level (left = clean)")
    a.set_title("(a) Accuracy under corruption", loc="left", fontsize=7, fontweight="bold")
    b.set_ylabel("Mean gate value")
    b.set_xlabel("Perturbation level (left = clean)")
    b.set_title("(b) Gate response", loc="left", fontsize=7, fontweight="bold")
    a.legend(fontsize=5.8)
    b.legend(fontsize=5.8)
    fig.tight_layout()
    save(fig, out, "Fig6_robustness")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig4(args.results, args.out)
    fig5(args.results, args.out)
    fig6(args.results, args.out)
    print("Figures written to", args.out)
