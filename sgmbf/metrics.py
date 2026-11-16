"""Evaluation metrics (Eq. 10-12) and the statistics of Section 4.5."""
import numpy as np

from .common import TRAITS


def per_trait(y, p):
    """y, p: (N, 5). Returns dict trait -> {acc, ccc, r, r2, mae}."""
    out = {}
    for j, t in enumerate(TRAITS):
        a, b = y[:, j].astype(np.float64), p[:, j].astype(np.float64)
        mae = np.mean(np.abs(a - b))
        ma, mb, va, vb = a.mean(), b.mean(), a.var(), b.var()
        cov = np.mean((a - ma) * (b - mb))
        r = cov / (np.sqrt(va * vb) + 1e-12)
        ccc = 2 * cov / (va + vb + (ma - mb) ** 2 + 1e-12)
        r2 = 1 - np.sum((a - b) ** 2) / (np.sum((a - ma) ** 2) + 1e-12)
        out[t] = {"acc": 1 - mae, "ccc": ccc, "r": r, "r2": r2, "mae": mae}
    out["mean"] = {k: float(np.mean([out[t][k] for t in TRAITS])) for k in ("acc", "ccc", "r", "r2", "mae")}
    return out


def _mean_acc(y, p):
    return 1 - np.mean(np.abs(y - p))


def bootstrap_ci(y, preds, metric="acc", n=10000, seed=0):
    """preds: list of (N, 5) arrays (one per seed). Resample clips, pool seeds. Returns (mean, lo, hi)."""
    rng = np.random.default_rng(seed)
    N = len(y)
    P = np.stack(preds)  # (S, N, 5)
    f = _mean_acc if metric == "acc" else _fast_ccc
    point = float(np.mean([f(y, p) for p in P]))
    vals = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, N, N)
        vals[i] = np.mean([f(y[idx], p[idx]) for p in P])
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _fast_ccc(y, p):
    ma, mb = y.mean(0), p.mean(0)
    va, vb = y.var(0), p.var(0)
    cov = ((y - ma) * (p - mb)).mean(0)
    return float(np.mean(2 * cov / (va + vb + (ma - mb) ** 2 + 1e-12)))


def paired_bootstrap(y, preds_a, preds_b, n=10000, seed=0):
    """Difference in mean accuracy (A - B), 95% CI and two-sided p-value; seeds paired by index."""
    rng = np.random.default_rng(seed)
    A, B = np.stack(preds_a), np.stack(preds_b)
    ea = np.abs(A - y[None]).mean((0, 2))  # per-clip error averaged over seeds and traits
    eb = np.abs(B - y[None]).mean((0, 2))
    diff = eb - ea                          # positive = A better
    N = len(y)
    obs = diff.mean()
    boots = np.array([diff[rng.integers(0, N, N)].mean() for _ in range(n)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = float(np.mean(np.abs(boots - obs) >= abs(obs)))  # centred bootstrap test of H0: no difference
    p = min(1.0, max(p, 1.0 / n))
    return float(obs), float(lo), float(hi), float(p)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m, running = len(p), 0.0
    for k, i in enumerate(order):
        running = max(running, (m - k) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()
