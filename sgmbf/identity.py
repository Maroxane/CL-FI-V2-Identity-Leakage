"""Reconstruct speaker identities in ChaLearn First Impressions V2 and build leakage-controlled splits.

    python -m sgmbf.identity

1. Embeddings per clip (cached in cache/identity.npz):
   - face  : FaceNet (InceptionResnetV1, VGGFace2) on 3 face crops (MediaPipe face detection), averaged;
   - voice : WavLM-base-plus speaker-verification x-vector on the first 10 s of audio.
2. Thresholds calibrated on the data itself: clips of the SAME YouTube video are (almost always) the same
   person -> positive pairs; random pairs of DIFFERENT videos are (almost always) different people ->
   negative pairs. Each modality's threshold = 99.9th percentile of the negative similarities.
3. Candidate identities = connected components of: same-video links + cross-video links where BOTH face and voice
   exceed their thresholds (or one modality exceeds its 99.99th percentile when the other is missing).
   Connected components chain (two 60+ clip components were found to be false merges by manual checking),
   so final identities = average-link clustering of videos on link density (refine(), tau = 0.25).
4. Splits written to splits/: S1_k = video-disjoint, S2_k = identity-disjoint with the single-link components
   (over-merged, conservative), S2A_k = identity-disjoint with the refined identities (k = 0, 1, 2), 60/20/20
   by clips, groups assigned whole. The official split is S0.
5. results/identity_report.json and results/identity_verification.html (pairs to check by eye).
"""
import base64
import io
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .common import ROOT, Progress, State, get_logger, load_config, remove_pid, write_pid
from .data import make_splits, scan, youtube_id
from .media import load_audio, sample_frames


# ----------------------------------------------------------------------------------------- models
def _ensure_facenet():
    try:
        import facenet_pytorch  # noqa: F401
    except ImportError:  # its pinned torch range conflicts with ours: install without dependencies
        subprocess.check_call([sys.executable, "-m", "pip", "install", "facenet-pytorch==2.6.0", "--no-deps"])


def load_hf(cls, name):
    """transformers models whose Hub weights are pickle-only: safetensors first, then tensors-only torch.load."""
    import torch
    try:
        return cls.from_pretrained(name, use_safetensors=True)
    except Exception:
        pass
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig
    m = cls(AutoConfig.from_pretrained(name))
    sd = torch.load(hf_hub_download(name, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    bad = [k for k in missing if "position_ids" not in k]
    if bad:
        raise RuntimeError(f"{name}: missing weights {bad[:5]}")
    return m


class Embedder:
    def __init__(self, cfg):
        import torch
        _ensure_facenet()
        from facenet_pytorch import InceptionResnetV1
        from transformers import AutoFeatureExtractor, WavLMForXVector
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.face = InceptionResnetV1(pretrained="vggface2").eval().to(self.dev)
        name = "microsoft/wavlm-base-plus-sv"
        self.fe = AutoFeatureExtractor.from_pretrained(name)
        self.voice = load_hf(WavLMForXVector, name).eval().to(self.dev)
        import mediapipe as mp
        self.det = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.sr = cfg["extraction"]["sample_rate"]

    def crops(self, path, n=3):
        import cv2
        frames, _ = sample_frames(path, fps=2.0, max_frames=40)
        if not frames:
            return []
        out = []
        pick = range(len(frames)) if len(frames) <= n else np.linspace(0, len(frames) - 1, n + 2).round().astype(int)[1:-1]
        for i in pick:
            f = frames[int(i)]
            r = self.det.process(f)
            if not r.detections:
                continue
            b = max(r.detections, key=lambda d: d.score[0]).location_data.relative_bounding_box
            h, w = f.shape[:2]
            cx, cy, s = (b.xmin + b.width / 2) * w, (b.ymin + b.height / 2) * h, max(b.width * w, b.height * h) * 1.2
            x0, y0, x1, y1 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2)), int(min(w, cx + s / 2)), int(min(h, cy + s / 2))
            if x1 - x0 < 20 or y1 - y0 < 20:
                continue
            out.append(cv2.resize(f[y0:y1, x0:x1], (160, 160), interpolation=cv2.INTER_AREA))
        return out

    def face_vec(self, crops):
        t = self.torch
        if not crops:
            return None
        x = t.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float().to(self.dev)
        with t.no_grad():
            e = self.face((x - 127.5) / 128.0)
        e = t.nn.functional.normalize(e, dim=1).mean(0)
        return t.nn.functional.normalize(e, dim=0).cpu().numpy()

    def voice_vec(self, path):
        t = self.torch
        wav = load_audio(path, self.sr)[: 10 * self.sr]
        if wav.size < self.sr or np.abs(wav).max() < 1e-4:
            return None
        inp = self.fe(wav, sampling_rate=self.sr, return_tensors="pt").to(self.dev)
        with t.no_grad():
            e = self.voice(**inp).embeddings
        return t.nn.functional.normalize(e[0], dim=0).cpu().numpy()


def embed_all(cfg, info, clips, state, log):
    path = Path(cfg["paths"]["cache"]) / "identity.npz"
    done = {}
    if path.exists():
        z = np.load(path, allow_pickle=True)
        done = {c: (f, v) for c, f, v in zip(z["clips"], z["face"], z["voice"])}
    todo = [c for c in clips if c not in done]
    if todo:
        emb = Embedder(cfg)
        pr = Progress(state, "Identity embeddings (face + voice)", len(todo))
        nan = np.full(512, np.nan, np.float32)
        for i, c in enumerate(todo):
            try:
                f = emb.face_vec(emb.crops(info["videos"][c]))
                v = emb.voice_vec(info["videos"][c])
            except Exception as ex:  # one broken clip must not stop the whole pass
                log.warning("identity: %s failed (%s)", c, ex)
                f = v = None
            done[c] = (nan if f is None else f.astype(np.float32), nan if v is None else v.astype(np.float32))
            pr.step(msg=c)
            if (i + 1) % 500 == 0 or i + 1 == len(todo):
                ks = sorted(done)
                np.savez(path, clips=np.array(ks), face=np.stack([done[k][0] for k in ks]),
                         voice=np.stack([done[k][1] for k in ks]))
    ks = list(clips)
    return np.stack([done[k][0] for k in ks]), np.stack([done[k][1] for k in ks])


# --------------------------------------------------------------------------------------- clustering
class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def calibrate(E, vids, rng, n_neg=200_000):
    ok = ~np.isnan(E[:, 0])
    idx = np.where(ok)[0]
    by = defaultdict(list)
    for i in idx:
        by[vids[i]].append(i)
    pos = [float(E[a] @ E[b]) for g in by.values() for j, a in enumerate(g) for b in g[j + 1:]]
    a, b = rng.choice(idx, n_neg), rng.choice(idx, n_neg)
    keep = vids[a] != vids[b]
    neg = np.einsum("ij,ij->i", E[a[keep]], E[b[keep]])
    return {"thr": float(np.quantile(neg, 0.999)), "thr_strict": float(np.quantile(neg, 0.9999)),
            "pos_median": float(np.median(pos)), "neg_median": float(np.median(neg)),
            "pos_recall_at_thr": float(np.mean(np.array(pos) > np.quantile(neg, 0.999))), "n_pos": len(pos)}


def candidate_pairs(F, V, vids, cf, cv, chunk=1024):
    """Cross-video pairs passing both thresholds (or one strict threshold when the other modality is missing)."""
    n = len(F)
    fok, vok = ~np.isnan(F[:, 0]), ~np.isnan(V[:, 0])
    F0, V0 = np.nan_to_num(F), np.nan_to_num(V)
    out = []
    for s in range(0, n, chunk):
        sf, sv = F0[s:s + chunk] @ F0.T, V0[s:s + chunk] @ V0.T
        for k in range(sf.shape[0]):
            i = s + k
            js = np.arange(i + 1, n)
            f, v = sf[k, i + 1:], sv[k, i + 1:]
            both = fok[i] & fok[js] & vok[i] & vok[js]
            m = (both & (f > cf["thr"]) & (v > cv["thr"])) | \
                (~both & (((f > cf["thr_strict"]) & fok[i] & fok[js]) | ((v > cv["thr_strict"]) & vok[i] & vok[js])))
            m &= vids[js] != vids[i]
            for j in js[m]:
                out.append((i, int(j), float(sf[k, j]), float(sv[k, j])))
    return out


def near_misses(F, V, vids, cf, cv, rng, n=150):
    """Cross-video pairs just below the decision rule (for the verification page)."""
    idx = np.where(~np.isnan(F[:, 0]) & ~np.isnan(V[:, 0]))[0]
    res = []
    for _ in range(60):
        a = rng.choice(idx, 2000)
        sf, sv = F[a] @ F[idx].T, V[a] @ V[idx].T
        band = (sf > cf["thr"] - 0.1) & (sv > cv["thr"] - 0.1) & ~((sf > cf["thr"]) & (sv > cv["thr"]))
        for r, c in zip(*np.where(band)):
            i, j = int(a[r]), int(idx[c])
            if vids[i] != vids[j]:
                res.append((i, j, float(sf[r, c]), float(sv[r, c])))
        if len(res) >= 5 * n:
            break
    rng.shuffle(res)
    return res[:n]


def refine(pairs, vids, tau=0.25):
    """Average-link clustering of VIDEOS, to break the chains that single-link (connected components) creates.
    Link density of two videos = linked clip pairs / all clip pairs. Two groups of videos are merged, best
    first, while their mean link density (unlinked video pairs count 0) is >= tau. Returns a group id per clip."""
    vl = sorted(set(vids))
    vi = {v: i for i, v in enumerate(vl)}
    nc = Counter(vids)
    W = Counter()
    for i, j, _, _ in pairs:
        a, b = sorted((vi[vids[i]], vi[vids[j]]))
        W[(a, b)] += 1.0 / (nc[vl[a]] * nc[vl[b]])
    cl = {i: {i} for i in range(len(vl))}
    of = list(range(len(vl)))
    while True:
        best = None
        for A, B in {(min(of[a], of[b]), max(of[a], of[b])) for a, b in W if of[a] != of[b]}:
            s = sum(W.get((min(a, b), max(a, b)), 0.0) for a in cl[A] for b in cl[B]) / (len(cl[A]) * len(cl[B]))
            if s >= tau and (best is None or (s, -A, -B) > best[0]):
                best = ((s, -A, -B), A, B)
        if best is None:
            break
        _, A, B = best
        cl[A] |= cl[B]
        for x in cl[B]:
            of[x] = A
        del cl[B]
    first = {}
    for i, v in enumerate(vids):
        first.setdefault(of[vi[v]], i)
    return [first[of[vi[v]]] for v in vids]


# -------------------------------------------------------------------------------------------- splits
def group_split(clips, group_of, seed, frac=(0.6, 0.2, 0.2)):
    groups = defaultdict(list)
    for c in clips:
        groups[group_of[c]].append(c)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    keys.sort(key=lambda k: -len(groups[k]))  # big groups first, then fill greedily (stable shuffle order)
    target = {s: f * len(clips) for s, f in zip(("train", "val", "test"), frac)}
    out = {s: [] for s in target}
    for k in keys:
        s = max(target, key=lambda s: (target[s] - len(out[s])) / target[s])
        out[s] += groups[k]
    return {s: sorted(v) for s, v in out.items()}


def thumb(path):
    import cv2
    fr, _ = sample_frames(path, fps=1.0, max_frames=3)
    if not fr:
        return ""
    f = fr[len(fr) // 2]
    h, w = f.shape[:2]
    f = cv2.resize(f, (160, int(160 * h / w)))
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def verification_page(info, clips, merged, missed, out):
    rows = []
    for kind, pairs in (("merged", merged), ("near-miss", missed)):
        for i, j, sf, sv in pairs:
            a, b = clips[i], clips[j]
            rows.append(f"<tr data-kind='{kind}'><td><img src='{thumb(info['videos'][a])}'><br><small>{a}</small></td>"
                        f"<td><img src='{thumb(info['videos'][b])}'><br><small>{b}</small></td>"
                        f"<td>{kind}<br>face {sf:.2f}<br>voice {sv:.2f}</td>"
                        f"<td><label><input type=radio name='{a}|{b}' value=same> same person</label><br>"
                        f"<label><input type=radio name='{a}|{b}' value=diff> different</label></td></tr>")
    html = """<!doctype html><meta charset=utf-8><title>Identity verification</title>
<style>body{font-family:sans-serif;margin:24px}td{padding:6px;border-bottom:1px solid #ddd;vertical-align:top}img{width:160px}
#bar{position:sticky;top:0;background:#fff;padding:8px 0;border-bottom:2px solid #333}</style>
<div id=bar><b>Identity verification</b> - judge each pair (audio is not shown: use the face).
<span id=stats></span> <button onclick=dl()>Export CSV</button></div><table>""" + "".join(rows) + """</table>
<script>
function stats(){let m={merged:[0,0],'near-miss':[0,0]};document.querySelectorAll('tr').forEach(r=>{let c=r.querySelector('input:checked');if(c){m[r.dataset.kind][0]+=c.value=='same'?1:0;m[r.dataset.kind][1]++}});
document.getElementById('stats').textContent=` | merged pairs judged same: ${m.merged[0]}/${m.merged[1]} (precision) | near-misses judged same: ${m['near-miss'][0]}/${m['near-miss'][1]} (missed links)`}
document.addEventListener('change',stats);
function dl(){let s='pair,kind,answer\\n';document.querySelectorAll('tr').forEach(r=>{let c=r.querySelector('input:checked');if(c)s+=c.name+','+r.dataset.kind+','+c.value+'\\n'});
let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([s],{type:'text/csv'}));a.download='identity_verification.csv';a.click()}
</script>"""
    out.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------------------------- main
def main():
    cfg = load_config()
    log = get_logger("identity", cfg)
    state = State(cfg)
    write_pid(cfg, "job")
    try:
        state.update(job="identity", status="running", stage="Scanning dataset", error=None)
        info = scan(cfg)
        off, kind = make_splits(cfg, info, log)
        clips = sorted(c for s in off.values() for c in s)
        vids = np.array([youtube_id(c) for c in clips])
        F, V = embed_all(cfg, info, clips, state, log)
        rng = np.random.default_rng(0)
        cf, cv = calibrate(F, vids, rng), calibrate(V, vids, rng)
        log.info("face  calibration: %s", cf)
        log.info("voice calibration: %s", cv)
        state.update(stage="Linking clips across videos")
        pairs = candidate_pairs(F, V, vids, cf, cv)
        uf = UF(len(clips))
        first = {}
        for i, v in enumerate(vids):
            uf.union(i, first.setdefault(v, i))
        for i, j, _, _ in pairs:
            uf.union(i, j)
        comp = [uf.find(i) for i in range(len(clips))]
        ident_single = {c: f"id{comp[i]:05d}" for i, c in enumerate(clips)}
        tau = float(cfg.get("identity", {}).get("link_density", 0.25))
        comp_a = refine(pairs, list(vids), tau)
        ident = {c: f"id{comp_a[i]:05d}" for i, c in enumerate(clips)}   # refined (average-link) identities
        single_size = Counter(ident_single.values())
        size = Counter(ident.values())
        vids_per_id = defaultdict(set)
        for c in clips:
            vids_per_id[ident[c]].add(youtube_id(c))
        big = [(k, n) for k, n in size.most_common(5)]
        if big[0][1] > 0.005 * len(clips):
            log.warning("Largest identity holds %d clips (>0.5%% of the corpus): possible chaining, check the "
                        "verification page", big[0][1])
        # official-split overlap at identity level
        o_id = {s: {ident[c] for c in off[s]} for s in off}
        report = {
            "n_clips": len(clips), "n_videos": int(len(set(vids))), "n_identities": len(size),
            "identities_with_several_videos": sum(len(v) > 1 for v in vids_per_id.values()),
            "cross_video_links": len(pairs), "largest_identities": big,
            "clustering": {"rule": "average-link over videos", "link_density_tau": tau,
                           "single_link_n_identities": len(single_size),
                           "single_link_largest": single_size.most_common(5)},
            "face_missing": int(np.isnan(F[:, 0]).sum()), "voice_missing": int(np.isnan(V[:, 0]).sum()),
            "calibration": {"face": cf, "voice": cv},
            "official_split": {
                "test_clips_with_train_video": sum(youtube_id(c) in {youtube_id(x) for x in off["train"]} for c in off["test"]),
                "test_clips_with_train_identity": sum(ident[c] in o_id["train"] for c in off["test"]),
                "n_test": len(off["test"])},
        }
        out = Path(cfg["paths"]["results"])
        sp = ROOT / "splits"
        sp.mkdir(exist_ok=True)
        (sp / "identities.json").write_text(json.dumps(ident))
        (sp / "identities_single_link.json").write_text(json.dumps(ident_single))
        # S2 = single-link identities (over-merged, kept for continuity); S2A = refined average-link identities
        for k in range(3):
            for tag, g in (("S1", {c: youtube_id(c) for c in clips}), ("S2", ident_single), ("S2A", ident)):
                s = group_split(clips, g, seed=100 + k)
                ov = {"train_test": len({g[c] for c in s["train"]} & {g[c] for c in s["test"]}),
                      "train_test_identity": len({ident[c] for c in s["train"]} & {ident[c] for c in s["test"]})}
                (sp / f"{tag}_{k}.json").write_text(json.dumps({"splits": s, "group_overlap": ov}))
                report[f"{tag}_{k}"] = {**{x: len(v) for x, v in s.items()}, **ov}
        (out / "identity_report.json").write_text(json.dumps(report, indent=1))
        state.update(stage="Writing verification page")
        merged = random.Random(1).sample(pairs, min(150, len(pairs)))
        verification_page(info, clips, merged, near_misses(F, V, vids, cf, cv, rng), out / "identity_verification.html")
        log.info("Identity report: %s", json.dumps({k: report[k] for k in ("n_videos", "n_identities",
                 "identities_with_several_videos", "cross_video_links", "official_split")}))
        state.update(status="finished", stage="Identities and splits written (splits/, results/identity_report.json)")
    except Exception as ex:
        import traceback
        log.error(traceback.format_exc())
        state.update(status="error", error=repr(ex))
        raise
    finally:
        remove_pid(cfg, "job")


if __name__ == "__main__":
    main()
