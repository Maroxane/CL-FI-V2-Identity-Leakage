"""Per-frame facial geometry / body pose descriptors and their clip-level aggregation.

Per-frame values are cached so that frame dropping (robustness study) can be simulated
without re-running MediaPipe. Missing detections are NaN.
"""
import numpy as np

GEO_NAMES = ["mouth_open", "mouth_width", "eye_open_l", "eye_open_r", "brow_l", "brow_r",
             "asymmetry", "yaw", "pitch", "roll", "face_cx", "face_cy"]
POSE_IDX = [0, 11, 12, 13, 14, 15, 16]  # nose, shoulders, elbows, wrists (MediaPipe Pose)
N_GEO, N_POSE = len(GEO_NAMES), len(POSE_IDX) * 3


def _d(a, b):
    return float(np.linalg.norm(a - b))


def face_geometry(lm):
    """lm: (468, 3) normalised MediaPipe Face Mesh landmarks -> (12,) descriptor."""
    p = lambda i: lm[i, :2]
    h = _d(p(10), p(152)) + 1e-6
    eye_l, eye_r = _d(p(159), p(145)) / h, _d(p(386), p(374)) / h
    brow_l, brow_r = _d(p(105), p(159)) / h, _d(p(334), p(386)) / h
    eyes_w = _d(p(33), p(263)) + 1e-6
    mid_eye = (p(33) + p(263)) / 2
    asym = abs(eye_l - eye_r) + abs(brow_l - brow_r) + abs(_d(p(61), p(1)) - _d(p(291), p(1))) / h
    yaw = (_d(p(1), p(33)) - _d(p(1), p(263))) / eyes_w
    pitch = _d(p(1), mid_eye) / (_d(p(1), p(152)) + 1e-6)
    roll = float(np.arctan2(p(263)[1] - p(33)[1], p(263)[0] - p(33)[0]))
    c = lm[:, :2].mean(0)
    return np.array([_d(p(13), p(14)) / h, _d(p(61), p(291)) / h, eye_l, eye_r, brow_l, brow_r,
                     asym, yaw, pitch, roll, c[0], c[1]], np.float32)


def pose_vector(lms):
    """lms: (33, 4) MediaPipe Pose landmarks (x, y, z, visibility) -> (21,) [x, y, vis] x 7 joints."""
    return np.concatenate([[lms[i, 0], lms[i, 1], lms[i, 3]] for i in POSE_IDX]).astype(np.float32)


def _nanstats(x):
    if x.size == 0 or np.all(np.isnan(x)):
        return np.zeros(3 * x.shape[1], np.float32)
    return np.concatenate([np.nanmean(x, 0), np.nanstd(x, 0), np.nanmax(x, 0)]).astype(np.float32)


def aggregate_geometry(geo):
    """geo: (T, 12) -> (33,) = mean/std/max of 10 shape descriptors + detection rate + head motion."""
    geo = np.asarray(geo, np.float32).reshape(-1, N_GEO)
    ok = ~np.isnan(geo[:, 0])
    stats = _nanstats(geo[:, :10]) if ok.any() else np.zeros(30, np.float32)
    motion = np.nan_to_num(np.nanstd(geo[ok, 10:12], 0)) if ok.sum() > 1 else np.zeros(2, np.float32)
    out = np.concatenate([stats, [ok.mean() if len(ok) else 0.0], motion]).astype(np.float32)
    return np.nan_to_num(out)


def aggregate_pose(pose, fps):
    """pose: (T, 21) -> (21,) kinematic descriptors (see paper, Section 3.4)."""
    pose = np.asarray(pose, np.float32).reshape(-1, N_POSE)
    out = np.zeros(21, np.float32)
    ok = ~np.isnan(pose[:, 0])
    if len(pose) == 0:
        return out
    out[0] = ok.mean()
    P = pose[ok]
    if len(P) < 3:
        return out
    J = P.reshape(len(P), 7, 3)  # nose, lsh, rsh, lel, rel, lwr, rwr
    sw = np.median(np.linalg.norm(J[:, 1, :2] - J[:, 2, :2], axis=1)) + 1e-6
    nose = J[:, 0, :2] / sw
    out[1:3] = nose.max(0) - nose.min(0)
    out[3:5] = nose.std(0)
    v = np.linalg.norm(np.diff(nose, axis=0), axis=1) * fps
    out[5], out[6] = v.mean(), v.max()
    k = 7
    for sh, el in ((1, 3), (2, 4)):
        ang = np.unwrap(np.arctan2(J[:, el, 1] - J[:, sh, 1], J[:, el, 0] - J[:, sh, 0]))
        w = np.abs(np.diff(ang)) * fps
        out[k:k + 3] = [w.mean(), w.std(), w.max()]
        k += 3
    for wr in (5, 6):
        s = np.linalg.norm(np.diff(J[:, wr, :2] / sw, axis=0), axis=1) * fps
        out[k:k + 2] = [s.mean(), s.max()]
        k += 2
    out[17] = np.mean(J[:, 5, 1] < J[:, 1, 1])  # left wrist above shoulder (image y grows downwards)
    out[18] = np.mean(J[:, 6, 1] < J[:, 2, 1])
    rel = J[:, 0, 1] - (J[:, 1, 1] + J[:, 2, 1]) / 2
    kw = min(5, len(rel))  # very short clips (< 5 pose frames, e.g. MuPTA one-word answers)
    rel = rel - np.convolve(rel, np.ones(kw) / kw, mode="same")
    out[19] = np.sum(np.diff(np.sign(rel)) != 0) / 2 / (len(rel) / fps)  # nods per second
    out[20] = sw
    return np.nan_to_num(out)


def aggregate_emotion(emo, dim=512):
    """emo: (K, dim) per-frame embeddings -> (3*dim,) mean/std/max (1536 for Emo-AffectNet)."""
    emo = np.asarray(emo, np.float32)
    if emo.size == 0:
        return np.zeros(3 * dim, np.float32)
    return np.concatenate([emo.mean(0), emo.std(0), emo.max(0)]).astype(np.float32)


def drop_frames(x, rate, rng):
    """Keep a random (1 - rate) fraction of frames, preserving order (at least one frame)."""
    x = np.asarray(x)
    if rate <= 0 or len(x) == 0:
        return x
    keep = rng.random(len(x)) >= rate
    if not keep.any():
        keep[rng.integers(len(x))] = True
    return x[keep]
