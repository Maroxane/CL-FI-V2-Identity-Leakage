"""Audio decoding (ffmpeg) and frame sampling (OpenCV)."""
import subprocess

import numpy as np


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def load_audio(path, sr=16000):
    """Mono float32 waveform in [-1, 1] at `sr` Hz (empty array if the clip has no audio)."""
    cmd = [ffmpeg_exe(), "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
           "-ar", str(sr), "-f", "s16le", "-"]
    flags = 0x08000000 if hasattr(subprocess, "CREATE_NO_WINDOW") else 0  # no console window on Windows
    out = subprocess.run(cmd, capture_output=True, creationflags=flags).stdout
    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0


def add_noise(wav, snr_db, noise=None, rng=None):
    rng = rng or np.random.default_rng(0)
    if wav.size == 0:
        return wav
    if noise is None:
        n = rng.standard_normal(wav.size).astype(np.float32)
    else:
        reps = int(np.ceil(wav.size / noise.size))
        start = rng.integers(0, max(1, noise.size))
        n = np.tile(noise, reps + 1)[start:start + wav.size].astype(np.float32)
    ps = np.mean(wav ** 2) + 1e-12
    pn = np.mean(n ** 2) + 1e-12
    return (wav + n * np.sqrt(ps / (pn * 10 ** (snr_db / 10)))).astype(np.float32)


def sample_frames(path, fps=5.0, max_frames=150, return_fps=False):
    """RGB frames sampled at `fps` (uint8, HxWx3) and the clip duration in seconds.

    Clips longer than max_frames / fps seconds are sampled uniformly over their whole duration;
    with return_fps=True the effective sampling rate is also returned (used for kinematics).
    """
    import cv2
    cap = cv2.VideoCapture(str(path))
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(vfps / fps)))
    if n and n / step > max_frames:
        step = int(np.ceil(n / max_frames))
    frames, i = [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            ok, fr = cap.retrieve()
            if ok:
                h, w = fr.shape[:2]
                if max(h, w) > 640:  # speed: MediaPipe does not need full HD
                    s = 640 / max(h, w)
                    fr = cv2.resize(fr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
                frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                if len(frames) >= max_frames:
                    break
        i += 1
    cap.release()
    dur = n / vfps if vfps else 0.0
    if return_fps:
        return frames, dur, vfps / step
    return frames, dur
