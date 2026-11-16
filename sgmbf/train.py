"""Training loop (Algorithm 1). Features live entirely on the GPU: one epoch takes well under a second."""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import per_trait
from .models import build, n_params


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_dev(X, dev):
    return {m: torch.as_tensor(x, device=dev) for m, x in X.items()}


@torch.no_grad()
def predict(model, X, bs=1024, modality_mask=None):
    model.eval()
    n = len(next(iter(X.values())))
    preds, gates, norms = [], [], []
    for i in range(0, n, bs):
        xb = {m: x[i:i + bs] for m, x in X.items()}
        mm = None if modality_mask is None else modality_mask[i:i + bs]
        p, ex = model(xb, mm) if mm is not None else model(xb)
        preds.append(p.float())
        if "gates" in ex:
            gates.append(ex["gates"].float())
        if "norms" in ex:
            norms.append(ex["norms"].float())
    cat = lambda L: torch.cat(L).cpu().numpy() if L else None
    return cat(preds), cat(gates), cat(norms)


def train_one(cfg, name, Xtr, ytr, Xva, yva, seed, run_dir, state=None, label="", extra_meta=None,
              gate_variant=None):
    """Train one model; returns (model, info). Logs one JSON line per epoch to run_dir/metrics.jsonl."""
    t = cfg["training"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    set_seed(seed)
    dims = {m: x.shape[1] for m, x in Xtr.items()}
    model = build(name, dims, cfg, gate_variant).to(dev)
    from .models import Head
    with torch.no_grad():  # start every output at the training-label mean
        for mod in model.modules():
            if isinstance(mod, Head):
                mod.out.bias.copy_(torch.as_tensor(ytr.mean(0)))
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=t["lr_factor"], patience=t["lr_patience"])
    use_amp = t["amp"] and dev == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    Xtr_d, Xva_d = to_dev(Xtr, dev), to_dev(Xva, dev)
    ytr_d = torch.as_tensor(ytr, device=dev)
    n, bs = len(ytr), t["batch_size"]
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {"model": name, "seed": seed, "gate_variant": getattr(model, "variant", None), "label": label, "dims": dims, "params": n_params(model),
            "n_train": int(n), "n_val": int(len(yva)), "started": time.time(), **(extra_meta or {})}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    log = open(run_dir / "metrics.jsonl", "w", encoding="utf-8")
    best, best_state, bad = float("inf"), None, 0
    gen = torch.Generator(device=dev).manual_seed(seed)
    for ep in range(1, t["max_epochs"] + 1):
        model.train()
        perm = torch.randperm(n, device=dev, generator=gen)
        tot, cnt = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:  # BatchNorm needs >1 sample
                continue
            xb = {m: x[idx] for m, x in Xtr_d.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                p, _ = model(xb)
                loss = F.mse_loss(p.float(), ytr_d[idx])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(idx)
            cnt += len(idx)
        pv, gv, _ = predict(model, Xva_d)
        vloss = float(np.mean((pv - yva) ** 2))
        m = per_trait(yva, pv)["mean"]
        sched.step(vloss)
        rec = {"epoch": ep, "train_loss": tot / max(cnt, 1), "val_loss": vloss, "val_acc": m["acc"],
               "val_ccc": m["ccc"], "lr": opt.param_groups[0]["lr"], "t": time.time()}
        if gv is not None and gv.ndim == 2:
            rec["val_gates"] = gv.mean(0).round(4).tolist()
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if state is not None:
            state.update(current_run=str(run_dir.name), epoch=ep, max_epochs=t["max_epochs"],
                         last_val_acc=m["acc"], last_val_ccc=m["ccc"])
        if vloss < best - 1e-7:
            best, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_ep = ep
        else:
            bad += 1
            if bad >= t["early_stop_patience"]:
                break
    log.close()
    model.load_state_dict(best_state)
    meta.update(best_epoch=best_ep, best_val_loss=best, epochs_run=ep, finished=time.time())
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return model, meta
