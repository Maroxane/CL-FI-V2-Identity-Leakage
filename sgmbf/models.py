"""SG-MBF (Section 3.5-3.6) and the re-implemented fusion baselines (Section 4.4).

Every model returns (predictions (B, 5), extras dict). SG-MBF puts gates and projected norms in extras.
Modalities absent from `dims` are simply not part of the model (used for modality ablations).
"""
import torch
import torch.nn as nn


def projection(d_in, d_h):
    return nn.Sequential(nn.Linear(d_in, d_h), nn.BatchNorm1d(d_h), nn.ReLU(inplace=True))


class Head(nn.Module):
    """psi: two-layer MLP (512, 256) with BN, ReLU, dropout, then five linear outputs."""

    def __init__(self, d_h, hidden=(512, 256), dropout=0.3, n_out=5):
        super().__init__()
        layers, d = [], d_h
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d = h
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(d, n_out)

    def forward(self, z):
        return self.out(self.mlp(z))


GATE_VARIANTS = {
    "sigmoid": "g = sigmoid(w.[h;c] + b), b initialised at 0 (original SG-MBF, g starts at 0.5)",
    "sigmoid_open": "same gate, b initialised at +2 (g starts at 0.88: starts close to the ungated sum)",
    "scaled": "g = 2 sigmoid(.) in (0, 2), starts at 1: a modality can be amplified as well as attenuated",
    "normalized": "w_m = M g_m / sum_k g_k: the gate only redistributes weight, the fused scale is preserved",
    "layernorm": "original gate followed by LayerNorm on the fused vector (removes the global shrinkage)",
}


class SGMBF(nn.Module):
    """Shared-gate multi-branch fusion. gated=False gives the 'ungated sum' ablation."""

    def __init__(self, dims, d_h=256, hidden=(512, 256), dropout=0.3, gated=True, gate_variant="sigmoid"):
        super().__init__()
        if gate_variant not in GATE_VARIANTS:
            raise ValueError(f"unknown gate_variant {gate_variant!r}; choose among {list(GATE_VARIANTS)}")
        self.mods = list(dims)
        self.proj = nn.ModuleDict({m: projection(d, d_h) for m, d in dims.items()})
        self.gated, self.variant = gated, gate_variant
        self.gate = nn.Linear(2 * d_h, 1)  # shared w_g in R^{2 d_h}, b_g in R  -> 513 parameters
        if gate_variant == "sigmoid_open":
            nn.init.constant_(self.gate.bias, 2.0)
        self.norm = nn.LayerNorm(d_h) if gate_variant == "layernorm" else None
        self.head = Head(d_h, hidden, dropout)

    def forward(self, x, modality_mask=None):
        h = {m: self.proj[m](x[m]) for m in self.mods}
        H = torch.stack([h[m] for m in self.mods], 1)               # (B, M, d_h)
        c = H.mean(1, keepdim=True).expand_as(H)                     # context vector
        if self.gated:
            g = torch.sigmoid(self.gate(torch.cat([H, c], -1))).squeeze(-1)   # (B, M)
            if self.variant == "scaled":
                g = 2 * g
        else:
            g = torch.ones(H.shape[:2], device=H.device, dtype=H.dtype)
        if modality_mask is not None:                                # (B, M) 0/1 - test-time removal
            g = g * modality_mask
        w = g
        if self.gated and self.variant == "normalized":
            k = (g > 0).sum(1, keepdim=True).clamp(min=1).to(g.dtype)  # active modalities
            w = k * g / g.sum(1, keepdim=True).clamp(min=1e-6)
        z = (w.unsqueeze(-1) * H).sum(1)
        if self.norm is not None:
            z = self.norm(z)
        return self.head(z), {"gates": g.detach(), "norms": H.detach().norm(dim=-1)}


class EarlyConcat(nn.Module):
    def __init__(self, dims, d_h=256, hidden=(512, 256), dropout=0.3):
        super().__init__()
        self.mods = list(dims)
        self.proj = projection(sum(dims.values()), d_h)
        self.head = Head(d_h, hidden, dropout)

    def forward(self, x, modality_mask=None):
        return self.head(self.proj(torch.cat([x[m] for m in self.mods], 1))), {}


class LateAverage(nn.Module):
    def __init__(self, dims, d_h=256, hidden=(512, 256), dropout=0.3):
        super().__init__()
        self.mods = list(dims)
        self.branch = nn.ModuleDict({m: nn.Sequential(projection(d, d_h), Head(d_h, hidden, dropout))
                                     for m, d in dims.items()})

    def forward(self, x, modality_mask=None):
        return torch.stack([self.branch[m](x[m]) for m in self.mods], 0).mean(0), {}


class SoftmaxAttention(nn.Module):
    def __init__(self, dims, d_h=256, hidden=(512, 256), dropout=0.3):
        super().__init__()
        self.mods = list(dims)
        self.proj = nn.ModuleDict({m: projection(d, d_h) for m, d in dims.items()})
        self.score = nn.Sequential(nn.Linear(d_h, d_h // 2), nn.Tanh(), nn.Linear(d_h // 2, 1))
        self.head = Head(d_h, hidden, dropout)

    def forward(self, x, modality_mask=None):
        H = torch.stack([self.proj[m](x[m]) for m in self.mods], 1)
        a = torch.softmax(self.score(H).squeeze(-1), 1)
        return self.head((a.unsqueeze(-1) * H).sum(1)), {"gates": a.detach()}


class GMU(nn.Module):
    """Gated multimodal unit (Arevalo et al., 2017), generalised to M modalities with vector gates."""

    def __init__(self, dims, d_h=256, hidden=(512, 256), dropout=0.3):
        super().__init__()
        self.mods = list(dims)
        self.hid = nn.ModuleDict({m: nn.Linear(d, d_h) for m, d in dims.items()})
        self.gates = nn.Linear(sum(dims.values()), d_h * len(dims))
        self.bn = nn.BatchNorm1d(d_h)
        self.head = Head(d_h, hidden, dropout)
        self.d_h = d_h

    def forward(self, x, modality_mask=None):
        H = torch.stack([torch.tanh(self.hid[m](x[m])) for m in self.mods], 1)
        Z = torch.sigmoid(self.gates(torch.cat([x[m] for m in self.mods], 1))).view(-1, len(self.mods), self.d_h)
        return self.head(self.bn((Z * H).sum(1))), {}


class TokenTransformer(nn.Module):
    """Transformer encoder over the M projected modality tokens (cross-modal attention)."""

    def __init__(self, dims, d_h=256, hidden=(512, 256), dropout=0.3, layers=2, heads=4):
        super().__init__()
        self.mods = list(dims)
        self.proj = nn.ModuleDict({m: projection(d, d_h) for m, d in dims.items()})
        self.mod_emb = nn.Parameter(torch.zeros(len(dims), d_h))
        enc = nn.TransformerEncoderLayer(d_h, heads, dim_feedforward=2 * d_h, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = Head(d_h, hidden, dropout)

    def forward(self, x, modality_mask=None):
        H = torch.stack([self.proj[m](x[m]) for m in self.mods], 1) + self.mod_emb
        return self.head(self.enc(H).mean(1)), {}


FUSIONS = {
    "SG-MBF": lambda dims, **k: SGMBF(dims, gated=True, **k),
    "Ungated sum": lambda dims, **k: SGMBF(dims, gated=False, **k),
    # control for the 'layernorm' variant: same LayerNorm, no gate -> isolates what the gate itself adds
    "Ungated sum + LN": lambda dims, **k: SGMBF(dims, gated=False, gate_variant="layernorm", **k),
    "Early concat": EarlyConcat,
    "Late average": LateAverage,
    "Softmax attention": SoftmaxAttention,
    "GMU": GMU,
    "Token transformer": TokenTransformer,
}


def build(name, dims, cfg, gate_variant=None):
    t = cfg["training"]
    kw = dict(d_h=t["d_h"], hidden=tuple(t["head_hidden"]), dropout=t["dropout"])
    if name == "SG-MBF":
        kw["gate_variant"] = gate_variant or t.get("gate_variant", "sigmoid")
    return FUSIONS[name](dims, **kw)


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
