# Gate variant selection - VALIDATION set only (the test set is not used)

Selection rule (fixed in advance): lowest mean validation MSE over seeds (fixed before looking at the results).

| Variant | Val MSE | Val Acc (%) | Val CCC | mean gates (t, a, v) |
|---|---|---|---|---|
| **layernorm** | 0.01184 ± 0.00009 | 91.32 ± 0.04 | 0.626 ± 0.005 | 0.36, 0.35, 0.36 |
| sigmoid_open | 0.01193 ± 0.00007 | 91.29 ± 0.03 | 0.617 ± 0.008 | 0.73, 0.72, 0.72 |
| normalized | 0.01214 ± 0.00008 | 91.22 ± 0.05 | 0.617 ± 0.005 | 0.17, 0.15, 0.16 |
| scaled | 0.01224 ± 0.00010 | 91.19 ± 0.03 | 0.603 ± 0.005 | 0.33, 0.30, 0.34 |
| sigmoid | 0.01226 ± 0.00009 | 91.18 ± 0.04 | 0.599 ± 0.011 | 0.18, 0.17, 0.19 |
| *Ungated sum (reference)* | 0.01200 ± 0.00007 | 91.27 ± 0.04 | 0.614 ± 0.009 | 1, 1, 1 |

Selected variant: **layernorm** - better than the ungated sum on validation (MSE 0.01184 vs 0.01200; Δacc +0.05 pts, 95% CI [+0.00, +0.10], p = 0.0314).

Next step: '10 · Évaluation finale (test)' evaluates ONLY the selected variant on the test set, once.

Variants:
- `sigmoid`: g = sigmoid(w.[h;c] + b), b initialised at 0 (original SG-MBF, g starts at 0.5)
- `sigmoid_open`: same gate, b initialised at +2 (g starts at 0.88: starts close to the ungated sum)
- `scaled`: g = 2 sigmoid(.) in (0, 2), starts at 1: a modality can be amplified as well as attenuated
- `normalized`: w_m = M g_m / sum_k g_k: the gate only redistributes weight, the fused scale is preserved
- `layernorm`: original gate followed by LayerNorm on the fused vector (removes the global shrinkage)