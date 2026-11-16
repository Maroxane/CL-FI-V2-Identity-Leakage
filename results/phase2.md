# Phase 2: secondary analyses with the models of the leakage audit

S0R = random clip-level splits (leaky); S2A = identity-disjoint splits (leak-free). Values are means over split draws; SD across draws in brackets (each draw first averaged over seeds).

## Seen vs unseen videos inside the leaky splits (S0R)

| Model | CCC seen video | CCC unseen video | Acc seen | Acc unseen | n unseen per draw |
|---|---|---|---|---|---|
| Unimodal T | 0.184 (0.016) | 0.179 (0.020) | 88.34 (0.14) | 88.21 (0.27) | 309 |
| Unimodal A | 0.537 (0.007) | 0.449 (0.039) | 90.52 (0.08) | 89.77 (0.45) | 309 |
| Unimodal V | 0.612 (0.011) | 0.545 (0.035) | 91.05 (0.09) | 90.39 (0.30) | 309 |
| Bimodal T+A | 0.452 (0.013) | 0.447 (0.026) | 89.93 (0.12) | 89.85 (0.34) | 309 |
| Bimodal T+V | 0.584 (0.013) | 0.561 (0.031) | 90.79 (0.09) | 90.53 (0.25) | 309 |
| Bimodal A+V | 0.662 (0.011) | 0.616 (0.037) | 91.61 (0.08) | 91.11 (0.40) | 309 |
| Early concat | 0.618 (0.011) | 0.606 (0.038) | 91.16 (0.09) | 90.98 (0.35) | 309 |
| Late average | 0.630 (0.010) | 0.602 (0.037) | 91.26 (0.09) | 90.90 (0.31) | 309 |
| Softmax attention | 0.628 (0.011) | 0.602 (0.038) | 91.23 (0.11) | 90.92 (0.34) | 309 |
| GMU | 0.640 (0.010) | 0.614 (0.035) | 91.37 (0.09) | 91.09 (0.32) | 309 |
| Token transformer | 0.625 (0.011) | 0.602 (0.038) | 91.16 (0.10) | 90.89 (0.36) | 309 |
| Ungated sum | 0.634 (0.011) | 0.609 (0.038) | 91.31 (0.08) | 90.97 (0.36) | 309 |
| Ungated sum + LN | 0.633 (0.011) | 0.604 (0.036) | 91.32 (0.08) | 90.97 (0.35) | 309 |
| SG-MBF | 0.631 (0.011) | 0.605 (0.037) | 91.28 (0.09) | 90.96 (0.35) | 309 |

## Robustness on identity-disjoint splits (S2A): SG-MBF vs the same model without gate

| Perturbation | Level | draws | SG-MBF CCC | Ungated + LN CCC | Δ CCC (gate) | SG-MBF Acc | Ungated + LN Acc | gates t / a / v |
|---|---|---|---|---|---|---|---|---|
| clean | 0.0 | 10 | 0.565 (0.015) | 0.566 (0.016) | -0.000 (0.005) | 90.71 (0.13) | 90.71 (0.14) | 0.47 / 0.47 / 0.47 |
| remove_t | 1.0 | 10 | 0.600 (0.016) | 0.599 (0.016) | 0.001 (0.005) | 90.73 (0.12) | 90.74 (0.11) | 0.00 / 0.47 / 0.47 |
| remove_a | 1.0 | 10 | 0.460 (0.024) | 0.462 (0.023) | -0.002 (0.007) | 89.89 (0.18) | 89.90 (0.20) | 0.47 / 0.00 / 0.47 |
| remove_v | 1.0 | 10 | 0.348 (0.014) | 0.348 (0.015) | 0.000 (0.005) | 89.28 (0.14) | 89.27 (0.16) | 0.47 / 0.47 / 0.00 |
| frame_drop | 0.25 | 3 | 0.547 (0.025) | 0.550 (0.026) | -0.002 (0.008) | 90.65 (0.18) | 90.68 (0.19) | 0.48 / 0.47 / 0.48 |
| frame_drop | 0.5 | 3 | 0.535 (0.027) | 0.537 (0.029) | -0.002 (0.007) | 90.50 (0.20) | 90.53 (0.19) | 0.48 / 0.47 / 0.48 |
| frame_drop | 0.75 | 3 | 0.492 (0.025) | 0.494 (0.027) | -0.002 (0.015) | 89.93 (0.22) | 89.96 (0.18) | 0.48 / 0.47 / 0.48 |
| clean (official-test subset) | 0.0 | 10 | 0.566 (0.013) | 0.566 (0.017) | 0.001 (0.006) | 90.69 (0.19) | 90.69 (0.22) | 0.47 / 0.47 / 0.47 |
| audio_snr_db | 40.0 | 10 | 0.565 (0.015) | 0.565 (0.018) | -0.000 (0.005) | 90.67 (0.21) | 90.66 (0.23) | 0.47 / 0.47 / 0.47 |
| audio_snr_db | 30.0 | 10 | 0.552 (0.016) | 0.553 (0.019) | -0.001 (0.006) | 90.51 (0.24) | 90.53 (0.26) | 0.47 / 0.46 / 0.47 |
| audio_snr_db | 20.0 | 10 | 0.532 (0.018) | 0.533 (0.020) | -0.000 (0.007) | 90.33 (0.28) | 90.36 (0.29) | 0.48 / 0.45 / 0.48 |
| audio_snr_db | 10.0 | 10 | 0.500 (0.018) | 0.498 (0.021) | 0.002 (0.009) | 90.14 (0.32) | 90.16 (0.29) | 0.48 / 0.43 / 0.48 |
| audio_snr_db | 0.0 | 10 | 0.450 (0.022) | 0.430 (0.028) | 0.020 (0.016) | 89.84 (0.32) | 89.73 (0.33) | 0.47 / 0.39 / 0.47 |

## Fairness: mean absolute error x100 per group

| Model | Attribute | Group | n per draw | S0R (leaky) | S2A (leak-free) |
|---|---|---|---|---|---|
| SG-MBF | gender | female | 226 | 8.49 (0.22) | 9.00 (0.31) |
| SG-MBF | gender | male | 177 | 8.49 (0.24) | 8.96 (0.32) |
| SG-MBF | ethnicity | African-American | 45 | 9.08 (0.65) | 9.58 (0.62) |
| SG-MBF | ethnicity | Asian | 10 | 8.08 (1.03) | 8.31 (1.32) |
| SG-MBF | ethnicity | Caucasian | 348 | 8.43 (0.17) | 8.92 (0.23) |
| GMU | gender | female | 226 | 8.42 (0.25) | 8.97 (0.30) |
| GMU | gender | male | 177 | 8.46 (0.22) | 8.94 (0.38) |
| GMU | ethnicity | African-American | 45 | 9.01 (0.68) | 9.49 (0.59) |
| GMU | ethnicity | Asian | 10 | 8.28 (1.34) | 7.94 (1.41) |
| GMU | ethnicity | Caucasian | 348 | 8.37 (0.16) | 8.91 (0.26) |

## Identity in the learned representation (top-1 same-video retrieval among test clips, %)

| Model | Representation | S0R (leaky) | S2A (leak-free) | chance S0R / S2A | queries S0R / S2A |
|---|---|---|---|---|---|
| Unimodal A | input features | 77.5 (0.6) | 87.0 (0.4) | 0.07 / 0.18 | 974 / 1853 |
| Unimodal A | last hidden layer | 68.2 (1.3) | 84.4 (0.6) | 0.07 / 0.18 | 974 / 1853 |
| Unimodal V | input features | 75.9 (0.8) | 88.1 (0.2) | 0.07 / 0.18 | 974 / 1853 |
| Unimodal V | last hidden layer | 49.6 (1.4) | 59.8 (2.0) | 0.07 / 0.18 | 974 / 1853 |
| Bimodal A+V | input features | 81.9 (1.0) | 91.3 (0.1) | 0.07 / 0.18 | 974 / 1853 |
| Bimodal A+V | last hidden layer | 69.7 (0.6) | 82.9 (0.7) | 0.07 / 0.18 | 974 / 1853 |
| GMU | input features | 67.0 (1.9) | 81.3 (0.9) | 0.07 / 0.18 | 974 / 1853 |
| GMU | last hidden layer | 34.9 (1.1) | 54.6 (1.1) | 0.07 / 0.18 | 974 / 1853 |
| SG-MBF | input features | 67.0 (1.9) | 81.3 (0.9) | 0.07 / 0.18 | 974 / 1853 |
| SG-MBF | last hidden layer | 40.2 (1.2) | 59.9 (2.1) | 0.07 / 0.18 | 974 / 1853 |

The two families have different query sets (a random split scatters the clips of a video), so compare each learned representation with the input features of the same column, not the columns with each other.

## Zero-shot transfer to MuPTA (speaker level, mean over traits)

| Model | CCC, trained on S0R | CCC, trained on S2A | r, trained on S0R | r, trained on S2A |
|---|---|---|---|---|
| SG-MBF | 0.011 (0.011) | 0.019 (0.013) | -0.003 (0.044) | 0.088 (0.076) |
| GMU | 0.022 (0.004) | 0.023 (0.011) | 0.071 (0.003) | 0.102 (0.045) |
| Ungated sum + LN | 0.018 (0.008) | 0.018 (0.012) | 0.031 (0.028) | 0.081 (0.063) |