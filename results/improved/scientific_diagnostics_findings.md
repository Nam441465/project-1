# Scientific Diagnostics Report (Random Forest-Only Analytical & Full Held-Out Evaluation)

All metrics below were computed strictly from existing `.npy`, `.joblib`, `.json`, `.csv` artifacts and `data/processed/heldout_test.parquet` without fitting or modifying any model.

## 1. Probability Calibration Diagnostics (`PaperRF` vs `Ours Improved RF`)

| Attack | Model | Brier | LogLoss | ECE-15 | MCE-15 | `t*_F1` | Test F1 | `t*_c10` | Cost `c=10` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DoS | PaperRF | 0.000590 | 0.002447 | 0.000241 | 0.173891 | 0.420 | 0.8810 | 0.155 | 729 |
| DoS | RF | 0.000989 | 0.003666 | 0.001611 | 0.439146 | 0.760 | 0.8650 | 0.485 | 767 |
| Exploits | PaperRF | 0.001443 | 0.005596 | 0.000626 | 0.138126 | 0.460 | 0.9470 | 0.125 | 1,443 |
| Exploits | RF | 0.001443 | 0.005596 | 0.000626 | 0.138126 | 0.460 | 0.9470 | 0.125 | 1,443 |
| Backdoor | PaperRF | 0.000095 | 0.000434 | 0.000048 | 0.441776 | 0.280 | 0.9385 | 0.155 | 225 |
| Backdoor | RF | 0.000084 | 0.000505 | 0.000046 | 0.320556 | 0.300 | 0.9535 | 0.145 | 174 |
| Analysis | PaperRF | 0.000254 | 0.001105 | 0.000131 | 0.561502 | 0.670 | 0.8786 | 0.090 | 698 |
| Analysis | RF | 0.000266 | 0.001522 | 0.000103 | 0.710000 | 0.550 | 0.8733 | 0.035 | 855 |
| Reconnaissance | PaperRF | 0.000491 | 0.002336 | 0.000261 | 0.222035 | 0.480 | 0.9674 | 0.235 | 337 |
| Reconnaissance | RF | 0.000492 | 0.002349 | 0.000276 | 0.216408 | 0.440 | 0.9673 | 0.225 | 339 |

## 2. Stratified Bootstrap 95% CIs (B=1000) & Exact Two-Sided McNemar Tests

| Attack | Operating Point | Pair | Model A F1 (95% CI) | Model B F1 (95% CI) | Disc (A+/B-, A-/B+) | Exact `p`-value |
|---|---|---|---|---|---|---:|
| DoS | F1-Optimal | RF vs PaperRF | 0.8650 [0.8507, 0.8786] | 0.8810 [0.8677, 0.8952] | (54, 95) | 9.833e-04 |
| DoS | Cost-Opt c=10 | RF vs PaperRF | 0.8262 [0.8131, 0.8391] | 0.8301 [0.8170, 0.8432] | (63, 74) | 3.930e-01 |
| Exploits | F1-Optimal | RF vs PaperRF | 0.9470 [0.9427, 0.9510] | 0.9470 [0.9428, 0.9509] | (0, 0) | 1.000e+00 |
| Exploits | Cost-Opt c=10 | RF vs PaperRF | 0.9197 [0.9151, 0.9243] | 0.9197 [0.9150, 0.9244] | (0, 0) | 1.000e+00 |
| Backdoor | F1-Optimal | RF vs PaperRF | 0.9535 [0.9380, 0.9674] | 0.9385 [0.9211, 0.9556] | (15, 3) | 7.538e-03 |
| Backdoor | Cost-Opt c=10 | RF vs PaperRF | 0.9212 [0.9031, 0.9395] | 0.9041 [0.8853, 0.9236] | (22, 7) | 8.130e-03 |
| Analysis | F1-Optimal | RF vs PaperRF | 0.8733 [0.8484, 0.8968] | 0.8786 [0.8539, 0.9028] | (1, 5) | 2.188e-01 |
| Analysis | Cost-Opt c=10 | RF vs PaperRF | 0.5424 [0.5241, 0.5616] | 0.6962 [0.6717, 0.7197] | (52, 425) | 7.747e-74 |
| Reconnaissance | F1-Optimal | RF vs PaperRF | 0.9673 [0.9626, 0.9722] | 0.9674 [0.9628, 0.9717] | (5, 6) | 1.000e+00 |
| Reconnaissance | Cost-Opt c=10 | RF vs PaperRF | 0.9640 [0.9594, 0.9688] | 0.9644 [0.9596, 0.9690] | (0, 2) | 5.000e-01 |

## 3. Mixed-Flow 5-Specialist Orchestration (All 363,035 Held-Out Flows)

| Model | Operating Point | Benign FPR (%) | Target-5 Recall (%) | Target-5 F1 | Unseen-4 Flagged (%) | All-9 Recall (%) | All-9 F1 | Inference (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| RF | F1-Optimal | 0.16% | 96.90% | 0.9575 | 29.54% | 72.20% | 0.8219 | 11.50s |
| RF | Cost-Opt c=10 | 0.40% | 99.77% | 0.9349 | 36.07% | 76.41% | 0.8258 | 11.50s |
| PaperRF | F1-Optimal | 0.16% | 96.69% | 0.9574 | 28.88% | 71.83% | 0.8200 | 7.98s |
| PaperRF | Cost-Opt c=10 | 0.35% | 99.43% | 0.9412 | 36.21% | 76.25% | 0.8299 | 7.98s |

### 3.1 Per-Class Unseen-4 & Target-5 Detection (`Cost-Opt c=10`)

| Traffic Class | Flows | PaperRF (%) | Ours RF (%) |
|---|---:|---:|---:|
| Benign | 346,841 | 0.35% | 0.40% |
| DoS | 1,130 | 99.47% | 99.56% |
| Exploits | 5,533 | 99.39% | 99.77% |
| Backdoor | 398 | 99.50% | 100.00% |
| Analysis | 446 | 96.64% | 98.65% |
| Reconnaissance | 2,749 | 99.96% | 100.00% |
| Fuzzers | 4,353 | 17.14% | 17.05% |
| Generic | 1,246 | 90.69% | 90.93% |
| Shellcode | 305 | 78.69% | 76.39% |
| Worms | 34 | 100.00% | 100.00% |
| TARGET_5_TOTAL | 10,256 | 99.43% | 99.77% |
| UNSEEN_4_TOTAL | 5,938 | 36.21% | 36.07% |
| ALL_ATTACKS_TOTAL | 16,194 | 76.25% | 76.41% |
