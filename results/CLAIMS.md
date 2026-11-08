# Claims and evidence (generated 2026-11-02T04:05:09 by apu_lib/report.py sha256:913d11fc625d)

## 1. Historical exploratory evidence (APU_Final2, already inspected on the test set; not used for selection)
- APU had the highest mean in 1 of 13 historical settings.
- Count reconciliation and corrections: see audit/count_reconciliation.csv and audit/corrections.csv (1 corrections).
- Generative results: Frechet distance in the feature space of a small classifier trained on the real data (NOT Inception-based FID); generated samples per score: 4864.
- Corruption results: 8 re-implemented corruption types; not the official CIFAR-10-C / CIFAR-100-C benchmark.

## 2. New evidence (protocol frozen before search; selection on validation only; test evaluated once after the lock)
- Protocol a3f970bfd0ecc8a9, lock d14302b877d38c45.
- Stage 1, primary (equal-search arm): ReLU 81.79 +/- 0.39 (n = 5/5); GELU 82.67 +/- 0.20 (n = 5/5); PReLU 82.08 +/- 0.31 (n = 5/5); APU frozen profile 79.28 +/- 0.38 (n = 5/5); Learned constant power 79.97 +/- 0.20 (n = 5/5); APU (full, learned) 82.88 +/- 0.32 (n = 5/5); ACON-C 82.55 +/- 0.36 (n = 5/5)
  - primary: apu - apu_frozen = +3.60 points, 95% CI [3.35, 3.86], paired t, p = 2.63e-06
  - secondary: strongest validated standard: apu - gelu = +0.21 points, 95% CI [-0.40, 0.81], paired t, p = 0.398, Holm p = 0.398
  - secondary: strongest validated adaptive: apu - acon_c = +0.33 points, 95% CI [-0.08, 0.74], paired t, p = 0.0882, Holm p = 0.176
  - secondary: learned constant power: apu - apu_const = +2.91 points, 95% CI [2.39, 3.44], paired t, p = 0.000104, Holm p = 0.000312
- Stage 2, predeclared confirmation: ReLU 51.38 +/- 0.36 (n = 5/5); GELU 52.34 +/- 0.27 (n = 5/5); PReLU 52.10 +/- 0.69 (n = 5/5); APU frozen profile 49.35 +/- 0.28 (n = 5/5); Learned constant power 52.05 +/- 0.29 (n = 5/5); APU (full, learned) 53.92 +/- 0.31 (n = 5/5); ACON-C 54.18 +/- 0.37 (n = 5/5)
  - primary: apu - apu_frozen = +4.57 points, 95% CI [4.26, 4.88], paired t, p = 2.06e-06
  - secondary: strongest validated standard: apu - gelu = +1.58 points, 95% CI [1.01, 2.15], paired t, p = 0.00154, Holm p = 0.00347
  - secondary: strongest validated adaptive: apu - acon_c = -0.26 points, 95% CI [-1.04, 0.52], paired t, p = 0.401, Holm p = 0.401
  - secondary: learned constant power: apu - apu_const = +1.86 points, 95% CI [1.24, 2.49], paired t, p = 0.00116, Holm p = 0.00347
- Stage 1, fixed historical recipe (exploratory, answers a different question): ReLU 81.52 +/- 0.30 (n = 5/5); GELU 81.82 +/- 0.53 (n = 5/5); PReLU 81.71 +/- 0.12 (n = 5/5); APU frozen profile 79.08 +/- 0.26 (n = 5/5); Learned constant power 79.85 +/- 0.16 (n = 5/5); APU (full, learned) 82.94 +/- 0.35 (n = 5/5); ACON-C 81.64 +/- 0.11 (n = 5/5)
  - primary: apu - apu_frozen = +3.86 points, 95% CI [3.16, 4.56], paired t, p = 0.000107
  - secondary: strongest validated standard: apu - gelu = +1.12 points, 95% CI [0.60, 1.65], paired t, p = 0.004, Holm p = 0.004
  - secondary: strongest validated adaptive: apu - acon_c = +1.30 points, 95% CI [0.92, 1.68], paired t, p = 0.00066, Holm p = 0.00132
  - secondary: learned constant power: apu - apu_const = +3.09 points, 95% CI [2.48, 3.70], paired t, p = 0.000148, Holm p = 0.000445

## 3. Interpretation under the predeclared decision rules
- [E] No reproducible advantage over strong controls -> revise the central claim or pursue a carefully supported behavioural analysis.
- Measured APU training-epoch overhead vs ReLU: 101.92982175142214 % (percent change in cost, not accuracy points).

## 4. Proposed or not-yet-run work (no results claimed)
- Depth study and Pade baseline: not run (conditional on decision A).
- Official CIFAR-10-C / CIFAR-100-C, ImageNet-scale runs and generative experiments are not part of this study.
- ACON-C and Pade are re-implementations; check them against the primary sources before submission.