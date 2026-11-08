# Adaptive Power Units: Learning Bounded, Input-Dependent Activation Functions for Image Classification

An Adaptive Power Unit (APU) is an activation function that raises each input to a power that depends on the input itself:

$$
f(z) = z\,(z^2 + \epsilon)^{(p(z)-1)/2}, \qquad p(z) = p_0 + A \tanh(wz + b), \qquad A = a_{\max}\tanh(a)
$$

The exponent stays in $(p_0 - a_{\max},\ p_0 + a_{\max})$, which is $(0.5, 1.5)$ with the defaults used here. Each activation layer learns three scalars ($a$, $w$, $b$), so a three-activation CNN gains 9 parameters.

This repository holds the code, the complete run record and the analysis for a controlled study of APU. The study asks whether learning this exponent improves a shallow image classifier compared with the same function frozen at its starting shape and with tuned standard and learnable activations, under equal tuning budgets, and at what cost.

## Main results

Final-epoch test top-1 accuracy (%), mean ± sample SD over 5 seeds, original SimpleCNN, 200 epochs. All 189 runs completed; none diverged.

| Method | CIFAR-10, equal search (primary) | CIFAR-100 (predeclared confirmation) | CIFAR-10, common recipe (exploratory) |
|---|---|---|---|
| ReLU | 81.79 ± 0.39 | 51.38 ± 0.36 | 81.52 ± 0.30 |
| GELU | 82.67 ± 0.20 | 52.34 ± 0.27 | 81.82 ± 0.53 |
| PReLU | 82.08 ± 0.31 | 52.10 ± 0.69 | 81.71 ± 0.12 |
| ACON-C | 82.55 ± 0.36 | **54.18 ± 0.37** | 81.64 ± 0.12 |
| Learned constant power | 79.97 ± 0.20 | 52.05 ± 0.29 | 79.85 ± 0.16 |
| APU, frozen at initial shape | 79.28 ± 0.38 | 49.35 ± 0.28 | 79.08 ± 0.26 |
| **APU (full, learned)** | **82.88 ± 0.32** | 53.92 ± 0.31 | **82.94 ± 0.35** |

Paired differences, full APU minus each control (percentage points, 95% confidence interval):

| Comparison | CIFAR-10 | CIFAR-100 |
|---|---|---|
| Frozen APU (primary comparison) | +3.60 [3.35, 3.86] | +4.57 [4.26, 4.88] |
| Strongest standard activation (GELU) | +0.21 [−0.40, 0.81] | +1.58 [1.01, 2.15] |
| Strongest adaptive activation (ACON-C) | +0.33 [−0.08, 0.74] | −0.26 [−1.04, 0.52] |
| Learned constant power | +2.91 [2.39, 3.44] | +1.86 [1.24, 2.49] |

What the results show:

- **Learning the exponent matters.** The same function with frozen parameters is the weakest method on both datasets, 3.6 and 4.6 points behind the learned version.
- **APU is competitive, not superior.** It ties GELU and ACON-C on CIFAR-10, beats GELU on CIFAR-100, and ties ACON-C on both. Under the decision rules fixed before the study, the outcome is "no reproducible advantage over strong controls".
- **The earlier advantage came mostly from tuning.** With one shared recipe for every method, APU beats GELU by 1.12 points. With equal search, GELU improves by 0.85 points and the gap is no longer significant.
- **Cost.** Compared with ReLU, an APU training epoch takes 102% longer in eager PyTorch and 18% longer with `torch.compile`. The study trained in eager mode.

The evidence covers one shallow CNN on CIFAR-10 and CIFAR-100. An earlier exploratory study, kept in `historical/`, found APU behind ReLU on ResNet-18 and ViT-Tiny.

## Repository layout

```
Adaptive-Power-Units/
├── APU_v2.ipynb          the study notebook (settings, launch, analysis); outputs included
├── apu_lib/              library code, written by the notebook (12 modules)
├── apu_v2_runs/          the complete run record, unchanged since the run
│   ├── configs/          one immutable configuration per run, named by its digest
│   ├── runs/             run records (status, history, diagnostics) and one-time test results
│   ├── protocol/         protocol.json (frozen before the search) and lock.json (before final fits)
│   ├── splits/           validation split indices with checksums
│   ├── engineering/      results of the 31 pre-training checks
│   ├── benchmarks/       cost measurements, one file per process
│   ├── audit/            audit of the historical study
│   ├── logs/             one environment record per session
│   └── paper_assets/     tables, figures, CLAIMS.md, CHECKLIST.md
├── experiments/          one CSV per stage: search, primary, confirmation, common_recipe
├── results/              copy of paper_assets for browsing
├── historical/           the earlier exploratory study (APU_Final2) and its records
├── scripts/              build_indexes.py regenerates experiments/ and results/
└── docs/                 methodology.md, reproducibility.md
```

Runs are stored by configuration digest, not by stage, because a run can belong to more than one stage. `experiments/*.csv` gives the stage view.

## Setup

Tested on Windows 11 with an NVIDIA GeForce RTX 5070 (12 GB), Python 3.12.10, PyTorch 2.11.0 and CUDA 12.8.

```
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Running the study

1. Open `APU_v2.ipynb` in Jupyter, set `PROFILE = "smoke"` in the first code cell and run all cells. This tests the whole pipeline on synthetic data in about 10 minutes and writes only to `apu_v2_smoke/`.
2. Set `PROFILE = "paper"` and run all cells again. CIFAR-10 and CIFAR-100 are downloaded to `data/` and checked before any training.

The full run took about 35 hours on the RTX 5070. If it stops, running all cells again resumes from the last saved epoch, and completed runs are reused. With the existing `apu_v2_runs/` in place, the notebook reuses every run and only rebuilds the analysis.

The historical audit expects the old study in the folders named in the settings cell. In this repository they are under `historical/`, so set `OLD_ROOT = Path("historical/apu_runs_paper")` and `OLD_NOTEBOOK = Path("historical/APU_Final2.ipynb")` before rerunning.

To refresh the stage CSVs and the `results/` copy:

```
python scripts/build_indexes.py
```

## Checkpoints

The final checkpoints (about 1.2 GB) are not in the repository. Download `checkpoints.zip` from the Releases page and unzip it into `apu_v2_runs/checkpoints/`. They are only needed to re-evaluate models; the test results are already recorded in `apu_v2_runs/runs/*.test.json`.

## Documentation

- [docs/methodology.md](docs/methodology.md): the activation, controls, model, training, search, locking, statistics and decision rules.
- [docs/reproducibility.md](docs/reproducibility.md): environment, determinism, run identities, the order of protocol steps, and how to verify the record.

## License

See [LICENSE](LICENSE).
