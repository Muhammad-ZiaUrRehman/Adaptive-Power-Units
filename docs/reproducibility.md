# Reproducibility

## Environment of the recorded run

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5070, 12 GB, driver 610.88 |
| OS | Windows 11 |
| Python | 3.12.10 |
| PyTorch / torchvision | 2.11.0 / 0.26.0 (CUDA 12.8 build) |
| cuDNN | 9.19 |
| Precision | bfloat16 autocast; the APU power is computed in float32 |
| Determinism | `torch.use_deterministic_algorithms(True, warn_only=True)`, cuDNN benchmark off, `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| Training mode | eager (`TRAIN_COMPILE = False`) |
| Run period | 31 October 2026, 17:20 to 2 November 2026, 04:05 |

Each session's full environment record is in `apu_v2_runs/logs/session_*.json`.

## Run identity

Every run is defined by one configuration file in `apu_v2_runs/configs/`. The file covers the dataset version and checksums, split indices and checksums, model, every activation argument, optimiser and parameter groups, schedule, augmentation, precision, compile and determinism settings, seed, budget, and a SHA-256 hash of the six training modules (`config`, `activations`, `models`, `data`, `train`, `evaluate`).

The run's fingerprint is the SHA-256 of that configuration, written in a fixed canonical form. Records, checkpoints and test results are all named by its first 16 characters.

Consequences:
- Changing any setting or any line of training code produces new fingerprints, so new runs never overwrite or mix with old ones.
- A stored result is reused only if its configuration matches exactly; a mismatch raises an error.
- A run counts as completed only when its final checkpoint reloads, its checksum matches the record, and its final metrics are finite.

## Order of steps

| Step | File | Recorded digest |
|---|---|---|
| Protocol frozen before any search run | `apu_v2_runs/protocol/protocol.json` | `a3f970bfd0ecc8a9…` |
| Lock written after the search, before any final run | `apu_v2_runs/protocol/lock.json` | `d14302b877d38c45…` |
| Test evaluation after every locked run finished | `apu_v2_runs/runs/*.test.json` | timestamp later than the lock in each file |

The training code never loads the test partition. It also checks every batch for validation images and would fail the run if one appeared.

## Checks before training

The notebook ran 31 checks before any long training and would not continue if one failed. Results are in `apu_v2_runs/engineering/checks.json`. They include:

- the APU forward pass matches a separate float64 reference implementation;
- float64 finite-difference gradient checks, including near-zero inputs and parameters near their limits;
- identity at $A = 0$, sign preservation, and finite outputs and gradients from −10⁴ to 10⁴;
- added parameter counts (9 for SimpleCNN, 51 for ResNet-18, 36 for ViT-Tiny), and full-model counts matching the historical study;
- the frozen APU starting from exactly the full APU's function, with its parameters never updated;
- identical backbone weights and data order across methods for a seed;
- a run interrupted at an epoch boundary and resumed reproducing the uninterrupted run bit for bit;
- diagnostics switched on or off giving the same trajectory;
- probes and evaluation leaving BatchNorm statistics unchanged.

## Reproducing the results

**From the record (minutes).** Open `APU_v2.ipynb` and run all cells with `PROFILE = "paper"`. Every run is found in `apu_v2_runs/` and reused, and the notebook rebuilds every table, figure, the decision and the checklist from the records.

**Retraining from scratch (about 35 hours on an RTX 5070).** Move `apu_v2_runs/` aside and run all cells. With deterministic kernels on the same software stack, results should match closely. Exact bit-for-bit agreement across different GPUs, drivers or library versions is not expected.

**Re-evaluating models.** Download the checkpoints (see the README) into `apu_v2_runs/checkpoints/`. Each record's `evaluation_checkpoint` field gives the file path and its SHA-256.

**Stage views.** `python scripts/build_indexes.py` regenerates `experiments/*.csv` and `results/` from the record without changing it.

## Historical study

`historical/` holds the earlier exploratory study (`APU_Final2.ipynb` and its records, without checkpoints). Its test results had been inspected before the controlled study, so they are not used for any selection or claim.

The audit in `apu_v2_runs/audit/` rebuilt its counts from the raw files: 483 classification jobs, 464 completed, 19 diverged, 404 completed test-evaluated runs. It also lists one correction: the old main-results table mixed ablation variants into the CIFAR-100 ResNet-18 rows.

## Known limitations

- Deterministic mode was set to warn rather than fail, so an unsupported kernel would log a warning instead of stopping. The resume check passed on short runs; every full run being bit-for-bit deterministic is not guaranteed.
- Five seeds measure run-to-run variation on fixed splits only.
- One validation split, six candidates and two selection seeds limit how precisely the best settings were found.
- Peak memory figures include warm-up and compilation phases.
