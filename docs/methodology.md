# Methodology

## Research question

Does learning a bounded, input-dependent power improve shallow image classification over a matched frozen profile and over competitive learned activations, under equal validation and training budgets, and at what computational cost?

## The activation

For a scalar pre-activation $z$:

$$
A = a_{\max}\tanh(a), \qquad p(z) = p_0 + A\tanh(wz + b), \qquad f(z) = z\,(z^2+\epsilon)^{(p(z)-1)/2}
$$

Defaults: $p_0 = 1$, $a_{\max} = 0.5$, $\epsilon = 10^{-3}$, initial $A = 0.3$ (so $a = \operatorname{atanh}(0.6)$), $w = 0.5$, $b = 0$. With these values $0.5 < p(z) < 1.5$.

- $a$, $w$ and $b$ are learned and shared by all channels and positions of one activation layer.
- The function maps 0 to 0, keeps the sign of every input, and is exactly the identity when $p_0 = 1$ and $A = 0$.
- The exponent and power are computed in float32, also under bfloat16 autocast.
- A bounded exponent does not guarantee monotonicity or bounded gradients. For example, $A = 0.49$, $w = 20$, $b = -10$ gives a negative derivative at $z = 0.5$.

## Methods compared

| Method | What is learned | Added parameters (SimpleCNN) |
|---|---|---|
| ReLU | nothing | 0 |
| GELU | nothing | 0 |
| PReLU | one negative slope per layer | 3 |
| ACON-C | $p_1$, $p_2$, $\beta$ per channel | 672 |
| Learned constant power | one exponent per layer, $p_c = 0.5 + \sigma(c)$, starting at $p_c = 1$ (identity) | 3 |
| APU frozen | nothing: the full APU at its initial values | 0 (9 fixed) |
| APU (full) | $a$, $w$, $b$ per layer | 9 |

The frozen APU starts from exactly the same function as the full APU; only its activation parameters are excluded from optimisation. It is the direct test of whether learning adds value. The learned constant power starts as the identity, so it also differs from APU in its starting function.

ACON-C is a re-implementation of Ma et al. (CVPR 2021), checked against two identities: $p_1 = 1, p_2 = 0, \beta = 1$ gives SiLU, and large $\beta$ approaches $\max(p_1 z, p_2 z)$.

## Model and training

- **Model.** SimpleCNN: two unpadded 3×3 convolutions (32 and 64 channels), each followed by the activation; 2×2 max pooling; a 128-unit fully connected layer with the activation; an output layer. No normalisation or dropout. 1,626,442 parameters with ReLU on CIFAR-10.
- **Data.** CIFAR-10 and CIFAR-100. Each 50,000-image training set is split into 45,000 for fitting and a fixed stratified validation set of 5,000 (split seed 20261030). The 10,000 test images are used once per final run.
- **Training.** AdamW (β = 0.9, 0.999), batch 128, 200 epochs, one epoch of linear warm-up then cosine decay per step. Random crop (4-pixel padding) and horizontal flip. Weight decay applies only to weight matrices and convolution kernels, not to biases or activation parameters. The final-epoch checkpoint is evaluated; no checkpoint is chosen by test accuracy.
- **Pairing.** For a given seed, every method starts from identical backbone weights and sees the same data order and augmentation. Each random stream (initialisation, order, augmentation, diagnostic samples) has its own generator keyed by seed, epoch or layer name.

## Procedure

The steps run in this order, and the code refuses to run them out of order:

1. **Protocol.** `apu_v2_runs/protocol/protocol.json` fixes the methods, search grid, seeds, comparisons, statistics and decision rules. It was written before any search run and cannot be changed afterwards.
2. **Equal search.** Each method gets the same 6 candidates (learning rate 5e-4, 1e-3, 2e-3 × weight decay 1e-4, 5e-4), 2 selection seeds (1000, 1001) and 200 epochs: 84 runs. The candidate with the highest mean final validation accuracy is selected; ties go to the earlier candidate in the grid.
3. **Lock.** `lock.json` records the selected settings and the fingerprint of every final run before any final run starts.
4. **Final runs.** Seeds 0–4, three groups of 35 runs:
    - primary: CIFAR-10 with each method's selected setting;
    - common recipe (exploratory): CIFAR-10 with learning rate 1e-3 and weight decay 1e-4 for every method;
    - confirmation: CIFAR-100 with the CIFAR-10 selections unchanged, declared as the confirmation setting before any results.
5. **Test evaluation.** After every locked run had finished, each was evaluated on the test set once.

Selected settings:

| Method | Learning rate | Weight decay |
|---|---|---|
| ReLU, GELU, PReLU | 2e-3 | 1e-4 |
| ACON-C | 2e-3 | 5e-4 |
| Learned constant power, APU (full) | 1e-3 | 5e-4 |
| APU frozen | 5e-4 | 1e-4 |

Four methods selected the largest learning rate, and every weight decay sits at an edge of its two-value range. The search establishes fairness within this grid, not each method's best possible accuracy.

## Statistics

- **Primary comparison:** full APU minus frozen APU, on CIFAR-10, repeated on CIFAR-100.
- **Secondary comparisons (Holm-corrected within each dataset):** full APU against the strongest standard activation (ReLU or GELU), the strongest adaptive activation (PReLU or ACON-C), and the learned constant power. "Strongest" is decided by validation accuracy, never test accuracy.
- **Tests:** paired t-intervals over the 5 seeds, used because identical initial weights and data order were verified for every pair. α = 0.05.
- **Reporting:** every seed is kept; failures would stay in the tables with their denominators (none occurred).

## Decision rules (fixed in the protocol)

| Rule | Condition | Interpretation |
|---|---|---|
| A | Full APU beats frozen, the strongest standard and the strongest adaptive baseline on both datasets | Evidence for learned power adaptation in these settings |
| B | Full and frozen APU are similar | A fixed profile may explain any gain |
| C | The benefit appears only on CIFAR-10 | A restricted observation needing confirmation |
| D | Accuracy improves but runtime increases | An accuracy–cost trade-off |
| E | No reproducible advantage over strong controls | Revise the central claim |

Outcome of this study: **E**. Full APU beat the frozen version clearly but did not beat ACON-C on either dataset. The conditional extensions (a depth study and a Padé baseline) ran only under rule A, so they were not run.

## Cost measurement

Each method was measured in a fresh Python process, in eager and compiled modes, over three interleaved rounds. Each measurement used warm-up iterations and several synchronised timing blocks. The measurements cover a training step, a full epoch including data loading, inference at batch sizes 1 and 128, and peak GPU memory. Results are in `apu_v2_runs/paper_assets/tables/resources_absolute_and_relative.csv`.

## Diagnostics

At epochs 1, 5, 25, 50, 100, 150 and 200, each run recorded:
- clean training and validation accuracy (same preprocessing, evaluation mode);
- quantiles of every activation's input and output on 1,024 fixed validation images;
- APU exponent distributions and the amplitude $A$;
- per-layer gradient norms and non-finite events.

These are observations. They describe what changed, not why accuracy changed.
