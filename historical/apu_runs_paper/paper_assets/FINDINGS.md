# Measured findings (auto-generated 2026-10-29 03:34)
Profile `local`, size `paper`, device cuda.

## Recognition: final-epoch test top-1 (mean ± std over seeds)
- **cifar10 / ResNet-18**: APU 94.51 ± 0.14 (n=5). Best baseline ESReLU 95.06. APU is 0.55 points behind it (95% CI [-0.75, -0.35], Welch p = 0.000242); versus ReLU -0.38.
- **cifar10 / SimpleCNN**: APU 83.33 ± 0.42 (n=5). Best baseline GELU 82.53. APU is 0.80 points ahead of it (95% CI [0.29, 1.31], Welch p = 0.0102); versus ReLU +1.20.
- **cifar10 / ViT-Tiny**: APU 80.59 ± 1.03 (n=5). Best baseline PReLU 84.71. APU is 4.12 points behind it (95% CI [-5.39, -2.84], Welch p = 0.000223); versus ReLU -3.67.
- **cifar100 / ResNet-18**: APU 75.40 ± 0.48 (n=5). Best baseline SiLU 77.43. APU is 2.03 points behind it (95% CI [-2.62, -1.44], Welch p = 0.000131); versus ReLU -1.22.
- **cifar100 / ViT-Tiny**: APU 52.92 ± 0.62 (n=5). Best baseline LeakyReLU 57.44. APU is 4.53 points behind it (95% CI [-5.33, -3.73], Welch p = 2.71e-06); versus ReLU -4.50.
- **emnist / ResNet-18**: APU 88.61 ± 0.08 (n=3). Best baseline ReLU 89.24. APU is 0.63 points behind it (95% CI [-0.80, -0.47], Welch p = 0.000471); versus ReLU -0.63.
- **emnist / SimpleCNN**: APU 84.52 ± 0.18 (n=3). Best baseline GELU 86.57. APU is 2.05 points behind it (95% CI [-2.43, -1.67], Welch p = 0.000144); versus ReLU -2.01.
- **emnist / ViT-Tiny**: APU 87.51 ± 0.19 (n=3). Best baseline ReLU 87.83. APU is 0.32 points behind it (95% CI [-0.73, 0.09], Welch p = 0.0992); versus ReLU -0.32.
- **mnist / ResNet-18**: APU 99.56 ± 0.04 (n=3). Best baseline ReLU 99.63. APU is 0.07 points behind it (95% CI [-0.15, 0.02], Welch p = 0.101); versus ReLU -0.07.
- **mnist / SimpleCNN**: APU 98.99 ± 0.10 (n=3). Best baseline GELU 99.24. APU is 0.25 points behind it (95% CI [-0.46, -0.04], Welch p = 0.03); versus ReLU -0.23.
- **mnist / ViT-Tiny**: APU 97.99 ± 0.02 (n=3). Best baseline LeakyReLU 99.09. APU is 1.10 points behind it (95% CI [-1.16, -1.04], Welch p = 3.47e-06); versus ReLU -1.07.
- **tinyimagenet / ResNet-18**: APU 58.87 ± 0.34 (n=5). Best baseline SiLU 62.88. APU is 4.01 points behind it (95% CI [-4.43, -3.59], Welch p = 7.9e-07); versus ReLU -3.27.
- **tinyimagenet / ViT-Tiny**: APU 36.90 ± 0.41 (n=5). Best baseline ReLU 39.91. APU is 3.01 points behind it (95% CI [-3.65, -2.37], Welch p = 4.95e-06); versus ReLU -3.01.

APU beats the best baseline in 1 of 13 settings.

Diverged runs (NaN loss, stopped early): Fixed p=2 ×5, Linear p(z) ×5.

## Efficiency (APU vs ReLU)
- cifar100_resnet18: train step ms +226%, train step compiled ms +40%, peak mem compiled mb +12%, latency bs ms +87%, latency bs amp ms +177%, peak mem mb +335%, epoch time s +39%; parameters +0.0005%.
- cifar10_resnet18: train step ms +227%, train step compiled ms +31%, peak mem compiled mb +12%, latency bs ms +87%, latency bs amp ms +178%, peak mem mb +336%, epoch time s +39%; parameters +0.0005%.

## Robustness (accuracy under corruption, mean over corruptions and severities)
- cifar10 / ResNet-18: APU 62.86% vs ReLU 64.76% (-1.90 points); best activation SiLU 65.92%. Source: reimplemented-subset.
- cifar100 / ResNet-18: APU 40.12% vs ReLU 39.85% (+0.27 points); best activation SiLU 41.12%. Source: reimplemented-subset.

## Learned exponent
- cifar100_resnet18: mean |ΔA| from init 0.213, |Δw_z| 2.476, |Δb_z| 2.195; std of p(z) inside a layer 0.305; std of layer-mean p across depth 0.244; mean p on negative inputs 0.613 vs positive inputs 1.147.
- cifar10_resnet18: mean |ΔA| from init 0.235, |Δw_z| 2.691, |Δb_z| 2.328; std of p(z) inside a layer 0.268; std of layer-mean p across depth 0.278; mean p on negative inputs 0.614 vs positive inputs 1.059.
- cifar10_vit_tiny: mean |ΔA| from init 0.200, |Δw_z| 0.653, |Δb_z| 4.250; std of p(z) inside a layer 0.050; std of layer-mean p across depth 0.203; mean p on negative inputs 1.412 vs positive inputs 1.463.

## GAN (Fréchet distance in classifier-feature space, lower is better)
- emnist / dcgan: SiLU 131.9, Mish 134.6, GELU 139.7, ReLU 144.8, PReLU 149.4, LeakyReLU 150.1, ESReLU 151.0, APU (ours) 168.2
- emnist / mlp: LeakyReLU 351.1, ESReLU 359.1, Mish 395.6, ReLU 404.8, SiLU 410.7, APU (ours) 413.4, PReLU 469.1, GELU 603.7
- mnist / dcgan: Mish 59.8, SiLU 63.0, GELU 63.5, PReLU 65.5, ReLU 66.7, ESReLU 68.7, LeakyReLU 69.4, APU (ours) 118.8
- mnist / mlp: LeakyReLU 187.4, ReLU 193.7, PReLU 225.5, ESReLU 288.7, APU (ours) 356.6, GELU 531.5, Mish 560.4, SiLU 564.0

## Convergence (epochs to reach 95% of ReLU's final accuracy)
- cifar10 / ResNet-18: APU 98.0 (reached in 5/5 runs) vs ReLU 76.0.
- cifar10 / SimpleCNN: APU 36.0 (reached in 5/5 runs) vs ReLU 20.8.
- cifar10 / ViT-Tiny: APU 105.2 (reached in 4/5 runs) vs ReLU 33.0.
- cifar100 / ResNet-18: APU 161.6 (reached in 5/5 runs) vs ReLU 151.2.
- cifar100 / ViT-Tiny: APU nan (reached in 0/5 runs) vs ReLU 41.4.
- emnist / ResNet-18: APU 2.3 (reached in 3/3 runs) vs ReLU 3.0.
- emnist / SimpleCNN: APU 2.0 (reached in 3/3 runs) vs ReLU 1.7.
- emnist / ViT-Tiny: APU 8.7 (reached in 3/3 runs) vs ReLU 4.3.
- mnist / ResNet-18: APU 1.0 (reached in 3/3 runs) vs ReLU 1.3.
- mnist / SimpleCNN: APU 1.0 (reached in 3/3 runs) vs ReLU 1.0.
- mnist / ViT-Tiny: APU 9.7 (reached in 3/3 runs) vs ReLU 2.7.
- tinyimagenet / ResNet-18: APU 96.7 (reached in 3/5 runs) vs ReLU 81.2.
- tinyimagenet / ViT-Tiny: APU 44.2 (reached in 5/5 runs) vs ReLU 13.6.

## How to read this
The decision rule from the requirements document applies: do not claim APU is better everywhere. State where it helps, where it does not, and what it costs.