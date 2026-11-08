**fig10_learned_shapes_cifar100_resnet18**: Figure 10. Learned f(z) and f'(z) by depth with spread over layers and seeds, compared with ReLU, GELU and SiLU.

**fig10_learned_shapes_cifar10_resnet18**: Figure 10. Learned f(z) and f'(z) by depth with spread over layers and seeds, compared with ReLU, GELU and SiLU.

**fig10_learned_shapes_cifar10_vit_tiny**: Figure 10. Learned f(z) and f'(z) by depth with spread over layers and seeds, compared with ReLU, GELU and SiLU.

**fig1_motivation**: Figure 1. (a) Distribution of pre-activations at three depths of a trained ReLU network (measured). (b) Shapes learned by APU at three depths (trained parameters). (c) APU computation: an input-dependent exponent p(z) followed by a smooth power.

**fig2_exponent_cifar100_resnet18**: Figure 2. Learned exponent of a trained APU network: (a) layer-wise mean and within-layer spread, (b) distribution by depth, (c) channel-wise means, (d) exponent against input, (e) learned f(z), (f) spread of p(z) over training.

**fig2_exponent_cifar10_resnet18**: Figure 2. Learned exponent of a trained APU network: (a) layer-wise mean and within-layer spread, (b) distribution by depth, (c) channel-wise means, (d) exponent against input, (e) learned f(z), (f) spread of p(z) over training.

**fig2_exponent_cifar10_vit_tiny**: Figure 2. Learned exponent of a trained APU network: (a) layer-wise mean and within-layer spread, (b) distribution by depth, (c) channel-wise means, (d) exponent against input, (e) learned f(z), (f) spread of p(z) over training.

**fig3_recognition_cifar10**: Figure 3 (one figure per dataset). Final-epoch test top-1 (bars: mean, whiskers: 95% CI, dots: individual seeds) and validation curves (mean ± std over seeds) for every model.

**fig3_recognition_cifar100**: Figure 3 (one figure per dataset). Final-epoch test top-1 (bars: mean, whiskers: 95% CI, dots: individual seeds) and validation curves (mean ± std over seeds) for every model.

**fig3_recognition_emnist**: Figure 3 (one figure per dataset). Final-epoch test top-1 (bars: mean, whiskers: 95% CI, dots: individual seeds) and validation curves (mean ± std over seeds) for every model.

**fig3_recognition_mnist**: Figure 3 (one figure per dataset). Final-epoch test top-1 (bars: mean, whiskers: 95% CI, dots: individual seeds) and validation curves (mean ± std over seeds) for every model.

**fig3_recognition_tinyimagenet**: Figure 3 (one figure per dataset). Final-epoch test top-1 (bars: mean, whiskers: 95% CI, dots: individual seeds) and validation curves (mean ± std over seeds) for every model.

**fig3b_gap_to_relu**: Figure 3b. Difference to ReLU in test top-1 (mean and 95% Welch confidence interval). Points left of zero are worse than ReLU.

**fig4a_optimization_cifar100_resnet18**: Figure 4a. Optimisation behaviour: training loss, global gradient norm, epochs to reach 95% of ReLU's final accuracy, and per-layer gradient norms at initialisation and after training.

**fig4a_optimization_cifar10_resnet18**: Figure 4a. Optimisation behaviour: training loss, global gradient norm, epochs to reach 95% of ReLU's final accuracy, and per-layer gradient norms at initialisation and after training.

**fig4a_optimization_cifar10_vit_tiny**: Figure 4a. Optimisation behaviour: training loss, global gradient norm, epochs to reach 95% of ReLU's final accuracy, and per-layer gradient norms at initialisation and after training.

**fig4b_robust_efficiency_cifar100_resnet18**: Figure 4b. Corruption accuracy (heat map, severity curves), PGD sanity check, and efficiency: overhead versus ReLU, accuracy versus cost, measured time per epoch.

**fig4b_robust_efficiency_cifar10_resnet18**: Figure 4b. Corruption accuracy (heat map, severity curves), PGD sanity check, and efficiency: overhead versus ReLU, accuracy versus cost, measured time per epoch.

**fig5_gan_emnist_dcgan**: Figure 5. GAN training curves, discriminator scores and sample-quality metrics computed in the feature space of a classifier trained on real data.

**fig5_gan_emnist_mlp**: Figure 5. GAN training curves, discriminator scores and sample-quality metrics computed in the feature space of a classifier trained on real data.

**fig5_gan_mnist_dcgan**: Figure 5. GAN training curves, discriminator scores and sample-quality metrics computed in the feature space of a classifier trained on real data.

**fig5_gan_mnist_mlp**: Figure 5. GAN training curves, discriminator scores and sample-quality metrics computed in the feature space of a classifier trained on real data.

**fig5b_gan_samples_emnist_dcgan**: Figure 5b. Generated samples from the same fixed noise vectors after the last epoch (first seed).

**fig5b_gan_samples_emnist_mlp**: Figure 5b. Generated samples from the same fixed noise vectors after the last epoch (first seed).

**fig5b_gan_samples_mnist_dcgan**: Figure 5b. Generated samples from the same fixed noise vectors after the last epoch (first seed).

**fig5b_gan_samples_mnist_mlp**: Figure 5b. Generated samples from the same fixed noise vectors after the last epoch (first seed).

**fig6_ablation_stability**: Figure 6. (a) Ablation of the exponent design, (b) largest activation at initialisation (log scale), (c) pilot sweep over APU initial settings on a held-out split.

**fig7_lr_sweep**: Figure 7. Learning-rate sensitivity on the held-out split and the best learning rate per activation.

**fig8_parameter_dynamics_cifar100_resnet18**: Figure 8. Evolution of the learned APU parameters per layer during training.

**fig8_parameter_dynamics_cifar10_resnet18**: Figure 8. Evolution of the learned APU parameters per layer during training.

**fig8_parameter_dynamics_cifar10_vit_tiny**: Figure 8. Evolution of the learned APU parameters per layer during training.

**fig9_rank_heatmap**: Figure 9. Rank of every activation within each dataset and model.

**figP01_activation_functions**: Figure P1. (a) The compared activation functions and (b) their derivatives; (c) APU for constant exponents p and a shape learned during training.

**figP02_main_results**: Figure P2. Difference in test top-1 to ReLU in every dataset/model setting (mean and 95% Welch confidence interval). * marks settings where APU differs from ReLU at p < 0.05.

**figP03_rank_cd_diagram**: Figure P3. (a) Critical-difference diagram (Friedman test, Nemenyi post-hoc, alpha = 0.05) of the average rank over all settings; activations joined by a bar are not significantly different. (b) Rank in each setting.

**figP04_win_tie_loss**: Figure P4. Number of settings in which APU is significantly better, not significantly different, or significantly worse than each baseline (Welch test, p < 0.05).

**figP05_seed_distributions**: Figure P5. Distribution of final test top-1 over seeds (box: quartiles, dots: individual seeds, diamond: mean).

**figP06_gap_over_training**: Figure P6. Gap in validation accuracy between APU and ReLU / GELU during training (mean and 95% band over seeds).

**figP07_accuracy_vs_cost**: Figure P7. (a) Accuracy against training cost (mean over settings; dotted line joins the non-dominated activations). (b) Accuracy against inference latency.

**figP08_robustness_cifar100_resnet18**: Figure P8. Robustness to common corruptions: (a) mean accuracy, (b) accuracy retained relative to clean accuracy, (c) difference to ReLU per corruption type.

**figP08_robustness_cifar10_resnet18**: Figure P8. Robustness to common corruptions: (a) mean accuracy, (b) accuracy retained relative to clean accuracy, (c) difference to ReLU per corruption type.

**figP09_efficiency_cifar100_resnet18**: Figure P9. Cost relative to ReLU: (a) training time, (b) inference latency in fp32 and AMP, (c) peak training memory in eager and compiled mode.

**figP09_efficiency_cifar10_resnet18**: Figure P9. Cost relative to ReLU: (a) training time, (b) inference latency in fp32 and AMP, (c) peak training memory in eager and compiled mode.

**figP10_gan_comparison**: Figure P10. GAN comparison: Frechet distance in classifier-feature space for every dataset and architecture (bars: mean, whiskers: std, dots: seeds).

**figP11_exponent_asymmetry**: Figure P11. Learned exponent of APU on negative and positive inputs by depth (mean and std over seeds). p < 1 compresses, p > 1 expands.

**figP12_generalization_gap**: Figure P12. Generalisation gap (train accuracy minus test accuracy) at the end of training.

**figP13_stability_vs_accuracy**: Figure P13. Largest activation at initialisation against held-out accuracy of the pilot variants; diverged variants are shown in the bottom band.

**figP14_overview_radar**: Figure P14. Overview of accuracy, seed stability, speed, robustness and memory (each axis min-max scaled over the shown activations; outer is better).