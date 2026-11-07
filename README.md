# HACD: Heterogeneous Asymmetric Contrastive Distillation

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg" alt="arXiv"></a>
  <img src="https://img.shields.io/badge/PyTorch-%E2%89%A52.0-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/checkpoints-CC%20BY--NC%204.0-lightgrey.svg" alt="License">
</p>

Official PyTorch implementation of **HACD (Heterogeneous Asymmetric Contrastive Distillation)**, a logit-only knowledge distillation method for CNN, Transformer, and MLP teacher–student pairs.

> **Status.** Pretrained checkpoints and evaluation scripts are coming soon. The full training code will be released upon publication.

<p align="center">
  <img src="assets/hacd_overview.png" width="95%" alt="HACD overview">
</p>

## News

- **[2026-11]** Paper submitted; preprint on [arXiv](https://arxiv.org/abs/XXXX.XXXXX).

## Overview

Distilling across architectural families, e.g., from a Transformer teacher to a CNN student, is hard because the two networks learn different internal representations. Logits share a label space, but their values still depend on the architecture: Transformer and MLP teachers produce sharper outputs whose non-target classes are less semantically organized than those of CNN teachers. Matching these values sample by sample asks the student to imitate how another architecture expresses its predictions.

HACD instead transfers **how the teacher arranges samples relative to one another**:

- **Shared, comparable space.** Teacher and student logits are mapped to centered, temperature-softened probability directions on a unit hypersphere. Centering removes architecture-dependent logit offsets, and the temperature controls how much of the teacher's sharpness enters the representation while keeping informative non-target classes.
- **Relational transfer through contrast.** Each student output must identify its own teacher target among all teacher targets in the mini-batch (bidirectional instance-level contrast).
- **Asymmetric views.** Teacher targets are computed on a moderately augmented view and supervise both a moderate and a strong student view, so the student gains from strong augmentation while its supervision stays reliable.
- **Lightweight.** No projectors, alignment modules, or extra learnable parameters, and a single teacher forward pass per sample.

## Main Results

Top-1 accuracy (%) on heterogeneous teacher–student pairs. HACD achieves the best or second-best accuracy on **25 of 27** pairs. On CIFAR-100 it uses **no ground-truth labels** during distillation.

**Notation.** **Bold**: best; <u>underline</u>: second best. T./S.: teacher/student accuracy before distillation. Feature-based: FitNet, CC, RKD, CRD, PAT, RSD; logit-based: KD, DKD, DIST, OFA, TeKAP, LDRLD, HCD, HACD. Δ vs KD: improvement of HACD over the reported vanilla KD result. Average Gain: mean improvement over the student trained from scratch. "--": not reported. Baseline results are taken from the published evaluations of LDRLD [12], HCD [13], and RSD [6]; numbers in brackets refer to the [References](#references).

### CIFAR-100 (12 pairs, label-free distillation)

| Teacher | Student | T. | S. | FitNet [1] | CC [2] | RKD [3] | CRD [4] | PAT [5] | RSD [6] | KD [7] | DKD [8] | DIST [9] | OFA [10] | TeKAP [11] | LDRLD [12] | HCD [13] | **HACD** | Δ vs KD |
|:--|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| *CNN-based students* | | | | | | | | | | | | | | | | | | |
| Swin-T | ResNet-18 | 89.26 | 74.01 | 78.87 | 74.19 | 74.11 | 77.63 | 81.22 | <u>83.92</u> | 78.74 | 80.26 | 77.75 | 80.54 | 81.38\* | 82.17 | 82.78 | **85.00** | +6.26 |
| ViT-S | ResNet-18 | 92.04 | 74.01 | 77.71 | 74.26 | 73.72 | 76.60 | 80.11 | <u>81.50</u> | 77.26 | 78.10 | 76.49 | 80.15 | 79.06\* | 80.36 | 81.33 | **84.76** | +7.50 |
| Mixer-B/16 | ResNet-18 | 87.29 | 74.01 | 77.15 | 74.26 | 73.75 | 76.42 | 80.07 | <u>81.85</u> | 77.79 | 78.67 | 76.36 | 79.39 | 80.05\* | 80.69 | 81.53 | **83.95** | +6.16 |
| Swin-T | MobileNetV2 | 89.26 | 73.68 | 74.28 | 71.19 | 69.00 | 79.80 | 78.78 | <u>83.68</u> | 74.68 | 71.07 | 72.89 | 80.98 | 80.23\* | 81.64 | 82.19 | **85.38** | +10.70 |
| ViT-S | MobileNetV2 | 92.04 | 73.68 | 73.54 | 70.67 | 68.46 | 78.14 | 78.87 | <u>81.68</u> | 72.77 | 69.80 | 72.54 | 78.45 | 78.41\* | 79.21 | 80.81 | **84.66** | +11.89 |
| Mixer-B/16 | MobileNetV2 | 87.29 | 73.68 | 73.78 | 70.73 | 68.95 | 78.15 | 78.62 | <u>81.74</u> | 73.33 | 70.20 | 73.26 | 78.78 | 79.89\* | 80.64 | 81.09 | **84.20** | +10.87 |
| *ViT-based students* | | | | | | | | | | | | | | | | | | |
| ConvNeXt-T | DeiT-T | 88.41 | 68.00 | 60.78 | 68.01 | 69.79 | 65.94 | 79.59 | <u>82.46</u> | 72.99 | 74.60 | 73.55 | 75.76 | 76.32\* | 77.46 | -- | **82.50** | +9.51 |
| Mixer-B/16 | DeiT-T | 87.29 | 68.00 | 71.05 | 68.13 | 69.89 | 65.35 | 74.66 | <u>78.50</u> | 71.36 | 73.44 | 71.67 | 73.90 | 74.83\* | 75.31 | -- | **81.46** | +10.10 |
| ConvNeXt-T | Swin-P | 88.41 | 72.63 | 24.06 | 72.63 | 71.73 | 67.09 | 80.74 | <u>82.21</u> | 76.44 | 76.80 | 76.41 | 78.32 | 79.18\* | 80.71 | -- | **82.46** | +6.02 |
| Mixer-B/16 | Swin-P | 87.29 | 72.63 | 75.20 | 73.32 | 70.82 | 67.03 | 78.44 | <u>81.28</u> | 75.93 | 76.39 | 75.85 | 78.93 | 78.97\* | 80.52 | -- | **82.84** | +6.91 |
| *MLP-based students* | | | | | | | | | | | | | | | | | | |
| ConvNeXt-T | ResMLP-S12 | 88.41 | 66.56 | 45.47 | 67.70 | 65.82 | 63.35 | <u>83.50</u> | **84.21** | 72.25 | 73.22 | 71.93 | 81.22 | 81.14\* | 79.28 | -- | 81.84 | +9.59 |
| Swin-T | ResMLP-S12 | 89.26 | 66.56 | 63.12 | 68.37 | 64.66 | 61.72 | 80.94 | **82.67** | 71.89 | 72.82 | 11.05 | 80.63 | 80.22\* | 80.54 | -- | <u>82.23</u> | +10.34 |
| **Average Gain** |  |  |  | -5.20 | -0.33 | -1.40 | -0.02 | +8.17 | <u>+10.69</u> | +3.17 | +3.16 | -2.31 | +7.47 | +7.69 | +8.42 | -- | **+11.99** |  |

\* Results reproduced by LDRLD [12]. HCD is excluded from the Average Gain because results are available for only six pairs.

### ImageNet-1K (15 pairs)

| Teacher | Student | T. | S. | FitNet [1] | CC [2] | RKD [3] | CRD [4] | PAT [5] | RSD [6] | KD [7] | DKD [8] | DIST [9] | OFA [10] | TeKAP [11] | HCD [13] | **HACD** | Δ vs KD |
|:--|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| *CNN-based students* | | | | | | | | | | | | | | | | | |
| DeiT-T | ResNet-18 | 72.19 | 69.75 | 70.44 | 69.77 | 69.47 | 69.25 | -- | **71.70** | 70.22 | 69.39 | 70.64 | 71.34 | -- | -- | <u>71.50</u> | +1.28 |
| Swin-T | ResNet-18 | 81.38 | 69.75 | 71.18 | 70.07 | 68.89 | 69.09 | 71.54 | <u>72.13</u> | 71.14 | 71.10 | 70.91 | 71.85 | 71.25† | 71.91 | **72.76** | +1.62 |
| Mixer-B/16 | ResNet-18 | 76.62 | 69.75 | 70.78 | 70.05 | 69.46 | 68.40 | -- | 71.41 | 70.89 | 69.89 | 70.66 | 71.38 | 70.95† | <u>71.66</u> | **71.81** | +0.92 |
| DeiT-T | MobileNetV2 | 72.17 | 68.87 | 70.95 | 70.69 | 69.72 | 69.60 | -- | <u>72.18</u> | 70.87 | 70.14 | 71.08 | 71.39 | 70.23† | 71.38 | **72.88** | +2.01 |
| Swin-T | MobileNetV2 | 81.38 | 68.87 | 71.75 | 70.69 | 67.52 | 69.58 | -- | 72.36 | 72.05 | 71.71 | 71.76 | 72.32 | 72.50† | <u>72.72</u> | **73.38** | +1.33 |
| Mixer-B/16 | MobileNetV2 | 76.62 | 68.87 | 71.59 | 70.79 | 69.86 | 68.89 | 72.22 | 71.90 | 71.92 | 70.93 | 71.74 | 72.12 | 72.01† | <u>72.32</u> | **73.13** | +1.21 |
| *ViT-based students* | | | | | | | | | | | | | | | | | |
| ResNet-50 | DeiT-T | 80.33 | 72.17 | 75.84 | 72.56 | 72.06 | 68.53 | -- | <u>76.17</u>\* | 75.10 | 75.60\* | 75.13\* | **76.55**\* | -- | -- | <u>76.17</u> | +1.07 |
| ConvNeXt-T | DeiT-T | 82.05 | 72.17 | 70.45 | 73.12 | 71.47 | 69.18 | 74.44 | <u>74.46</u> | 74.00 | 73.95 | 74.07 | 74.41 | -- | -- | **75.89** | +1.89 |
| Mixer-B/16 | DeiT-T | 76.55 | 72.17 | 74.38 | 72.82 | 72.24 | 68.23 | -- | 74.26 | 74.16 | 72.82 | 74.22 | <u>74.46</u> | -- | -- | **75.76** | +1.60 |
| ResNet-50 | Swin-N | 80.33 | 75.53 | 78.33 | 76.05 | 75.90 | 73.90 | -- | **78.78**\* | 77.58 | 78.23\* | 77.95\* | <u>78.64</u>\* | -- | -- | 78.30 | +0.72 |
| ConvNeXt-T | Swin-N | 82.05 | 75.53 | 74.81 | 75.79 | 75.48 | 74.15 | -- | <u>77.70</u> | 77.15 | 77.00 | 77.25 | 77.50 | -- | -- | **77.97** | +0.82 |
| Mixer-B/16 | Swin-N | 76.55 | 75.53 | 76.17 | 75.81 | 75.52 | 73.38 | -- | 76.55 | 76.26 | 75.03 | 76.54 | <u>76.63</u> | -- | -- | **77.00** | +0.74 |
| *MLP-based students* | | | | | | | | | | | | | | | | | |
| ResNet-50 | ResMLP-S12 | 80.33 | 76.65 | 78.13 | 76.21 | 75.45 | 73.23 | -- | 78.32\* | 77.41 | 78.23\* | 77.71\* | <u>78.53</u>\* | -- | -- | **78.71** | +1.30 |
| ConvNeXt-T | ResMLP-S12 | 82.05 | 76.65 | 74.69 | 75.79 | 75.28 | 73.57 | -- | <u>78.41</u> | 76.84 | 77.23 | 77.24 | 77.53 | -- | -- | **78.65** | +1.81 |
| Swin-T | ResMLP-S12 | 81.35 | 76.65 | 76.48 | 76.15 | 75.10 | 73.40 | 77.59 | <u>77.61</u> | 76.67 | 76.99 | 77.25 | 77.31 | -- | -- | **78.35** | +1.68 |
| **Average Gain** |  |  |  | +1.14 | +0.50 | -0.37 | -1.77 | -- | <u>+2.34</u> | +1.56 | +1.29 | +1.68 | +2.20 | -- | -- | **+2.89** |  |

\* Results of the method combined with FitNet, following OFA [10]. † Results taken from HCD [13]. Average Gain is reported only for methods with results on all 15 pairs.

## Model Zoo

| Dataset | Teacher → Student | Top-1 | Checkpoint | Log |
|:--|:--|:-:|:-:|:-:|
| CIFAR-100 | 12 pairs | see [Main Results](#cifar-100-12-pairs-label-free-distillation) | Coming soon | Coming soon |
| ImageNet-1K | 15 pairs | see [Main Results](#imagenet-1k-15-pairs) | Coming soon | Coming soon |

## Getting Started

- **Evaluation:** Coming soon, together with the checkpoints.
- **Training:** Coming soon. The training code will be released upon publication.

## Roadmap

- [ ] Distilled student checkpoints and training logs
- [ ] Evaluation scripts
- [ ] Training code and configurations for all 27 pairs (upon publication)

## Citation

If you find this work useful, please cite:

```bibtex
@article{wang2026hacd,
  title   = {Heterogeneous Asymmetric Contrastive Distillation},
  author  = {Wang, Qi and Zhou, Jinjia},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## References

1. A. Romero et al., "FitNets: Hints for Thin Deep Nets," *ICLR*, 2015.
2. B. Peng et al., "Correlation Congruence for Knowledge Distillation," *ICCV*, 2019.
3. W. Park et al., "Relational Knowledge Distillation," *CVPR*, 2019.
4. Y. Tian et al., "Contrastive Representation Distillation," *ICLR*, 2020.
5. J.-H. Lin et al., "Perspective-Aware Teaching: Adapting Knowledge for Heterogeneous Distillation," *ICCV*, 2025.
6. W. Zhang et al., "Cross-Architecture Distillation Made Simple with Redundancy Suppression," *ICCV*, 2025.
7. G. Hinton et al., "Distilling the Knowledge in a Neural Network," *NIPS Deep Learning Workshop*, 2015.
8. B. Zhao et al., "Decoupled Knowledge Distillation," *CVPR*, 2022.
9. T. Huang et al., "Knowledge Distillation from A Stronger Teacher," *NeurIPS*, 2022.
10. Z. Hao et al., "One-for-All: Bridge the Gap Between Heterogeneous Architectures in Knowledge Distillation," *NeurIPS*, 2023.
11. M. I. Hossain et al., "Single Teacher, Multiple Perspectives: Teacher Knowledge Augmentation for Enhanced Knowledge Distillation," *ICLR*, 2025.
12. L. Xu et al., "Local Dense Logit Relations for Enhanced Knowledge Distillation," *ICCV*, 2025.
13. L. Xu et al., "Heterogeneous Complementary Distillation," *AAAI*, 2026.

## License

- **Checkpoints and training logs:** [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Free for non-commercial research with attribution.
- **Code:** will be released under an open-source license together with the training code.

The datasets and the pretrained teacher models remain subject to their original licenses and terms of use.

## Acknowledgments

Our experimental protocol follows OFA [10] ([code](https://github.com/Hao840/OFAKD)). We thank the authors for making their code available.

## Contact

Qi Wang — qi.wang.7c@stu.hosei.ac.jp
