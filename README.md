# HACD: Heterogeneous Asymmetric Contrastive Distillation

<p align="center">
  <a href="http://arxiv.org/abs/2611.10672"><img src="https://img.shields.io/badge/arXiv-2611.10672-b31b1b.svg" alt="arXiv"></a>
  <img src="https://img.shields.io/badge/PyTorch-%E2%89%A52.0-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/checkpoints-CC%20BY--NC%204.0-lightgrey.svg" alt="License">
</p>

Official PyTorch implementation of **HACD (Heterogeneous Asymmetric Contrastive Distillation)**, a logit-only knowledge distillation method for CNN, Transformer, and MLP teacher–student pairs.

> **Status.** CIFAR-100 checkpoints are available in the [Model Zoo](#model-zoo). ImageNet-1K checkpoints and evaluation scripts are coming soon. The full training code will be released upon publication.

<!-- <p align="center">
  <img src="assets/hacd_overview.png" width="95%" alt="HACD overview">
</p> -->

## News

- **[2026-11]** CIFAR-100 checkpoints for all 12 pairs released ([release](https://github.com/BTAEYANG/HACD-Heterogeneous-Asymmetric-Contrastive-Distillation/releases/tag/cifar-100-checkpoints)).
- **[2026-11]** Paper submitted; preprint on [arXiv](http://arxiv.org/abs/2611.10672).

## Overview

Distilling across architectural families, e.g., from a Transformer teacher to a CNN student, is hard because the two networks learn different internal representations. Logits share a label space, but their values still depend on the architecture: Transformer and MLP teachers produce sharper outputs whose non-target classes are less semantically organized than those of CNN teachers. Matching these values sample by sample asks the student to imitate how another architecture expresses its predictions.

HACD instead transfers **how the teacher arranges samples relative to one another**:

- **Shared, comparable space.** Teacher and student logits are mapped to centered, temperature-softened probability directions on a unit hypersphere. Centering removes architecture-dependent logit offsets, and the temperature controls how much of the teacher's sharpness enters the representation while keeping informative non-target classes.
- **Relational transfer through contrast.** Each student output must identify its own teacher target among all teacher targets in the mini-batch (bidirectional instance-level contrast).
- **Asymmetric views.** Teacher targets are computed on a moderately augmented view and supervise both a moderate and a strong student view, so the student gains from strong augmentation while its supervision stays reliable.
- **Lightweight.** No projectors, alignment modules, or extra learnable parameters, and a single teacher forward pass per sample.

## Main Results

Top-1 accuracy (%) on heterogeneous teacher–student pairs. HACD achieves the best or second-best accuracy on **25 of 27** pairs. On CIFAR-100 it uses **no ground-truth labels** during distillation.

**Notation.** Each dataset is split into two tables, (a) feature-based and (b) logit-based methods, with HACD repeated in both. **Bold**: best; <u>underline</u>: second best, ranked over all methods of both tables. Teacher → Student (T. / S. acc.): teacher and student accuracies before distillation. Feature-based: FitNet, CC, RKD, CRD, PAT, RSD; logit-based: KD, DKD, DIST, OFA, TeKAP, LDRLD, HCD, HACD. Δ: improvement of HACD over the reported vanilla KD result. Average Gain: mean improvement over the student trained from scratch. "--": not reported. Baseline results are taken from the published evaluations of LDRLD [12], HCD [13], and RSD [6]; numbers in brackets refer to the [References](#references).

### CIFAR-100 (12 pairs, label-free distillation)

**(a) vs. feature-based methods**

| Teacher<br>→ Student<br><sub>(T. / S.)</sub> | FitNet<sup>1</sup> | CC<sup>2</sup> | RKD<sup>3</sup> | CRD<sup>4</sup> | PAT<sup>5</sup> | RSD<sup>6</sup> | **HACD** |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ***CNN-based students*** | | | | | | | |
| Swin-T<br>→ ResNet-18<br><sub>(89.26 / 74.01)</sub> | 78.87 | 74.19 | 74.11 | 77.63 | 81.22 | <u>83.92</u> | **85.00** |
| ViT-S<br>→ ResNet-18<br><sub>(92.04 / 74.01)</sub> | 77.71 | 74.26 | 73.72 | 76.60 | 80.11 | <u>81.50</u> | **84.76** |
| Mixer-B/16<br>→ ResNet-18<br><sub>(87.29 / 74.01)</sub> | 77.15 | 74.26 | 73.75 | 76.42 | 80.07 | <u>81.85</u> | **83.95** |
| Swin-T<br>→ MobileNetV2<br><sub>(89.26 / 73.68)</sub> | 74.28 | 71.19 | 69.00 | 79.80 | 78.78 | <u>83.68</u> | **85.38** |
| ViT-S<br>→ MobileNetV2<br><sub>(92.04 / 73.68)</sub> | 73.54 | 70.67 | 68.46 | 78.14 | 78.87 | <u>81.68</u> | **84.66** |
| Mixer-B/16<br>→ MobileNetV2<br><sub>(87.29 / 73.68)</sub> | 73.78 | 70.73 | 68.95 | 78.15 | 78.62 | <u>81.74</u> | **84.20** |
| ***ViT-based students*** | | | | | | | |
| ConvNeXt-T<br>→ DeiT-T<br><sub>(88.41 / 68.00)</sub> | 60.78 | 68.01 | 69.79 | 65.94 | 79.59 | <u>82.46</u> | **82.50** |
| Mixer-B/16<br>→ DeiT-T<br><sub>(87.29 / 68.00)</sub> | 71.05 | 68.13 | 69.89 | 65.35 | 74.66 | <u>78.50</u> | **81.46** |
| ConvNeXt-T<br>→ Swin-P<br><sub>(88.41 / 72.63)</sub> | 24.06 | 72.63 | 71.73 | 67.09 | 80.74 | <u>82.21</u> | **82.46** |
| Mixer-B/16<br>→ Swin-P<br><sub>(87.29 / 72.63)</sub> | 75.20 | 73.32 | 70.82 | 67.03 | 78.44 | <u>81.28</u> | **82.84** |
| ***MLP-based students*** | | | | | | | |
| ConvNeXt-T<br>→ ResMLP-S12<br><sub>(88.41 / 66.56)</sub> | 45.47 | 67.70 | 65.82 | 63.35 | <u>83.50</u> | **84.21** | 81.84 |
| Swin-T<br>→ ResMLP-S12<br><sub>(89.26 / 66.56)</sub> | 63.12 | 68.37 | 64.66 | 61.72 | 80.94 | **82.67** | <u>82.23</u> |
| **Average<br>Gain** | -5.20 | -0.33 | -1.40 | -0.02 | +8.17 | <u>+10.69</u> | **+11.99** |

**(b) vs. logit-based methods**

| Teacher<br>→ Student<br><sub>(T. / S.)</sub> | KD<sup>7</sup> | DKD<sup>8</sup> | DIST<sup>9</sup> | OFA<sup>10</sup> | TeKAP<sup>11</sup> | LDRLD<sup>12</sup> | HCD<sup>13</sup> | **HACD** | Δ |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ***CNN-based students*** | | | | | | | | | |
| Swin-T<br>→ ResNet-18<br><sub>(89.26 / 74.01)</sub> | 78.74 | 80.26 | 77.75 | 80.54 | 81.38<sup>\*</sup> | 82.17 | 82.78 | **85.00** | +6.26 |
| ViT-S<br>→ ResNet-18<br><sub>(92.04 / 74.01)</sub> | 77.26 | 78.10 | 76.49 | 80.15 | 79.06<sup>\*</sup> | 80.36 | 81.33 | **84.76** | +7.50 |
| Mixer-B/16<br>→ ResNet-18<br><sub>(87.29 / 74.01)</sub> | 77.79 | 78.67 | 76.36 | 79.39 | 80.05<sup>\*</sup> | 80.69 | 81.53 | **83.95** | +6.16 |
| Swin-T<br>→ MobileNetV2<br><sub>(89.26 / 73.68)</sub> | 74.68 | 71.07 | 72.89 | 80.98 | 80.23<sup>\*</sup> | 81.64 | 82.19 | **85.38** | +10.70 |
| ViT-S<br>→ MobileNetV2<br><sub>(92.04 / 73.68)</sub> | 72.77 | 69.80 | 72.54 | 78.45 | 78.41<sup>\*</sup> | 79.21 | 80.81 | **84.66** | +11.89 |
| Mixer-B/16<br>→ MobileNetV2<br><sub>(87.29 / 73.68)</sub> | 73.33 | 70.20 | 73.26 | 78.78 | 79.89<sup>\*</sup> | 80.64 | 81.09 | **84.20** | +10.87 |
| ***ViT-based students*** | | | | | | | | | |
| ConvNeXt-T<br>→ DeiT-T<br><sub>(88.41 / 68.00)</sub> | 72.99 | 74.60 | 73.55 | 75.76 | 76.32<sup>\*</sup> | 77.46 | -- | **82.50** | +9.51 |
| Mixer-B/16<br>→ DeiT-T<br><sub>(87.29 / 68.00)</sub> | 71.36 | 73.44 | 71.67 | 73.90 | 74.83<sup>\*</sup> | 75.31 | -- | **81.46** | +10.10 |
| ConvNeXt-T<br>→ Swin-P<br><sub>(88.41 / 72.63)</sub> | 76.44 | 76.80 | 76.41 | 78.32 | 79.18<sup>\*</sup> | 80.71 | -- | **82.46** | +6.02 |
| Mixer-B/16<br>→ Swin-P<br><sub>(87.29 / 72.63)</sub> | 75.93 | 76.39 | 75.85 | 78.93 | 78.97<sup>\*</sup> | 80.52 | -- | **82.84** | +6.91 |
| ***MLP-based students*** | | | | | | | | | |
| ConvNeXt-T<br>→ ResMLP-S12<br><sub>(88.41 / 66.56)</sub> | 72.25 | 73.22 | 71.93 | 81.22 | 81.14<sup>\*</sup> | 79.28 | -- | 81.84 | +9.59 |
| Swin-T<br>→ ResMLP-S12<br><sub>(89.26 / 66.56)</sub> | 71.89 | 72.82 | 11.05 | 80.63 | 80.22<sup>\*</sup> | 80.54 | -- | <u>82.23</u> | +10.34 |
| **Average<br>Gain** | +3.17 | +3.16 | -2.31 | +7.47 | +7.69 | +8.42 | -- | **+11.99** |  |

\* Results reproduced by LDRLD [12]. HCD is excluded from the Average Gain because results are available for only six pairs.

### ImageNet-1K (15 pairs)

**(a) vs. feature-based methods**

| Teacher<br>→ Student<br><sub>(T. / S.)</sub> | FitNet<sup>1</sup> | CC<sup>2</sup> | RKD<sup>3</sup> | CRD<sup>4</sup> | PAT<sup>5</sup> | RSD<sup>6</sup> | **HACD** |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ***CNN-based students*** | | | | | | | |
| DeiT-T<br>→ ResNet-18<br><sub>(72.19 / 69.75)</sub> | 70.44 | 69.77 | 69.47 | 69.25 | -- | **71.70** | <u>71.50</u> |
| Swin-T<br>→ ResNet-18<br><sub>(81.38 / 69.75)</sub> | 71.18 | 70.07 | 68.89 | 69.09 | 71.54 | <u>72.13</u> | **72.76** |
| Mixer-B/16<br>→ ResNet-18<br><sub>(76.62 / 69.75)</sub> | 70.78 | 70.05 | 69.46 | 68.40 | -- | 71.41 | **71.81** |
| DeiT-T<br>→ MobileNetV2<br><sub>(72.17 / 68.87)</sub> | 70.95 | 70.69 | 69.72 | 69.60 | -- | <u>72.18</u> | **72.88** |
| Swin-T<br>→ MobileNetV2<br><sub>(81.38 / 68.87)</sub> | 71.75 | 70.69 | 67.52 | 69.58 | -- | 72.36 | **73.38** |
| Mixer-B/16<br>→ MobileNetV2<br><sub>(76.62 / 68.87)</sub> | 71.59 | 70.79 | 69.86 | 68.89 | 72.22 | 71.90 | **73.13** |
| ***ViT-based students*** | | | | | | | |
| ResNet-50<br>→ DeiT-T<br><sub>(80.33 / 72.17)</sub> | 75.84 | 72.56 | 72.06 | 68.53 | -- | <u>76.17</u><sup>\*</sup> | <u>76.17</u> |
| ConvNeXt-T<br>→ DeiT-T<br><sub>(82.05 / 72.17)</sub> | 70.45 | 73.12 | 71.47 | 69.18 | 74.44 | <u>74.46</u> | **75.89** |
| Mixer-B/16<br>→ DeiT-T<br><sub>(76.55 / 72.17)</sub> | 74.38 | 72.82 | 72.24 | 68.23 | -- | 74.26 | **75.76** |
| ResNet-50<br>→ Swin-N<br><sub>(80.33 / 75.53)</sub> | 78.33 | 76.05 | 75.90 | 73.90 | -- | **78.78**<sup>\*</sup> | 78.30 |
| ConvNeXt-T<br>→ Swin-N<br><sub>(82.05 / 75.53)</sub> | 74.81 | 75.79 | 75.48 | 74.15 | -- | <u>77.70</u> | **77.97** |
| Mixer-B/16<br>→ Swin-N<br><sub>(76.55 / 75.53)</sub> | 76.17 | 75.81 | 75.52 | 73.38 | -- | 76.55 | **77.00** |
| ***MLP-based students*** | | | | | | | |
| ResNet-50<br>→ ResMLP-S12<br><sub>(80.33 / 76.65)</sub> | 78.13 | 76.21 | 75.45 | 73.23 | -- | 78.32<sup>\*</sup> | **78.71** |
| ConvNeXt-T<br>→ ResMLP-S12<br><sub>(82.05 / 76.65)</sub> | 74.69 | 75.79 | 75.28 | 73.57 | -- | <u>78.41</u> | **78.65** |
| Swin-T<br>→ ResMLP-S12<br><sub>(81.35 / 76.65)</sub> | 76.48 | 76.15 | 75.10 | 73.40 | 77.59 | <u>77.61</u> | **78.35** |
| **Average<br>Gain** | +1.14 | +0.50 | -0.37 | -1.77 | -- | <u>+2.34</u> | **+2.89** |

**(b) vs. logit-based methods**

| Teacher<br>→ Student<br><sub>(T. / S.)</sub> | KD<sup>7</sup> | DKD<sup>8</sup> | DIST<sup>9</sup> | OFA<sup>10</sup> | TeKAP<sup>11</sup> | HCD<sup>13</sup> | **HACD** | Δ |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ***CNN-based students*** | | | | | | | | |
| DeiT-T<br>→ ResNet-18<br><sub>(72.19 / 69.75)</sub> | 70.22 | 69.39 | 70.64 | 71.34 | -- | -- | <u>71.50</u> | +1.28 |
| Swin-T<br>→ ResNet-18<br><sub>(81.38 / 69.75)</sub> | 71.14 | 71.10 | 70.91 | 71.85 | 71.25<sup>†</sup> | 71.91 | **72.76** | +1.62 |
| Mixer-B/16<br>→ ResNet-18<br><sub>(76.62 / 69.75)</sub> | 70.89 | 69.89 | 70.66 | 71.38 | 70.95<sup>†</sup> | <u>71.66</u> | **71.81** | +0.92 |
| DeiT-T<br>→ MobileNetV2<br><sub>(72.17 / 68.87)</sub> | 70.87 | 70.14 | 71.08 | 71.39 | 70.23<sup>†</sup> | 71.38 | **72.88** | +2.01 |
| Swin-T<br>→ MobileNetV2<br><sub>(81.38 / 68.87)</sub> | 72.05 | 71.71 | 71.76 | 72.32 | 72.50<sup>†</sup> | <u>72.72</u> | **73.38** | +1.33 |
| Mixer-B/16<br>→ MobileNetV2<br><sub>(76.62 / 68.87)</sub> | 71.92 | 70.93 | 71.74 | 72.12 | 72.01<sup>†</sup> | <u>72.32</u> | **73.13** | +1.21 |
| ***ViT-based students*** | | | | | | | | |
| ResNet-50<br>→ DeiT-T<br><sub>(80.33 / 72.17)</sub> | 75.10 | 75.60<sup>\*</sup> | 75.13<sup>\*</sup> | **76.55**<sup>\*</sup> | -- | -- | <u>76.17</u> | +1.07 |
| ConvNeXt-T<br>→ DeiT-T<br><sub>(82.05 / 72.17)</sub> | 74.00 | 73.95 | 74.07 | 74.41 | -- | -- | **75.89** | +1.89 |
| Mixer-B/16<br>→ DeiT-T<br><sub>(76.55 / 72.17)</sub> | 74.16 | 72.82 | 74.22 | <u>74.46</u> | -- | -- | **75.76** | +1.60 |
| ResNet-50<br>→ Swin-N<br><sub>(80.33 / 75.53)</sub> | 77.58 | 78.23<sup>\*</sup> | 77.95<sup>\*</sup> | <u>78.64</u><sup>\*</sup> | -- | -- | 78.30 | +0.72 |
| ConvNeXt-T<br>→ Swin-N<br><sub>(82.05 / 75.53)</sub> | 77.15 | 77.00 | 77.25 | 77.50 | -- | -- | **77.97** | +0.82 |
| Mixer-B/16<br>→ Swin-N<br><sub>(76.55 / 75.53)</sub> | 76.26 | 75.03 | 76.54 | <u>76.63</u> | -- | -- | **77.00** | +0.74 |
| ***MLP-based students*** | | | | | | | | |
| ResNet-50<br>→ ResMLP-S12<br><sub>(80.33 / 76.65)</sub> | 77.41 | 78.23<sup>\*</sup> | 77.71<sup>\*</sup> | <u>78.53</u><sup>\*</sup> | -- | -- | **78.71** | +1.30 |
| ConvNeXt-T<br>→ ResMLP-S12<br><sub>(82.05 / 76.65)</sub> | 76.84 | 77.23 | 77.24 | 77.53 | -- | -- | **78.65** | +1.81 |
| Swin-T<br>→ ResMLP-S12<br><sub>(81.35 / 76.65)</sub> | 76.67 | 76.99 | 77.25 | 77.31 | -- | -- | **78.35** | +1.68 |
| **Average<br>Gain** | +1.56 | +1.29 | +1.68 | +2.20 | -- | -- | **+2.89** |  |

\* Results of the method combined with FitNet, following OFA [10]. † Results taken from HCD [13]. Average Gain is reported only for methods with results on all 15 pairs.

## Model Zoo

Distilled student checkpoints are hosted on [GitHub Releases](https://github.com/BTAEYANG/HACD-Heterogeneous-Asymmetric-Contrastive-Distillation/releases).

### CIFAR-100 ([release](https://github.com/BTAEYANG/HACD-Heterogeneous-Asymmetric-Contrastive-Distillation/releases/tag/cifar-100-checkpoints))

| Teacher → Student | Top-1 (%) | Size | Checkpoint |
|:--|:-:|:-:|:-:|
| Swin-T → ResNet-18 | 85.00 | 85.8 MB | [download](https://github.com/BTAEYANG/HACD-Heterogeneous-Asymmetric-Contrastive-Distillation/releases/download/cifar-100-checkpoints/cifar100-hacd-swin_t-resnet18.pth.tar) |
| ViT-S → ResNet-18 | 84.76 | 85.8 MB | [download](https://github.com/BTAEYANG/HACD-Heterogeneous-Asymmetric-Contrastive-Distillation/releases/download/cifar-100-checkpoints/cifar100-hacd-vit_s-resnet18.pth.tar) |
| Mixer-B/16 → ResNet-18 | 83.95 | 85.8 MB | [download](https://github.com/BTAEYANG/HACD-Heterogeneous-Asymmetric-Contrastive-Distillation/releases/download/cifar-100-checkpoints/cifar100-hacd-mixer_b16-resnet18.pth.tar) |
| Swin-T → MobileNetV2 | 85.38 | 18.2 MB | [download](https://github.com/BTAEYANG/HACD-Heterogeneous-Asymmetric-Contrastive-Distillation/releases/download/cifar-100-checkpoints/cifar100-hacd-swin_t-mobilenetv2.pth.tar) |
| ViT-S → MobileNetV2 | 84.66 | 18.2 MB | [download](https://github.com/BTAEYANG/HACD-Heterogeneous-Asymmetric-Contrastive-Distillation/releases/download/cifar-100-checkpoints/cifar100-hacd-vit_s-mobilenetv2.pth.tar) |
| Mixer-B/16 → MobileNetV2 | 84.20 | 18.2 MB | [download](https://github.com/BTAEYANG/HACD-Heterogeneous-Asymmetric-Contrastive-Distillation/releases/download/cifar-100-checkpoints/cifar100-hacd-mixer_b16-mobilenetv2.pth.tar) |
| ConvNeXt-T → DeiT-T | 82.50 | 63.6 MB | [download](https://github.com/BTAEYANG/HACD-Heterogeneous-Asymmetric-Contrastive-Distillation/releases/download/cifar-100-checkpoints/cifar100-hacd-convnext_t-deit_t.pth.tar) |
| Mixer-B/16 → DeiT-T | 81.46 | 63.6 MB | [download](https://github.com/BTAEYANG/HACD-Heterogeneous-Asymmetric-Contrastive-Distillation/releases/download/cifar-100-checkpoints/cifar100-hacd-mixer_b16-deit_t.pth.tar) |
| ConvNeXt-T → Swin-P | 82.46 | 60.1 MB | [download](https://github.com/BTAEYANG/HACD-Heterogeneous-Asymmetric-Contrastive-Distillation/releases/download/cifar-100-checkpoints/cifar100-hacd-convnext_t-swin_p.pth.tar) |
| Mixer-B/16 → Swin-P | 82.84 | 60.1 MB | [download](https://github.com/BTAEYANG/HACD-Heterogeneous-Asymmetric-Contrastive-Distillation/releases/download/cifar-100-checkpoints/cifar100-hacd-mixer_b16-swin_p.pth.tar) |
| ConvNeXt-T → ResMLP-S12 | 81.84 | 172 MB | [download](https://github.com/BTAEYANG/HACD-Heterogeneous-Asymmetric-Contrastive-Distillation/releases/download/cifar-100-checkpoints/cifar100-hacd-convnext_t-resmlp_s12.pth.tar) |
| Swin-T → ResMLP-S12 | 82.23 | 172 MB | [download](https://github.com/BTAEYANG/HACD-Heterogeneous-Asymmetric-Contrastive-Distillation/releases/download/cifar-100-checkpoints/cifar100-hacd-swin_t-resmlp_s12.pth.tar) |

### ImageNet-1K

Checkpoints for the 15 ImageNet-1K pairs: coming soon.

## Getting Started

- **Evaluation:** Coming soon.
- **Training:** Coming soon. The training code will be released upon publication.

## Roadmap

- [x] CIFAR-100 student checkpoints (12 pairs)
- [ ] ImageNet-1K student checkpoints (15 pairs)
- [ ] Training logs
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
