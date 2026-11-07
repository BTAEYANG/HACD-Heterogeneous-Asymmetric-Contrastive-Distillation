# HACD: Heterogeneous Asymmetric Contrastive Distillation

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg" alt="arXiv"></a>
  <a href="https://doi.org/10.5281/zenodo.XXXXXXX"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.XXXXXXX-blue.svg" alt="DOI"></a>
  <img src="https://img.shields.io/badge/PyTorch-%E2%89%A52.0-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/checkpoints-CC%20BY--NC%204.0-lightgrey.svg" alt="License">
</p>

Official PyTorch implementation of **HACD (Heterogeneous Asymmetric Contrastive Distillation)**, a logit-only knowledge distillation method for CNN, Transformer, and MLP teacher–student pairs.

> **Status.** Checkpoints and evaluation scripts are available now. The full training code will be released upon publication.

<p align="center">
  <img src="assets/hacd_overview.png" width="95%" alt="HACD overview">
</p>

## News

- **[2026-11]** Checkpoints and evaluation code released. Paper submitted; preprint on [arXiv](https://arxiv.org/abs/XXXX.XXXXX).

## Overview

Distilling across architectural families, e.g., from a Transformer teacher to a CNN student, is hard because the two networks learn different internal representations. Logits share a label space, but their values still depend on the architecture: Transformer and MLP teachers produce sharper outputs whose non-target classes are less semantically organized than those of CNN teachers. Matching these values sample by sample asks the student to imitate how another architecture expresses its predictions.

HACD instead transfers **how the teacher arranges samples relative to one another**:

- **Shared, comparable space.** Teacher and student logits are mapped to centered, temperature-softened probability directions on a unit hypersphere. Centering removes architecture-dependent logit offsets, and the temperature controls how much of the teacher's sharpness enters the representation while keeping informative non-target classes.
- **Relational transfer through contrast.** Each student output must identify its own teacher target among all teacher targets in the mini-batch (bidirectional instance-level contrast).
- **Asymmetric views.** Teacher targets are computed on a moderately augmented view and supervise both a moderate and a strong student view, so the student gains from strong augmentation while its supervision stays reliable.
- **Lightweight.** No projectors, alignment modules, or extra learnable parameters, and a single teacher forward pass per sample.

## Main Results

HACD achieves the best or second-best top-1 accuracy on **25 of 27** heterogeneous teacher–student pairs. On CIFAR-100 it uses **no ground-truth labels** during distillation.

### CIFAR-100 (12 pairs, label-free distillation)

| Teacher | Student | Teacher Acc. | Student (scratch) | **HACD** | Gain |
|:--|:--|:-:|:-:|:-:|:-:|
| Swin-T | ResNet-18 | 89.26 | 74.01 | **85.00** | +10.99 |
| ViT-S | ResNet-18 | 92.04 | 74.01 | **84.76** | +10.75 |
| Mixer-B/16 | ResNet-18 | 87.29 | 74.01 | **83.95** | +9.94 |
| Swin-T | MobileNetV2 | 89.26 | 73.68 | **85.38** | +11.70 |
| ViT-S | MobileNetV2 | 92.04 | 73.68 | **84.66** | +10.98 |
| Mixer-B/16 | MobileNetV2 | 87.29 | 73.68 | **84.20** | +10.52 |
| ConvNeXt-T | DeiT-T | 88.41 | 68.00 | **82.50** | +14.50 |
| Mixer-B/16 | DeiT-T | 87.29 | 68.00 | **81.46** | +13.46 |
| ConvNeXt-T | Swin-P | 88.41 | 72.63 | **82.46** | +9.83 |
| Mixer-B/16 | Swin-P | 87.29 | 72.63 | **82.84** | +10.21 |
| ConvNeXt-T | ResMLP-S12 | 88.41 | 66.56 | **81.84** | +15.28 |
| Swin-T | ResMLP-S12 | 89.26 | 66.56 | **82.23** | +15.67 |

### ImageNet-1K (15 pairs)

| Teacher | Student | Teacher Acc. | Student (scratch) | **HACD** | Gain |
|:--|:--|:-:|:-:|:-:|:-:|
| DeiT-T | ResNet-18 | 72.19 | 69.75 | **71.50** | +1.75 |
| Swin-T | ResNet-18 | 81.38 | 69.75 | **72.76** | +3.01 |
| Mixer-B/16 | ResNet-18 | 76.62 | 69.75 | **71.81** | +2.06 |
| DeiT-T | MobileNetV2 | 72.17 | 68.87 | **72.88** | +4.01 |
| Swin-T | MobileNetV2 | 81.38 | 68.87 | **73.38** | +4.51 |
| Mixer-B/16 | MobileNetV2 | 76.62 | 68.87 | **73.13** | +4.26 |
| ResNet-50 | DeiT-T | 80.33 | 72.17 | **76.17** | +4.00 |
| ConvNeXt-T | DeiT-T | 82.05 | 72.17 | **75.89** | +3.72 |
| Mixer-B/16 | DeiT-T | 76.55 | 72.17 | **75.76** | +3.59 |
| ResNet-50 | Swin-N | 80.33 | 75.53 | **78.30** | +2.77 |
| ConvNeXt-T | Swin-N | 82.05 | 75.53 | **77.97** | +2.44 |
| Mixer-B/16 | Swin-N | 76.55 | 75.53 | **77.00** | +1.47 |
| ResNet-50 | ResMLP-S12 | 80.33 | 76.65 | **78.71** | +2.06 |
| ConvNeXt-T | ResMLP-S12 | 82.05 | 76.65 | **78.65** | +2.00 |
| Swin-T | ResMLP-S12 | 81.35 | 76.65 | **78.35** | +1.70 |

Gain = HACD − student trained from scratch (percentage points). Full comparisons with feature-based, logit-based, and heterogeneous KD methods are in the paper.

## Model Zoo

Each checkpoint contains the distilled student weights. Training logs are provided alongside.

| Dataset | Teacher → Student | Top-1 | Checkpoint | Log |
|:--|:--|:-:|:-:|:-:|
| CIFAR-100 | Swin-T → ResNet-18 | 85.00 | [download](#) | [log](#) |
| CIFAR-100 | Swin-T → MobileNetV2 | 85.38 | [download](#) | [log](#) |
| CIFAR-100 | ConvNeXt-T → DeiT-T | 82.50 | [download](#) | [log](#) |
| ImageNet-1K | Swin-T → MobileNetV2 | 73.38 | [download](#) | [log](#) |
| ImageNet-1K | ConvNeXt-T → DeiT-T | 75.89 | [download](#) | [log](#) |
| ... | ... | ... | ... | ... |

<!-- TODO: fill in the released pairs and links (GitHub Releases / Hugging Face / Zenodo). -->

## Getting Started

### Installation

```bash
git clone https://github.com/<user>/HACD-Heterogeneous-Asymmetric-Contrastive-Distillation.git
cd HACD-Heterogeneous-Asymmetric-Contrastive-Distillation

conda create -n hacd python=3.10 -y
conda activate hacd
pip install -r requirements.txt
```

### Data Preparation

Download [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) and [ImageNet-1K](https://www.image-net.org/) and organize them as:

```
data/
├── cifar-100-python/
└── imagenet/
    ├── train/
    └── val/
```

Following the OFA protocol, CIFAR-100 images are resized to 224×224 for all architectures.

### Evaluation

Download a checkpoint from the [Model Zoo](#model-zoo) and run:

```bash
# CIFAR-100
python eval.py --dataset cifar100 --data-path data/ \
    --model mobilenetv2 --checkpoint checkpoints/cifar100_swint_mobilenetv2.pth

# ImageNet-1K
python eval.py --dataset imagenet --data-path data/imagenet \
    --model deit_tiny --checkpoint checkpoints/in1k_convnextt_deitt.pth
```

<!-- TODO: align script name, arguments, and model identifiers with the released code. -->

### Training

The training code will be released upon publication of the paper.

## Roadmap

- [x] Distilled student checkpoints
- [x] Evaluation scripts and training logs
- [ ] Training code and configurations for all 27 pairs
- [ ] Teacher checkpoints

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

## License

- **Checkpoints and training logs:** [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Free for non-commercial research with attribution.
- **Code:** will be released under an open-source license together with the training code.

The datasets and the pretrained teacher models remain subject to their original licenses and terms of use.

## Acknowledgments

Our experimental protocol follows [OFA-KD](https://github.com/Hao840/OFAKD). We thank the authors for making their code available.

## Contact

Qi Wang — qi.wang.7c@stu.hosei.ac.jp
