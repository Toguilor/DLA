# Lab 1 — Residual Networks: MLPs, CNNs and Transfer Learning

> 🇫🇷 A French version of this document is available in [README_FR.md](README_FR.md).

This repository contains my solution to the first laboratory of the **Deep Learning Applications** course. The goal of the lab is to reproduce, on a small scale, the main result of the ResNet paper:

> Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. [*Deep Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385). CVPR 2016.

The key message of this paper is that **making a network deeper does not guarantee better results**. Beyond a certain depth, "plain" networks become harder to optimize, and even their *training* error increases. **Residual (skip) connections** solve this problem. In this lab I:

1. build a training pipeline and a baseline MLP on MNIST;
2. compare MLPs with and without residual connections at increasing depths;
3. repeat the experiment with CNNs on CIFAR-10, with a fair plain vs residual comparison from 8 to 56 layers, and analyze the gradients;
4. reuse the best residual CNN for **transfer learning** to CIFAR-100 (feature extraction + fine-tuning).

---

## Table of contents

- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Background: plain networks, degradation and residual connections](#background-plain-networks-degradation-and-residual-connections)
- [Exercise 1.1 — Baseline MLP and training pipeline](#exercise-11--baseline-mlp-and-training-pipeline)
- [Exercise 1.2 — MLP with residual connections](#exercise-12--mlp-with-residual-connections)
- [Exercise 1.3 — Plain vs residual CNNs on CIFAR-10](#exercise-13--plain-vs-residual-cnns-on-cifar-10)
- [Exercise 2.1 — Transfer learning to CIFAR-100](#exercise-21--transfer-learning-to-cifar-100)
- [Conclusion](#conclusion)

---

## Repository structure

```
.
├── Lab-CNNs.ipynb          # Notebook with all the code, experiments and outputs
├── README.md               # This file (English)
├── README_FR.md            # French version
└── images/                 # Plots used in this README
    ├── ex1_1_mlp_baseline.png
    ├── ex1_2_mlp_vs_resmlp.png
    ├── ex1_3_training_curves.png
    ├── ex1_3_results_vs_depth.png
    ├── ex1_3_gradients.png
    ├── ex2_1_finetuning_curves.png
    └── ex2_1_test_accuracy.png
```

| Section | Content |
|---|---|
| Exercise 1.1 | Generic training pipeline (train/val loss and accuracy, early stopping, checkpointing, Weights & Biases logging) and a baseline MLP on MNIST |
| Exercise 1.2 | `ResidualMLP`, comparison with the plain MLP at depths 2, 4, 8 and 16, gradient norms |
| Exercise 1.3 | `CifarNet` (ResNet-style CNN for CIFAR-10) with a `residual` flag; 8 models (8, 20, 32, 56 layers × plain/residual); training curves; gradient analysis on a fixed batch |
| Exercise 2.1 | ResNet-20 pre-trained on CIFAR-10 used on CIFAR-100: KNN and SVM baselines on frozen features, then 5 fine-tuning strategies |

Exercises 2.2 (knowledge distillation) and 2.3 (class activation maps) were not addressed (at least one of the three was required).

## Requirements

```bash
pip install torch torchvision numpy matplotlib scikit-learn tqdm wandb
```

A GPU is strongly recommended. Training the 8 CNNs of Exercise 1.3 takes about one hour on a mid-range GPU.

---

## Background: plain networks, degradation and residual connections

A **plain network** stacks layers one after the other: each block computes $y = F(x)$. A **residual network** adds a *shortcut* that skips the block: $y = F(x) + x$.

**The degradation problem.** In theory, a deeper network should never be worse than a shallower one: the extra layers could simply learn the identity function and copy their input. In practice, He et al. observed that a 56-layer plain network has a **higher training error** than a 20-layer one. This is **not overfitting**, because it is the error on the *training* set that increases. It is an **optimization problem**: gradient descent fails to find good weights for very deep plain networks.

**Why residual connections help.**
- With $y = F(x) + x$, "doing nothing" becomes easy: it is enough to learn $F(x) \approx 0$.
- During backpropagation, the gradient of each block contains a direct term: $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}\left(1 + \frac{\partial F}{\partial x}\right)$. The error signal can flow from the output to the first layers through the shortcuts, without being multiplied by every intermediate layer.

---

## Exercise 1.1 — Baseline MLP and training pipeline

### Objective

Implement a simple MLP to classify MNIST digits, write my **own training pipeline**, and train the model to convergence while monitoring the loss and accuracy on the training and validation sets at every epoch.

### Method

**Data.** MNIST (60 000 training images, 10 000 test images, 28×28 grayscale), normalized with the standard MNIST mean and std. 5 000 training images are held out as a validation set.

**Pipeline.** I wrote reusable functions, which are used throughout the lab:
- `train_one_epoch(...)` and `evaluate_model(...)`: return the average loss and accuracy on a data loader;
- `train_model(...)`: full training loop that logs train/val loss and accuracy at every epoch (also to **Weights & Biases**), **saves the best model** (lowest validation loss), and implements **early stopping**: training stops if the validation loss keeps increasing, or stops improving, for several epochs;
- `plot_validation_curves(...)`: train and validation curves on the same plot.

**Model.** MLP 784 → 256 → 128 → 10, ReLU activations, dropout 0.2, Adam (lr = 1e-3), cross-entropy loss, up to 50 epochs.

### Results

<!-- Plot: train/validation loss and accuracy of the baseline MLP -->
![Baseline MLP on MNIST](images/ex1_1_mlp_baseline.png)

| Metric | Value |
|---|---|
| Epochs (early stopping) | 19 |
| Test loss | 0.0601 |
| **Test accuracy** | **98.19 %** |

### Analysis

The MLP converges quickly: the validation accuracy is already above 96 % after 2-3 epochs. Then the training loss keeps decreasing while the validation loss stagnates and starts to oscillate, which is the beginning of overfitting. Early stopping halts training at that point, and the best checkpoint (lowest validation loss) is reloaded for the test evaluation. This model, with **98.19 %** test accuracy, is the reference for the next experiments.

---

## Exercise 1.2 — MLP with residual connections

### Objective

Implement an MLP made of **residual blocks**, and compare it with the plain MLP for a range of depths. The goal is to verify that deep networks *with* residual connections are easier to train than networks of the same depth *without* them.

### Method

- **Residual block**: Linear → ReLU → Dropout → Linear, then the block input is added to the output, followed by ReLU: $y = \text{ReLU}(F(x) + x)$. Input and output have the same size so that they can be added.
- **ResidualMLP**: input projection 784 → 128, then `depth` residual blocks of width 128, then a linear classifier 128 → 10.
- **Plain MLP**: `depth` hidden layers whose width decreases geometrically from 512 to 32 (e.g. depth 4: 784 → 512 → 203 → 80 → 32 → 10).
- Depths tested: **2, 4, 8 and 16**; same optimizer (Adam, lr = 1e-3), dropout 0.2, up to 30 epochs with early stopping.
- **Gradient analysis**: L2 norm of the gradient of each weight matrix, read after training.

### Results

<!-- Plot: validation loss, validation accuracy, gradient norms and summary for each depth -->
![MLP vs ResidualMLP](images/ex1_2_mlp_vs_resmlp.png)

| Depth | MLP test acc | MLP epochs | ResidualMLP test acc | ResidualMLP epochs |
|---|---|---|---|---|
| 2 | 98.04 % | 21 | 97.86 % | 15 |
| 4 | 98.36 % | 26 | 97.55 % | 25 |
| 8 | 98.08 % | 22 | 97.88 % | 16 |
| 16 | **81.90 %** | 10 | **97.74 %** | 23 |

### Analysis

- **Up to 8 layers**, both architectures reach ~98 %. MNIST is an easy problem and shallow networks train without difficulty.
- **At 16 layers**, the plain MLP becomes much harder to optimize. Its validation accuracy starts at ~47 %, oscillates strongly, and the final test accuracy drops to **81.9 %**. The ResidualMLP of the same depth reaches ~95 % from the first epoch and ends at **97.7 %**. This is the behavior described in the ResNet paper: depth alone does not improve performance, and residual connections make deep networks trainable.
- **Gradients.** In the 16-layer plain MLP, gradient norms are large at both ends of the network (≈ 0.9 for the first layer, ≈ 1.0 for the last one) but much smaller in the middle (≈ 0.12). The learning signal is very unevenly distributed across the layers. In the ResidualMLP, gradient norms decrease smoothly from block to block, without any break, because each shortcut provides a direct path for the gradient.

**Limitations of this experiment** (corrected in Exercise 1.3):
1. The two architectures are not strictly identical: the plain MLP has decreasing widths (512 → 32), while the ResidualMLP has a constant width (128) and 2 linear layers per block.
2. The 16-layer plain MLP was stopped by early stopping after 10 epochs while its validation accuracy was still increasing, which amplifies the gap.
3. Gradients were read after training, from the last training batch, and not on the same batch at the same moment for both models.

---

## Exercise 1.3 — Plain vs residual CNNs on CIFAR-10

### Objective

Repeat the verification with convolutional networks on CIFAR-10. The goal is to show that **deeper plain CNNs do not always work better**, while **even deeper residual CNNs** do.

### Method

**Data.** CIFAR-10 (50 000 training images, 10 000 test images, 32×32 color, 10 classes). 5 000 training images are used for validation. Training images are augmented (random crop with 4-pixel padding + horizontal flip). **Validation and test images are not augmented**: the validation split uses a second copy of the dataset with the evaluation transform.

**Architecture: `CifarNet(n, residual)`**, following Section 4.2 of the ResNet paper:
- a 3×3 convolution with 16 channels, then **3 stages** of *n* blocks each, with 16, 32 and 64 channels on 32×32, 16×16 and 8×8 feature maps;
- downsampling is done by a stride-2 convolution in the first block of stages 2 and 3, without max-pooling;
- global average pooling, then a linear layer with 10 outputs;
- each block has 2 convolutions 3×3 with BatchNorm, so the total depth is **6n + 2** layers. I tested *n* = 1, 3, 5, 9, i.e. **8, 20, 32 and 56 layers**.

**The only difference between plain and residual is the `residual` flag of the block.** Both versions have the same layers, the same widths and the same initialization (same seed, Kaiming init). When the shape changes between the input and the output of a block, the shortcut is a 1×1 convolution with BatchNorm (option B of the paper). This adds only ~2 750 parameters to the residual model:

| Depth | Plain (params) | Residual (params) |
|---|---|---|
| 8 | 75 290 | 78 042 |
| 20 | 269 722 | 272 474 |
| 32 | 464 154 | 466 906 |
| 56 | 853 018 | 855 770 |

The block is adapted from torchvision's `BasicBlock`, modified so that the skip connection can be disabled.

**Training.** Identical for all 8 models: SGD with Nesterov momentum 0.9, initial lr 0.1, weight decay 5e-4, **cosine** learning-rate schedule, batch 128, **30 epochs, without early stopping**, so that training losses are compared at equal budget. The checkpoint with the best validation accuracy is used for the test evaluation.

**Gradient analysis.** For each depth, the L2 norm of the gradient of every 3×3 convolution is computed on **the same training batch**, for the plain and the residual model, at two moments: **at initialization** (same starting weights) and **after training** (best checkpoints).

### Results

<!-- Plot: training loss and validation error per depth, plain (left) vs residual (right) -->
![Training curves — plain vs residual](images/ex1_3_training_curves.png)

<!-- Plot: final training loss and test accuracy as a function of depth -->
![Results vs depth](images/ex1_3_results_vs_depth.png)

| Layers | Plain: final train loss | Plain: test acc | Residual: final train loss | Residual: test acc |
|---|---|---|---|---|
| 8 | 0.377 | 84.84 % | 0.375 | 84.72 % |
| 20 | **0.208** | **88.02 %** | 0.147 | 89.93 % |
| 32 | 0.250 | 87.22 % | 0.108 | 90.83 % |
| 56 | **0.592** | **78.60 %** | **0.080** | **91.24 %** |

<!-- Plot: gradient norm per convolution layer on the same batch, at initialization and after training -->
![Gradient norms](images/ex1_3_gradients.png)

| Model | Ratio ‖∇W‖ first layer / last layer at initialization |
|---|---|
| plain-8 / res-8 | 1.75 / 2.44 |
| plain-20 / res-20 | 2.9 / 3.05 |
| plain-32 / res-32 | 37.2 / 5.86 |
| plain-56 / res-56 | **701** / **9.6** |

### Analysis

- **Degradation of plain networks.** Beyond 20 layers, the final training loss of plain CNNs *increases* with depth: 0.208 (20 layers) → 0.250 (32) → **0.592 (56)**. The 56-layer plain network has a training loss almost three times higher than the 20-layer one, and its curve stays above all the others during the whole training. Since it is the *training* error that is higher, this is not overfitting but an **optimization** problem. This reproduces the main result of the ResNet paper (its Figure 6).
- **Residual networks benefit from depth.** With exactly the same layers, the training loss decreases at every increase in depth (0.375 → 0.147 → 0.108 → 0.080) and the test accuracy increases (84.7 → 89.9 → 90.8 → **91.2 %**). The residual 56-layer network is the best model, while the plain 56-layer network is the worst (**+12.6 points** in favor of the residual version).
- **At 8 layers**, both versions are identical (84.8 % vs 84.7 %): the network is too shallow for the skip connections to matter.
- **Gradients.** At initialization, the gradient of the first layer of plain-56 is about **700 times larger** than that of its last layer (≈ 200 vs ≈ 0.3), and the ratio grows quickly with depth (1.75 → 2.9 → 37 → 701). In deep plain networks, gradients **explode** as they propagate back towards the input: the first layers receive updates that are far too large compared to the last ones, which makes optimization very difficult. For residual networks the ratio stays moderate (9.6 for res-56), because each shortcut adds a direct term to the gradient. As the ResNet paper points out, with BatchNorm the problem is not simply *vanishing* gradients, but a badly conditioned propagation through many successive layers. After training, gradient norms are of the same order of magnitude for all models.
- **Limitations.** Only 30 epochs (the paper uses ~160) and a single seed per configuration: differences smaller than ~0.5 point (e.g. at 8 layers) are not significant.

---

## Exercise 2.1 — Transfer learning to CIFAR-100

### Objective

Use a residual CNN trained on CIFAR-10 for a new task, CIFAR-100 (100 classes):
1. as a **feature extractor**, with a classical classifier, to establish a stable baseline;
2. with **fine-tuning**, testing several strategies: unfreezing different parts of the network and different optimizers.

### Method

**Pre-trained model:** the residual 20-layer `CifarNet` (res-20) from Exercise 1.3 (89.9 % test accuracy on CIFAR-10).

**Data.** CIFAR-100 (50 000 training images, 10 000 test images, 100 classes), normalized with CIFAR-100 statistics; 5 000 training images for validation. The training split is augmented for fine-tuning; validation, test and feature extraction use non-augmented images.

**1. Feature extraction + classical classifiers.** The 64-dimensional activations after global average pooling (the input of the classifier) are extracted for all images, then standardized (`StandardScaler` fitted on train). Two classifiers are trained, with hyperparameters chosen on the validation split:
- **KNN** (cosine distance), K ∈ {1, 5, 10, 20, 50};
- **Linear SVM**, C ∈ {0.001, 0.01, 0.1, 1, 10}.

**2. Fine-tuning.** The 10-class classifier is replaced by a new, randomly initialized **100-class** linear layer, and the CIFAR-10 weights are loaded for everything else. Five configurations, 30 epochs each:

| Configuration | What is trained | Optimizer | lr | Trained params |
|---|---|---|---|---|
| `fc_adam` | the new head only | Adam | 1e-3 | 6 500 |
| `last_stage_adam` | last stage (64 channels) + head | Adam | 1e-4 | 212 196 |
| `last_stage_sgd` | last stage + head | SGD (momentum 0.9) | 1e-2 | 212 196 |
| `all_adam` | the whole network | Adam | 1e-4 | 278 324 |
| `all_sgd` | the whole network | SGD (momentum 0.9) | 1e-2 | 278 324 |

**Frozen BatchNorm.** Setting `requires_grad = False` does not freeze the running statistics (mean and variance) of BatchNorm layers: they keep being updated in `train()` mode. To really freeze part of the network, the BatchNorm layers of frozen blocks are kept in `eval()` mode during training.

### Results

**Baselines on frozen features**

| Classifier | Best hyperparameter | Val acc | Test acc |
|---|---|---|---|
| KNN | K = 20 | 20.96 % | 20.87 % |
| Linear SVM | C = 1.0 | 31.10 % | **30.63 %** |

**Fine-tuning**

<!-- Plot: validation loss and accuracy of the 5 fine-tuning strategies, with the baselines -->
![Fine-tuning curves on CIFAR-100](images/ex2_1_finetuning_curves.png)

<!-- Plot: test accuracy of all methods -->
![Test accuracy on CIFAR-100](images/ex2_1_test_accuracy.png)

| Configuration | Trained params | Best val acc | Test acc |
|---|---|---|---|
| `fc_adam` | 6 500 | 25.28 % | 24.41 % |
| `last_stage_adam` | 212 196 | 50.50 % | 50.78 % |
| `last_stage_sgd` | 212 196 | 54.50 % | 54.58 % |
| `all_adam` | 278 324 | 53.70 % | 53.75 % |
| **`all_sgd`** | 278 324 | **57.80 %** | **57.91 %** |

### Analysis

- **Weak baselines.** The frozen features give only 20.9 % (KNN) and 30.6 % (SVM) on CIFAR-100. The 64-dimensional features were learned to separate 10 classes. They contain general visual information (above the 1 % chance level), but they are not rich enough to distinguish 100 fine-grained classes. The linear SVM is better than KNN, which suffers from distances in a space that was not designed for these classes.
- **Fine-tuning brings a large gain.** Fine-tuning the whole network with SGD reaches **57.9 %**, almost **twice** the SVM baseline. Adapting the convolutional features to the new classes is essential.
- **The more layers are unfrozen, the better.** Training only the head (24.4 %) is clearly not enough. It is even below the SVM, because it is also a linear classifier on the same frozen features, but trained with Adam, without feature standardization and without tuning, while C of the SVM was optimized on validation. Unfreezing the last stage (76 % of the parameters) already gives most of the gain (54.6 %). Unfreezing everything adds about 3 more points.
- **SGD beats Adam in both cases** (+3.8 points for the last stage, +4.2 points for the whole network). This is consistent with common practice for ResNet-type CNNs, where SGD with momentum often generalizes better than Adam. The learning rates were not tuned for each optimizer, though, so part of the gap may come from the choice of lr (1e-2 for SGD, 1e-4 for Adam).
- The curves of most configurations are still rising after 30 epochs: longer training would probably improve all the fine-tuning results.

---

## Conclusion

- **Depth alone does not guarantee better performance.** On MNIST (Exercise 1.2), the 16-layer plain MLP drops to 81.9 %, while the residual MLP of the same depth stays at 97.7 %. On CIFAR-10 (Exercise 1.3), with a strictly fair comparison, the training loss of plain CNNs increases beyond 20 layers, and the plain 56-layer network ends at 78.6 % test accuracy. This is the **degradation problem** of the ResNet paper, an optimization problem and not overfitting.
- **Residual connections solve this problem.** With the same layers, residual CNNs improve at every increase in depth, up to **91.2 %** for 56 layers. The gradient analysis shows why: in deep plain networks, gradients explode towards the input at initialization (ratio of 701 between the first and the last layer for 56 layers), while shortcuts keep the propagation balanced (ratio of 9.6).
- **Residual networks transfer well.** A ResNet-20 trained on CIFAR-10 can be reused on CIFAR-100. Frozen features with a linear SVM give a 30.6 % baseline, and fine-tuning reaches 57.9 %. Unfreezing more layers and using SGD gave the best results.
- **Methodology matters.** Several bugs or biases in my first versions were only found by checking the protocol: architectures that were not comparable, an augmented validation set, gradients read at the wrong moment, BatchNorm layers that were not really frozen. A fair comparison (same layers, same seed, same training budget) is what made the degradation phenomenon clearly visible.

**Limitations:** short training (30 epochs) and a single seed per configuration; Exercises 2.2 and 2.3 were not addressed.
