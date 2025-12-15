# Deep Learning Architectures – CNNs, Feature Extraction and Fine-Tuning

## Introduction

This repository contains the full implementation, experimentation, and analysis for **Lab 1 – Convolutional Neural Networks (CNNs)**. The goal of this lab is to progressively study:

1. Training CNN architectures from scratch
2. Understanding the role of residual connections
3. Evaluating learned representations using classical machine learning baselines
4. Applying transfer learning and fine-tuning strategies

All experiments are implemented in **PyTorch** and executed on the **CIFAR-10** and **CIFAR-100** datasets.

---

## Exercise 1 – Training a CNN / ResNet from Scratch (CIFAR-10)

### Objective

The first exercise aims to design and train a custom **ResNet-like CNN architecture** from scratch on CIFAR-10, without relying on pretrained models.

### Model Architecture

The implemented network consists of:

* An initial convolutional layer (3×3)
* Three residual stages (`layer1`, `layer2`, `layer3`)
* Each stage composed of multiple **Residual Blocks**
* Downsampling performed via strided convolutions
* Global Average Pooling
* A fully connected classification layer

Residual blocks follow the standard formulation:

* Two convolutional layers
* Batch Normalization
* ReLU activations
* Skip connections with projection when dimensions differ

### Training Setup

* Dataset: CIFAR-10
* Loss: Cross-Entropy Loss
* Optimizer: Adam
* Early stopping based on validation loss
* Best model checkpoint saved to disk

### Results

The model successfully converges and achieves stable performance on CIFAR-10, serving as a **reliable baseline** for subsequent experiments.

Key observations:

* Residual connections significantly stabilize training
* Global average pooling reduces overfitting

The trained model is saved and reused in later exercises.

---

## Exercise 2 – CNN as a Feature Extractor

### Objective

This exercise evaluates the **quality of the learned representations** by using the pretrained CNN as a **fixed feature extractor**.

Instead of training the classifier end-to-end, features are extracted from the network using:

```python
model.forward_features(x)
```

These features are then classified using classical machine learning algorithms.

### Feature Extraction

* Features are extracted after global average pooling
* Output dimensionality: `out_channel × 4`
* The CNN weights are frozen during this process

### Baseline Classifiers

The following classifiers are evaluated:

* **K-Nearest Neighbors (KNN)**
* **Linear SVM (LinearSVC)**
* **Kernel SVM (SVC with linear kernel)**

For these baselines:

* Features are optionally normalized using `StandardScaler`
* No gradients are propagated through the CNN

### Results and Analysis

Observed accuracies are extremely low (≈ 0.1–1%). This behavior is explained by:

* Insufficient feature separability for linear classifiers
* High intra-class variability in CIFAR images
* Sensitivity of SVM/KNN to feature scaling and dimensionality

This experiment highlights that **good CNN performance does not necessarily imply linearly separable features**.

---

## Exercise 3 – Fine-Tuning on CIFAR-100

### Objective

The goal of this exercise is to transfer the CIFAR-10 pretrained model to a **more challenging dataset (CIFAR-100)** using fine-tuning.

### Fine-Tuning Strategy

The following steps are applied:

1. Load the pretrained CIFAR-10 model
2. Replace the final classification layer (`10 → 100 classes`)
3. Freeze most of the network parameters
4. Unfreeze selected deeper layers (`layer3`)
5. Train only the unfrozen parameters

### Optimizer and Loss

* Optimizer: Adam (learning rate = 1e-4)
* Loss: Cross-Entropy Loss

Only parameters with `requires_grad=True` are passed to the optimizer.

### Results

Fine-tuning leads to:

* Faster convergence compared to training from scratch
* Improved generalization on CIFAR-100
* Better performance than classical baselines

Unfreezing deeper layers consistently outperforms training only the classifier.

---

## Comparison and Discussion

| Method                    | Dataset   | Performance             |
| ------------------------- | --------- | ----------------------- |
| CNN from scratch          | CIFAR-10  | Stable baseline         |
| KNN / SVM on CNN features | CIFAR-10  | Very low                |
| Fine-tuned CNN            | CIFAR-100 | Significant improvement |

Key conclusions:

* Residual CNNs learn hierarchical features that are not necessarily linearly separable
* Classical ML baselines are useful diagnostics but not competitive
* Fine-tuning is essential when transferring to more complex datasets

---

## Conclusion

This lab demonstrates the full lifecycle of modern CNN usage:

* Architecture design
* Training from scratch
* Representation evaluation
* Transfer learning and fine-tuning

The experiments confirm that **deep representations are most effective when trained and adapted end-to-end**, especially for complex visual tasks.

---

## Author

**Loric TONGO**

Deep Learning Architectures – Lab 1
