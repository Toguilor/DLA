# Description
This notebook constitutes the first laboratory of the "Deep Learning Applications" course. The objective is to explore deep neural network architectures, particularly Convolutional Neural Networks (CNNs), and reproduce the fundamental results presented in the seminal paper "Deep Residual Learning for Image Recognition" (He et al., 2016).

# Objectives
- Understand and implement simple CNN architectures
- Experiment with Multi-Layer Perceptrons (MLPs) on the MNIST dataset
- Empirically verify that deeper networks do not necessarily guarantee reduced training loss
- Develop a robust and modular training pipeline
- Utilize experiment tracking tools (TensorBoard, Weights & Biases)

# Structure
## Exercise 1: Warming Up
Goal: Reproduce ResNet paper results at a small scale with an MLP on MNIST.
Subsections:
1. Data Preparation: Loading and preprocessing the MNIST dataset
2. Boilerplate Code: Basic training and evaluation functions
3. MLP Implementation: Parametric multi-layer perceptron architecture
4. Minimal Training Pipeline: Basic example to improve upon

## Exercise 1.1: Baseline MLP
- Implementation of a simple MLP for 10-digit MNIST classification
- Custom training pipeline with metric tracking
- Integration with Weights & Biases for monitoring
- Early stopping mechanisms and overfitting detection

