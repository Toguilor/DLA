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

Implementation Details:
Provided example uses a 3-layer MLP: 28×28 → 256 → 128 → 10
Includes ReLU activations between layers
Uses Adam optimizer with learning rate 0.001
Implements CrossEntropyLoss for multi-class classification
Features early stopping and overfitting detection mechanisms

The training converged in fewer than the maximum allowed epochs thanks to early stopping.
On the test set, the model achieved:
- Test Loss: 0.1053
- Test Accuracy: 97.39%
- Test Precision (weighted): 97.43%

<img width="1320" height="699" alt="image" src="https://github.com/user-attachments/assets/d0592a3e-da69-4fce-8f5e-fc27df8d7daf" />

## Exercise 1.2: Adding Residual Connections
Coming Soon - Expected Tasks:
1. Systematically increase network depth while keeping width constant
2. Compare performance of shallow vs. deep MLPs
3. Analyze gradient flow and potential vanishing/exploding gradient issues
4. Implement solutions like better weight initialization or skip connections

### Comparison of MLP and ResidualMLP
All models were trained on the MNIST dataset with the following parameters kept constant: learning rate = 1e-3, Adam optimiser.
Residual connections provided faster convergence (training stopped at epoch 15 vs 20 for the plain MLP) and slightly better performance on the test set.

<img width="1320" height="699" alt="image" src="https://github.com/user-attachments/assets/801325f3-a950-411f-b1ef-616dd8d93d85" />

## Exercise 1.3: Rinse and Repeat (but with a CNN)
Coming Soon - Expected Tasks:
- Implement residual blocks for MLPs
- Compare plain vs. residual networks of same depth
- Measure improvement in training dynamics

Two model variants were implemented: a Simple CNN and a Residual CNN. The Simple CNN applies successive convolutional layers with ReLU activation, doubling the number of channels at each step and applying max pooling every two layers. After feature extraction, a fully connected layer reduces the feature map to 128 neurons, followed by dropout and the final classification layer. The Residual CNN starts with a single convolution + batch normalization + ReLU block, followed by a configurable number of ResNet BasicBlocks (as in the original ResNet architecture), which implement skip connections over sequences of 3×3 convolutions.

<p align="center">
  <img alt="Screenshot 2025-12-15 174925" src="https://github.com/user-attachments/assets/458a3b80-7808-446c-8af2-d1986390a6f4" width="50%" />
  <img alt="Screenshot 2025-12-15 174938" src="https://github.com/user-attachments/assets/7377f174-7e87-4e34-821c-3eefe7ba5c5b" width="50%" />
</p>


