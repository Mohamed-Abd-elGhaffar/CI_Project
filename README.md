# Deep Learning Framework from Scratch 🧠

This repository contains a fully functional Deep Learning library built from absolute scratch using pure math and NumPy. It was developed as the final project for **CSE473s: Computational Intelligence** at **Engineering Ain Shams University**.

## 👥 Team Members

| **Name** | **Student ID** | 
| :--- | :--- | 
| **Mohamed Reda Mohamed** | 2100459 | 
| **Abdulrahman Khaled Elsayed** | 2101799 | 
| **Nour Emad Rabea Hosin** | 2101443 | 

## 🎯 Project Overview

The goal of this project was to understand the mathematical foundations of Neural Networks by implementing the entire backpropagation engine, optimizers, and layers without relying on modern frameworks (like PyTorch or TensorFlow) for the core logic.

After proving the mathematical correctness of our library, we applied it to solve classical logic problems and built a complex Denoising Autoencoder for image reconstruction and latent space feature extraction.

## ✨ Features Implemented

Our custom `lib` package includes the following from-scratch implementations:

* **Core Framework:** A `Network` class to orchestrate forward/backward passes and training loops.

* **Layers:** Fully Connected `Dense` layers and Regularization `Dropout` layers.

* **Activations:** `ReLU`, `Sigmoid`, `Tanh`, and `Softmax` (with analytical derivatives).

* **Loss Functions:** Mean Squared Error (`MSE`), Binary Cross-Entropy (`BCE`), and Categorical Cross-Entropy (`CCE`).

* **Optimizers:** Stochastic Gradient Descent (`SGD`) with Velocity/Momentum.

## 🚀 Tasks & Milestones

### 1. Mathematical Proof (Gradient Checking)

Before training any models, we implemented a numerical gradient checking algorithm. By comparing the analytical gradients (from our backpropagation calculus) against brute-force numerical approximations, we proved our library's mathematical integrity (yielding an error margin of `< 1e-10`).

### 2. The XOR Problem

We successfully trained a `2-4-1` Multi-Layer Perceptron (MLP) with Tanh and Sigmoid activations to perfectly map the non-linear XOR truth table.

### 3. Denoising Autoencoder (Fashion-MNIST)

We built an Encoder-Decoder architecture to process the Fashion-MNIST dataset.

* **Input:** Images corrupted with random Gaussian noise.

* **Bottleneck:** Compressed 784 pixels down to a 32-dimensional latent space.

* **Output:** The network successfully learned to reconstruct the original, clean clothing items by filtering out the noise.

### 4. Latent Space SVM Classification

Using the trained Encoder, we extracted the 32-dimensional latent representations of the images.

* Visualized the unsupervised clustering of clothing items using **PCA** (2D projection).

* Trained an **SVM Classifier** on these extracted features, generating a comprehensive Confusion Matrix and Classification Report.

### 5. TensorFlow / Keras Comparison

As a final benchmark, we rebuilt our XOR and Autoencoder models using `TensorFlow/Keras`. We provided a critical analysis comparing our from-scratch library against the industry standard in terms of:

1. Training Time (Python/NumPy vs. C++ optimization)

2. Final Reconstruction Loss

3. Ease of Implementation

## 📂 Project Structure

```text
├── .venv/               # Virtual environment containing project dependencies
├── lib/
│   ├── __init__.py      # Package constructor
│   ├── activations.py   # Non-linear activation functions & derivatives
│   ├── layers.py        # Dense and Dropout layer logic
│   ├── losses.py        # Error calculation formulas
│   ├── network.py       # The main neural network orchestrator
│   └── optimizer.py     # SGD with Momentum weight updates
├── notebooks/
│   └── project_demo.ipynb # Main interactive notebook containing all 5 project tasks
└── README.md            # Project documentation
```
### 🛠️ Virtual Environment Setup
This project uses a Python virtual environment to manage dependencies. To run the notebook, activate the virtual environment (.venv) from the main project folder in your terminal:

```Bash
.venv\Scripts\activate
```
(Note: The environment already contains NumPy, Matplotlib, Scikit-learn, and TensorFlow. TensorFlow is only used in this project to fetch the Fashion-MNIST dataset and for the Section 5 performance comparison. All core training logic relies exclusively on NumPy).

Once the virtual environment is activated, you can navigate to the notebooks/ directory and run the Jupyter Notebook to view the project.
