# Variational Autoencoder (VAE) for Fashion MNIST

This project implements a Variational Autoencoder (VAE) to learn the latent representations of the Fashion MNIST dataset. The VAE is trained to reconstruct images of clothing items while simultaneously learning a meaningful 2D latent space.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Overview

A Variational Autoencoder (VAE) is a generative model that combines principles from deep learning and probabilistic modeling. This project trains a VAE to generate Fashion MNIST images and visualize the learned 2D latent space.

The **Fashion MNIST** dataset contains grayscale images of 10 types of clothing items, each represented as a 28x28 pixel image.

---

## Features

- **Custom Sampling Layer**:
  Implements the reparameterization trick for backpropagation through stochastic nodes.

- **Encoder**:
  Encodes input images into a 2D latent space (`mean` and `log_var`) for probabilistic representation.

- **Decoder**:
  Decodes sampled latent vectors into reconstructed images.

- **Custom VAE Class**:
  Combines encoder and decoder, computes reconstruction and KL divergence losses, and updates weights.

- **Latent Space Visualization**:
  - Manifold of generated images from the 2D latent space.
  - Clustering of latent representations, colored by labels.

---

## Technologies Used

- Python 3.x
- TensorFlow 2.x / Keras
- NumPy
- Matplotlib

---

## Dataset

The **Fashion MNIST** dataset consists of:
- **Training Set**: 60,000 images
- **Testing Set**: 10,000 images

Each image is a 28x28 grayscale representation of a clothing item, classified into 10 categories:
1. T-shirt / top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

The dataset is preloaded using `keras.datasets.fashion_mnist`.

---

## Architecture

### Encoder
- 2 Convolutional Layers (with ReLU activation)
- Dense Layer for `mean` and `log_var` (latent space parameters)
- Custom Sampling Layer for reparameterization

### Decoder
- Dense Layer for upsampling
- 2 Transposed Convolutional Layers
- Final Transposed Convolutional Layer with sigmoid activation

### Loss Function
- **Reconstruction Loss**: Measures the difference between original and reconstructed images using binary cross-entropy.
- **KL Divergence**: Regularizes the latent space to follow a normal distribution.

---

