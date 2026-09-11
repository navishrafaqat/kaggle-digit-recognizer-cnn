# Kaggle Digit Recognizer — Handwritten Digit Classification with CNNs

A computer vision project built around Kaggle's **Digit Recognizer** competition, using the MNIST handwritten-digit dataset to investigate how traditional machine learning models compare with neural networks and convolutional neural networks for image classification.

The project follows a progressive modeling approach:

**Logistic Regression → Random Forest → Neural Network → CNN**

The final CNN achieved **98.92% validation accuracy**, substantially outperforming the traditional machine-learning baselines.

---

## 📌 Project Overview

Handwritten digit recognition is a classic computer vision problem where each image represents one of ten digits (0–9).

The dataset contains grayscale **28 × 28 pixel images**, represented as 784 pixel features.

The objective of this project was not only to obtain predictions, but to build and compare several different modeling approaches and understand how model architecture affects image-classification performance.

### Key objectives

- Explore and preprocess image data
- Establish a traditional machine-learning baseline
- Compare tree-based and linear approaches
- Build a fully connected neural network
- Build a convolutional neural network
- Evaluate models on a held-out validation set
- Generate predictions for Kaggle's test set
- Submit the final model to Kaggle

---

## 🧠 Modeling Strategy

Rather than immediately using a deep-learning model, I established progressively more sophisticated baselines.

### 1. Logistic Regression

Logistic Regression was used as the initial baseline.

This provided a reference point for determining how well a relatively simple linear classifier could perform on the pixel representation.

**Validation Accuracy: 91.15%**

---

### 2. Random Forest

A Random Forest classifier was then trained using the normalized pixel features.

Configuration:

- `n_estimators = 100`
- `random_state = 42`
- `n_jobs = -1`

**Validation Accuracy: 96.39%**

The improvement over Logistic Regression demonstrated that the relationship between pixel patterns and digit classes is not adequately captured by a simple linear decision boundary.

---

### 3. Fully Connected Neural Network

The next model was a neural network operating on the 28 × 28 images.

Architecture:

```text
Input: 28 × 28 × 1
        ↓
Flatten
        ↓
Dense(128, ReLU)
        ↓
Dropout(0.2)
        ↓
Dense(10, Softmax)
