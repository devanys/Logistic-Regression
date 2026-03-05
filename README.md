# 📊 Logistic Regression from Scratch

<img width="810" height="556" alt="image" src="https://github.com/user-attachments/assets/0476fc4c-8ac0-4c13-a569-947b27d675c7" />

> A implementation of Logistic Regression — built entirely without NumPy, PyTorch, or any ML library. Covers Binomial & Multinomial Logistic Regression, F1 Score, AUC-ROC, and AUC-PR.

---

## 📌 Overview

This project demonstrates the **mathematical foundations of classification** by implementing Logistic Regression and its full evaluation pipeline from first principles. Every component — from the sigmoid function to ROC curves 

---

## 📐 Mathematical Foundations

### 1. Sigmoid Function

The sigmoid (logistic) function squashes any real value $z$ into the range $(0, 1)$, making it interpretable as a probability:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

<img width="1065" height="665" alt="image" src="https://github.com/user-attachments/assets/a2337837-e3af-43b9-934f-b9fce416b5ee" />

Key properties:
- $\sigma(0) = 0.5$
- $\sigma(z) \to 1$ as $z \to +\infty$
- $\sigma(z) \to 0$ as $z \to -\infty$

<img width="405" height="93" alt="Screenshot 2026-03-06 020057" src="https://github.com/user-attachments/assets/5d9db2d6-a36a-4163-ae2b-692e68bbe182" />

---

### 2. Softmax Function

For multi-class problems with $K$ classes, softmax converts a vector of raw scores $\mathbf{z}$ into a probability distribution:

$$\text{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}, \quad \sum_{k=1}^{K} \text{softmax}(\mathbf{z})_k = 1$$

Numerically stable version subtracts $\max(\mathbf{z})$ before exponentiation to prevent overflow.

---

### 3. Binomial Logistic Regression — Forward Pass

Given input $\mathbf{x} \in \mathbb{R}^n$, weight vector $\mathbf{w}$, and bias $b$:

$$z = \mathbf{w} \cdot \mathbf{x} + b$$

$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}} = P(y=1 \mid \mathbf{x})$$

$$\text{pred} = \begin{cases} 1 & \text{if } \hat{y} \geq 0.5 \\ 0 & \text{otherwise} \end{cases}$$

<img width="487" height="164" alt="image" src="https://github.com/user-attachments/assets/7197eb9e-a096-4975-a553-813249d6bc58" />

---

### 4. Binary Cross-Entropy Loss

The loss function for binary classification penalises incorrect probability estimates:

$$\mathcal{L}(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

- When $y=1$: loss $= -\log(\hat{y})$ — penalises low confidence on positive samples
- When $y=0$: loss $= -\log(1-\hat{y})$ — penalises high confidence on negative samples
- Untrained model loss $\approx \log(2) \approx 0.6931$

<img width="573" height="161" alt="image" src="https://github.com/user-attachments/assets/3c5430d5-30a6-4ab6-9fd4-a4fe631ddae8" />

---

### 5. Gradient Computation (Binary)

Partial derivatives of the loss with respect to parameters:

$$\frac{\partial \mathcal{L}}{\partial w_j} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i) \cdot x_{ij}$$

$$\frac{\partial \mathcal{L}}{\partial b} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)$$

The gradient of the sigmoid loss has the same elegant form as linear regression — the error $(\hat{y} - y)$ times the input.

<img width="344" height="133" alt="Screenshot 2026-03-06 020438" src="https://github.com/user-attachments/assets/7e8a6a43-136a-4db1-ba8e-dd56ec5e87c4" />


---

### 6. SGD Update

Parameters are updated in the direction that reduces the loss:

$$\mathbf{w} \leftarrow \mathbf{w} - \eta \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{w}}$$

$$b \leftarrow b - \eta \cdot \frac{\partial \mathcal{L}}{\partial b}$$

where $\eta$ is the **learning rate**.

<img width="353" height="161" alt="image" src="https://github.com/user-attachments/assets/a950b38e-eb63-4796-9fa0-3c0ded92d823" />

---

### 7. Multinomial Logistic Regression — Forward Pass

For $K$ classes, each class $k$ has its own weight vector $\mathbf{W}_k$ and bias $b_k$:

$$z_k = \mathbf{W}_k \cdot \mathbf{x} + b_k, \quad k = 0, 1, \ldots, K-1$$

$$\hat{y}_k = \text{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{j=0}^{K-1} e^{z_j}}$$

$$\text{pred} = \arg\max_k \, \hat{y}_k$$

<img width="442" height="206" alt="image" src="https://github.com/user-attachments/assets/605e39f8-8f47-4d3e-9767-e20ce04e66d3" />

---

### 8. Categorical Cross-Entropy Loss

Using one-hot encoding where $y_{ik} = 1$ if sample $i$ belongs to class $k$:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=0}^{K-1} y_{ik} \cdot \log(\hat{y}_{ik})$$

Simplified (only the true class contributes):

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log\left(\hat{y}_{i, \, y_i}\right)$$

- Untrained model loss $\approx \log(K) = \log(3) \approx 1.0986$

<img width="561" height="137" alt="image" src="https://github.com/user-attachments/assets/37a1d014-2141-41a2-a9a7-a4e35676a9a9" />

---

### 9. Multinomial Gradients

For each class $k$:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_k} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_{ik} - y_{ik}) \cdot \mathbf{x}_i$$

$$\frac{\partial \mathcal{L}}{\partial b_k} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_{ik} - y_{ik})$$

<img width="414" height="164" alt="image" src="https://github.com/user-attachments/assets/0c0b4fec-125a-4d8a-b24e-b965ea50e644" />

---

## 🖼️ Training Results

### Binomial — Loss & Accuracy Curves

<img width="1565" height="513" alt="image" src="https://github.com/user-attachments/assets/08a7eb35-c857-481c-9a27-627380aea101" />

### Binomial — Decision Boundary

$$\mathbf{w} \cdot \mathbf{x} + b = 0 \quad \Rightarrow \quad x_2 = -\frac{w_1 x_1 + b}{w_2}$$

<img width="940" height="778" alt="image" src="https://github.com/user-attachments/assets/33479eea-e15f-49ee-a454-04bfbbd1d332" />

### Multinomial — Loss & Accuracy Curves

<img width="1596" height="529" alt="image" src="https://github.com/user-attachments/assets/4d80227f-37bf-46a4-a1b8-fa1e645e60cb" />

### Multinomial — Decision Regions

<img width="951" height="775" alt="image" src="https://github.com/user-attachments/assets/6b8e4fd9-da7e-4420-a33b-7e17deff3672" />

---

## 📏 Evaluation Metrics

### 10. Confusion Matrix

|  | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | TN | FP |
| **Actual 1** | FN | TP |

<img width="479" height="281" alt="image" src="https://github.com/user-attachments/assets/a0790503-4ac0-4577-b7c9-437b58d4aab8" />

---

### 11. F1 Score

**Precision** measures how many predicted positives are actually positive:

$$\text{Precision} = \frac{TP}{TP + FP}$$

**Recall** measures how many actual positives were correctly found:

$$\text{Recall} = \frac{TP}{TP + FN}$$

**F1 Score** is the harmonic mean of Precision and Recall:

$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

For multiclass, **Macro F1** averages F1 across all $K$ classes:

$$F_1^{\text{macro}} = \frac{1}{K} \sum_{k=0}^{K-1} F_1^{(k)}$$

<img width="558" height="252" alt="image" src="https://github.com/user-attachments/assets/31e7d118-c531-431a-908f-76c89df9e63a" />

---

### 12. AUC-ROC

At each threshold $t$, compute:

$$\text{TPR} = \frac{TP}{TP + FN} \quad \text{(True Positive Rate / Recall)}$$

$$\text{FPR} = \frac{FP}{FP + TN} \quad \text{(False Positive Rate)}$$

The ROC curve plots **TPR vs FPR** across all thresholds. AUC is computed via the **trapezoidal rule**:

$$\text{AUC-ROC} = \int_0^1 \text{TPR}(\text{FPR}) \, d(\text{FPR})$$

| Score | Interpretation |
|-------|----------------|
| 1.00 | Perfect classifier |
| 0.90+ | Excellent |
| 0.80+ | Good |
| 0.50 | Random (no skill) |

---

### 13. AUC-PR

The PR curve plots **Precision vs Recall** across all thresholds:

$$\text{AUC-PR} = \int_0^1 \text{Precision}(\text{Recall}) \, d(\text{Recall})$$

- Baseline (random classifier) $= \frac{\text{Positive samples}}{N}$ (class prevalence)
- **More informative than ROC on imbalanced datasets**

<img width="1669" height="613" alt="image" src="https://github.com/user-attachments/assets/945af6e3-40a1-441c-889a-f73d7ed78ce6" />

---

### 14. Final Summary

<img width="571" height="564" alt="image" src="https://github.com/user-attachments/assets/3a58c4b4-9f2b-48ab-bf35-6b6eb4af392b" />

---

## 🧪 Synthetic Datasets

### Binary Dataset

$$y = \begin{cases} 0 & \mathbf{x} \sim \mathcal{N}((-1,-1),\, 0.8^2 I) \\ 1 & \mathbf{x} \sim \mathcal{N}((+1,+1),\, 0.8^2 I) \end{cases}$$

- $N = 200$ samples, 2 features, 2 classes (balanced)

### Multiclass Dataset

$$y = k, \quad \mathbf{x} \sim \mathcal{N}(\mu_k,\, 0.7^2 I), \quad k \in \{0,1,2\}$$

- Centers: $\mu_0 = (-2,0)$, $\mu_1 = (2,0)$, $\mu_2 = (0, 2.5)$
- $N = 240$ samples, 2 features, 3 classes (balanced)

---

## ⚙️ Hyperparameters

| Parameter | Binary | Multinomial | Description |
|-----------|--------|-------------|-------------|
| `lr` | 0.1 | 0.1 | Learning rate |
| `epochs` | 300 | 400 | Training epochs |
| `batch_size` | 16 | 16 | Mini-batch size |
| `n_features` | 2 | 2 | Input features |
| `n_classes` | 2 | 3 | Output classes |

---

## 📊 Results Summary

| Metric | Binary | Multinomial |
|--------|--------|-------------|
| Final Loss | ~0.07 | ~0.05 |
| Accuracy | ~97.5% | ~98.75% |
| F1 Score | ~0.97 | ~0.99 (macro) |
| AUC-ROC | ~0.99 | — |
| AUC-PR | ~0.99 | — |

<img width="1636" height="751" alt="image" src="https://github.com/user-attachments/assets/23930ff6-e975-4925-b115-736b4d4efe8a" />

---

## 📚 Concepts Covered

- ✅ Sigmoid & Softmax activation functions
- ✅ Binomial Logistic Regression (binary classification)
- ✅ Multinomial Logistic Regression (multi-class)
- ✅ Binary Cross-Entropy loss
- ✅ Categorical Cross-Entropy loss
- ✅ Manual gradient computation
- ✅ Mini-batch SGD
- ✅ Decision boundary & decision regions visualization
- ✅ Confusion matrix
- ✅ Precision, Recall, F1 Score (binary & macro)
- ✅ AUC-ROC with trapezoidal rule
- ✅ AUC-PR (Precision-Recall curve)
