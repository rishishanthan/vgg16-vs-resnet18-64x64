# Deep CNN Architectures: VGG-16 vs ResNet-18 on 64×64 Multi-Class Dataset

This project explores and compares two cornerstone convolutional architectures — **VGG-16** and **ResNet-18** — trained from scratch on a custom **64×64 RGB dataset** comprising three object categories: **dogs, cars, and food**.  
It demonstrates the evolution from classical stacked convolutional networks to modern residual architectures, focusing on convergence behavior, feature extraction, and performance metrics.

---

## 🎯 Objective

To design, train, and evaluate two deep CNN architectures:
1. **VGG-16** — a deep sequential network emphasizing depth and convolutional uniformity.  
2. **ResNet-18** — a modern architecture employing residual (skip) connections to overcome vanishing gradients and enable efficient training of deep networks.

The project highlights how architectural innovations influence **training stability**, **learning efficiency**, and **final accuracy**.

---

## 🧠 Dataset Overview

- **Classes:** 3 (dogs, cars, food)  
- **Image size:** 64×64×3  
- **Split:** 70% train, 20% validation, 10% test  
- **Augmentations:**
  - Random horizontal flips  
  - Random rotations (±15°)  
  - Color jitter (brightness, contrast, saturation)  
  - Normalization to mean = 0.5, std = 0.5  

Data loaders were implemented with `torchvision.datasets.ImageFolder` and `DataLoader` (batch size = 32).

---

## 🏗️ Model Architectures

### 🧩 VGG-16 (Baseline)
- 13 convolutional layers + 3 fully connected layers.  
- Uniform 3×3 convolutions with stride 1 and padding 1.  
- ReLU activation after each conv.  
- Max pooling (2×2) after each block.  
- FC layers: [4096 → 4096 → 3] with dropout (p=0.5).  
- No skip connections — purely sequential.

**Optimizer:** Adam (lr = 0.001)  
**Scheduler:** ReduceLROnPlateau (patience=3)  
**Loss:** CrossEntropyLoss  
**Epochs:** 40  

---

### ⚙️ ResNet-18 (Improved)
- Consists of 8 residual blocks (2 layers per block).  
- Skip connections directly add input to the output (identity mapping).  
- Reduces vanishing gradient issues in deep nets.  
- Global average pooling replaces dense FC layers → fewer parameters, better generalization.

**Optimizer:** SGD with momentum (0.9), lr = 0.001  
**Scheduler:** StepLR (step_size=7, gamma=0.1)  
**Loss:** CrossEntropyLoss  
**Epochs:** 40  

---

## 📈 Training Progress

### VGG-16 Results
| Epoch | Train Acc | Val Acc | Test Acc | Notes |
|-------|------------|----------|-----------|-------|
| 10 | 63.25% | 61.90% | — | Initial learning stabilization |
| 20 | 78.32% | 75.41% | — | Feature convergence |
| 30 | 83.75% | 82.10% | — | Saturation region |
| 40 | **86.47%** | **84.60%** | **85.12%** | Final convergence |

### ResNet-18 Results
| Epoch | Train Acc | Val Acc | Test Acc | Notes |
|-------|------------|----------|-----------|-------|
| 10 | 66.28% | 64.22% | — | Stable early learning |
| 20 | 80.92% | 77.85% | — | Faster feature learning |
| 30 | 88.76% | 86.15% | — | High generalization |
| 40 | **90.84%** | **88.30%** | **89.62%** | Best model performance |

---

## 📊 Visualizations

### 1. Accuracy & Loss Curves
- **VGG-16**: Gradual but steady convergence, mild overfitting after epoch 35.  
- **ResNet-18**: Faster and smoother convergence; validation accuracy closely follows training accuracy.

### 2. Confusion Matrices
Both models achieved balanced performance across classes, with **ResNet-18** showing fewer false negatives on “cars” and “dogs.”

### 3. Feature Maps
Feature visualization revealed:
- VGG-16 learned fine textures and color boundaries.  
- ResNet-18 captured larger contextual patterns due to skip connections and deeper representation.

---

## 📊 Comparative Analysis

| Metric | VGG-16 | ResNet-18 |
|--------|---------|-----------|
| Parameters | ~134M | ~11.7M |
| Convergence Speed | Moderate | Faster |
| Final Test Accuracy | 85.12% | **89.62%** |
| Overfitting Tendency | Higher | Lower |
| Gradient Stability | Moderate | Excellent |
| Computation Time | ~1.3× longer | Faster (residuals enable reuse) |

### Conclusion
ResNet-18 **outperformed VGG-16** across all evaluation metrics while using significantly fewer parameters.  
Residual learning allowed deeper networks to train faster and generalize better.

---

## 🧩 Implementation Notes
- Trained with PyTorch’s native modules only — no pre-trained weights.  
- Used manual learning rate scheduling, early stopping, and dynamic validation feedback.  
- Reproducible seeds ensure deterministic results (`torch.manual_seed(42)`).

---

## 🧰 Dependencies

**requirements**
```txt
torch==2.4.1
torchvision==0.19.1
numpy==2.1.3
pandas==2.2.3
matplotlib==3.9.3
seaborn==0.13.2
scikit-learn==1.5.2
tqdm==4.66.5
```

---

## 🧠 Learnings & Insights
- Residual connections are game-changers for training deep networks effectively.
- VGG-style architectures are still powerful but require stronger regularization and tuning.
- Dataset normalization and augmentation drastically improve generalization.
- Manual LR scheduling and early stopping were key to achieving optimal performance.
- Building these networks from scratch gave complete interpretability of layer behaviors and feature extraction.

## 🏆 Results Summary
| Model     | Best Test Accuracy | Remarks                                    |
| --------- | ------------------ | ------------------------------------------ |
| VGG-16    | 85.12%             | Classic deep stack, slightly overfits      |
| ResNet-18 | **89.62%**         | Deeper but more stable, faster convergence |


## 🚀 Future Work
- Add EfficientNet or DenseNet to extend the comparison.
- Experiment with transfer learning on larger datasets (CIFAR-10 / Tiny-ImageNet).
- Implement Grad-CAM visualizations to analyze activation relevance.
- Deploy the best-performing model for inference via a simple web interface.
