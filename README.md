# Toxic Comment Classification & Image Generation with GANs

A comprehensive NLP and Computer Vision project portfolio showcasing multiple machine learning techniques for toxic comment detection and image generation.

## 📋 Table of Contents

- [Overview](#overview)
- [Projects](#projects)
  - [Toxic Comment Classification](#toxic-comment-classification)
  - [Image Generation with GANs](#image-generation-with-gans)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Results](#results)
- [Setup](#setup)
- [Usage](#usage)

## 🎯 Overview

This repository contains two main projects:

1. **Multi-label Toxic Comment Classification**: A comprehensive NLP project comparing multiple techniques (TF-IDF+SVC, BiLSTM, DistilBERT) for detecting toxic content in online comments across 6 categories.
2. **Image Generation with GANs**: A generative adversarial network implementation for creating realistic handwritten digits using the MNIST dataset.

## 📊 Projects

### Toxic Comment Classification

A multi-label classification system that identifies toxic comments across six categories:
- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

#### Methodology

**1. EDA & Preprocessing** (`EDA.ipynb`)
- Exploratory data analysis of the Kaggle Jigsaw Toxic Comment Classification dataset
- Text cleaning and normalization
- Label distribution analysis
- Stratified train/validation/test split (80/10/10)
- Feature engineering and statistics

**2. Model Implementations**

- **BiLSTM Baseline** (`BiLSTM Model.ipynb`)
  - Bidirectional LSTM with word-level embeddings
  - Mean + max pooling for sentence representation
  - Multi-label classification with BCE loss and class weighting
  - Per-label threshold tuning for optimal F1 scores

- **DistilBERT (Final Model)** (`DistillBERT_ChosenTechnique.ipynb`)
  - Fine-tuned DistilBERT transformer model
  - Subword tokenization for robust handling of slang/typos
  - Contextual understanding through self-attention mechanisms
  - Interactive Gradio interface for end-user demonstration
  - Comprehensive evaluation metrics

#### Key Features

- **Class Imbalance Handling**: Uses `pos_weight` in BCE loss to handle rare labels
- **Threshold Optimization**: Per-label threshold tuning on validation set
- **Comprehensive Metrics**: Micro-F1, Macro-F1, PR-AUC, Hamming Accuracy
- **Reproducibility**: Fixed seeds and consistent train/val/test splits
- **Production-Ready Interface**: Gradio demo for real-time inference

#### Performance

| Model | Micro-F1 | Macro-F1 | Mean PR-AUC | Hamming Accuracy |
|-------|----------|----------|-------------|------------------|
| DistilBERT | 0.789 | 0.694 | 0.741 | 0.984 |
| BiLSTM | 0.674 | 0.567 | 0.555 | 0.976 |

**DistilBERT** was selected as the final model for its superior performance on rare labels and better contextual understanding.

### Image Generation with GANs

A Generative Adversarial Network implementation for generating realistic handwritten digits from the MNIST dataset.

#### Architecture

- **Generator**: Maps random noise (100-dim) → 28×28 grayscale images
  - Fully connected layers with ReLU activations
  - Tanh output for normalized pixel values [-1, 1]
  
- **Discriminator**: Binary classifier for real vs. generated images
  - Deep feedforward network with dropout regularization
  - Sigmoid output for probability scores

#### Training Process

- Adversarial training with alternating updates
- Binary cross-entropy loss for both networks
- Adam optimizer (lr=0.0001)
- 30 epochs on MNIST training set

## 🛠 Technologies Used

### NLP Project
- **PyTorch**: Deep learning framework
- **Transformers (Hugging Face)**: Pre-trained DistilBERT model
- **scikit-learn**: Data splitting, metrics, and baseline models
- **pandas & numpy**: Data manipulation and analysis
- **matplotlib & seaborn**: Visualization
- **Gradio**: Interactive web interface

### GAN Project
- **PyTorch**: Neural network implementation
- **torchvision**: MNIST dataset loading and transforms
- **matplotlib**: Visualization of generated images

## 📁 Project Structure

```
.
├── EDA.ipynb                          # Exploratory data analysis and preprocessing
├── BiLSTM Model.ipynb                 # BiLSTM baseline implementation
├── DistillBERT_ChosenTechnique.ipynb  # DistilBERT final model with Gradio interface
├── ImageGeneration.ipynb              # GAN-based image generation
└── README.md                          # This file
```

## 📈 Results

### Toxic Comment Classification

- Successfully handles multi-label classification with 6 toxicity categories
- Achieves 98.4% Hamming accuracy with DistilBERT
- Strong performance on rare labels (threat, identity_hate) through class weighting
- Interactive demo interface for real-time classification

### Image Generation

- Generator learns to produce realistic MNIST-like digits
- Adversarial training achieves equilibrium between generator and discriminator
- Generated images demonstrate learned distribution of handwritten digits

## 🚀 Setup

### Prerequisites

```bash
pip install torch torchvision transformers
pip install pandas numpy scikit-learn matplotlib seaborn
pip install gradio
```

### Data

For the toxic comment classification project:
- Download the [Kaggle Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge) dataset
- Place `train.csv` in the working directory

For the GAN project:
- MNIST dataset is automatically downloaded by PyTorch

## 💻 Usage

### Toxic Comment Classification

1. Run `EDA.ipynb` to preprocess the data and create train/val/test splits
2. Run `BiLSTM Model.ipynb` to train the baseline model
3. Run `DistillBERT_ChosenTechnique.ipynb` to train the final model and launch the Gradio interface

### Image Generation

1. Run `ImageGeneration.ipynb` to train the GAN and generate images
2. Training takes approximately 30 epochs
3. Generated images are displayed in a 4×4 grid

## 📝 Notes

- All notebooks are designed to run on Google Colab with GPU support
- Model artifacts and metrics are saved for reproducibility
- Content warning: The toxic comment dataset contains offensive language
- Seeds are fixed for reproducible results across all experiments

## 🔗 Resources

- [Kaggle Jigsaw Competition](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [GAN Paper](https://arxiv.org/abs/1406.2661)

## 📄 License

This project is for educational and portfolio purposes.

---

**Author**: Luke Nguyen 
**Repository**: https://github.com/tuilachit/ToxicComment_NLP_Project
