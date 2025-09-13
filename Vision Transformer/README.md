# Vision Transformer (ViT) Implementation

This folder contains a PyTorch implementation of the Vision Transformer (ViT) model as described in the paper ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929).

## Components

### Model Architecture (`model.py`)
- Implementation of the Vision Transformer architecture with the following components:
  - Patch Embedding layer for converting images into sequences of patches
  - Transformer Encoder blocks with Multi-head Self-Attention
  - MLP Head for classification
  - Learnable CLS token and positional embeddings

### Training Pipeline (`train.py`)
- Complete training setup for CIFAR-10 dataset
- Features:
  - Data augmentation (random cropping, horizontal flipping)
  - AdamW optimizer with cosine learning rate scheduling
  - TensorBoard logging for metrics
  - Checkpoint saving and loading
  - Model evaluation during training

### Configuration (`config.py`)
- Configurable hyperparameters through command-line arguments:
  - Model architecture parameters (embedding dimension, number of heads, etc.)
  - Training parameters (batch size, learning rate, epochs)
  - Checkpoint handling settings

## Dataset
The implementation is configured for the CIFAR-10 dataset, which is automatically downloaded and preprocessed during training.

## Usage
To train the model:
```bash
python train.py
```

Additional arguments can be passed to modify the default configuration:
```bash
python train.py --batch_size 128 --learning_rate 1e-4 --num_epochs 200
```