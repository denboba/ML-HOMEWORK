# Quick Start Guide - Tema 2 Machine Learning Homework

## Overview

This homework implementation includes:
- **Part 1**: Image Classification using MLP and CNN models
- **Part 2**: Romanian Sentiment Analysis using RNN and LSTM models

## Prerequisites

All required packages are installed. Run verification:
```bash
python verify_implementation.py
```

## Step-by-Step Guide

### 1. Data Exploration

#### Part 1: Image Classification
```bash
python src/part1_image_classification/data_exploration.py
```

**What it does:**
- Analyzes the Imagebits dataset (10 classes, 8000 train images)
- Analyzes the Land Patches dataset (10 classes, 2000 train images)
- Generates visualizations showing class distributions
- Creates sample image grids for each dataset
- Saves results to `results/part1/`

**Expected output:**
- `Imagebits_class_distribution.png`
- `Imagebits_sample_images.png`
- `Land_Patches_class_distribution.png`
- `Land_Patches_sample_images.png`

#### Part 2: Sentiment Analysis
```bash
python src/part2_sentiment_analysis/data_exploration.py
```

**What it does:**
- Analyzes the Romanian sentiment dataset (17,941 train, 11,005 test)
- Shows sentiment distribution (positive/negative)
- Analyzes text length statistics
- Visualizes most frequent words per sentiment
- Saves results to `results/part2/`

**Expected output:**
- `sentiment_distribution.png`
- `text_length_analysis.png`
- `word_frequency.png`

### 2. Training Models

#### Option A: Train Single Model (Part 1)

**Train MLP on Imagebits:**
```bash
python src/part1_image_classification/train.py
```

**Train CNN on Imagebits with augmentation:**
Edit the configuration in `train.py` or use directly.

#### Option B: Train All Models (Recommended)

**Part 1 - All Image Classification Models:**
```bash
python src/part1_image_classification/run_experiments.py
```

This will train:
1. MLP Basic on Imagebits (no augmentation)
2. MLP Basic on Imagebits (with augmentation)
3. CNN Basic on Imagebits (no augmentation)
4. CNN Basic on Imagebits (with augmentation)
5. CNN Improved on Imagebits (with augmentation)
6. MLP Basic on Land Patches
7. CNN Basic on Land Patches
8. CNN Basic on Land Patches (with augmentation)

**Expected time:** 2-4 hours depending on hardware (GPU recommended)

**Part 2 - All Sentiment Analysis Models:**
```bash
python src/part2_sentiment_analysis/run_experiments.py
```

This will train:
1. Simple RNN
2. LSTM Unidirectional
3. LSTM Bidirectional
4. Improved LSTM with Attention

**Expected time:** 1-2 hours depending on hardware

### 3. Generate Summary Report

After training models:
```bash
python generate_report.py
```

**What it does:**
- Collects results from all experiments
- Creates summary tables with metrics
- Generates CSV files with comparisons
- Prints comprehensive summary to console

**Output files:**
- `results/part1_summary.csv`
- `results/part2_summary.csv`

## Understanding the Results

### For Each Experiment

Each trained model creates a folder in `results/part1/` or `results/part2/` containing:

- **`training_history.png`**: Training and validation loss/accuracy curves
- **`confusion_matrix.png`**: Confusion matrix on test set
- **`classification_report.txt`**: Detailed precision, recall, F1 scores
- **`config.json`**: Full configuration used for training
- **`history.json`**: Complete training history (all epochs)
- **`best_model.pth`**: Best model weights
- **`final_model.pth`**: Final model weights

### Key Metrics

- **Accuracy**: Overall classification accuracy
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Shows which classes are confused

## Model Architectures

### Part 1: Image Classification

#### MLP (Multi-Layer Perceptron)
```
Input (96×96×3) → Flatten → FC(512) → BatchNorm → ReLU → Dropout →
FC(256) → BatchNorm → ReLU → Dropout → FC(128) → BatchNorm → ReLU →
Dropout → FC(10)
```

#### CNN (Convolutional Neural Network)
```
Input (96×96×3 or 64×64×3) →
[Conv(32) → BatchNorm → ReLU × 2 → MaxPool] →
[Conv(64) → BatchNorm → ReLU × 2 → MaxPool] →
[Conv(128) → BatchNorm → ReLU × 2 → MaxPool] →
GlobalAvgPool → Dropout → FC(10)
```

### Part 2: Sentiment Analysis

#### Simple RNN
```
Input → Embedding(100) → RNN(128, 2 layers) → Dropout → FC(2)
```

#### LSTM
```
Input → Embedding(128) → LSTM(128, 2 layers, bidirectional) →
Dropout → FC(2)
```

#### Improved LSTM with Attention
```
Input → Embedding(200) → BiLSTM(256, 2 layers) → Attention →
Dropout → FC(256) → ReLU → Dropout → FC(2)
```

## Data Augmentation

### Part 1 (Images)
When `use_augmentation=True`:
- Horizontal flip (p=0.5)
- Random brightness/contrast (p=0.5)
- Random shift/scale/rotation (p=0.5)
- Coarse dropout (p=0.3)

### Part 2 (Text)
Currently using basic tokenization. Can be extended with:
- Random word swap/delete/insert
- Back-translation
- Synonym replacement

## Hardware Recommendations

- **Minimum**: CPU with 8GB RAM (will be slow)
- **Recommended**: GPU with 4GB+ VRAM
- **Optimal**: GPU with 8GB+ VRAM (CUDA compatible)

To check if GPU is available:
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
```

## Adjusting Training Parameters

### Reduce Training Time

If training is too slow, you can:

1. **Reduce epochs**: Change `epochs` parameter (e.g., from 40 to 20)
2. **Increase batch size**: Change `batch_size` (if you have enough memory)
3. **Reduce model size**: Use basic models instead of improved versions
4. **Train fewer experiments**: Comment out some experiments in `run_experiments.py`

### Improve Model Performance

To get better results:

1. **Increase epochs**: Train for more iterations
2. **Use data augmentation**: Set `use_augmentation=True`
3. **Tune learning rate**: Try different values (0.0001, 0.0005, 0.001)
4. **Adjust dropout**: Lower dropout may help if underfitting
5. **Use improved architectures**: Use ImprovedCNN or ImprovedLSTM

## Troubleshooting

### Out of Memory Error
- Reduce batch size
- Use smaller model
- Close other applications

### Poor Performance
- Check if augmentation helps
- Try different learning rates
- Increase model capacity
- Train for more epochs

### Slow Training
- Ensure GPU is being used
- Increase batch size
- Reduce number of workers in data loader

## Final Report

After completing all experiments, you should have:

1. **Visualizations**: All plots in `results/` directories
2. **Metrics**: Summary CSVs with performance comparisons
3. **Models**: Trained model checkpoints
4. **Analysis**: Confusion matrices and classification reports

Use these to write your final PDF report including:
- Data exploration insights
- Architecture choices and justifications
- Training curves and metrics
- Confusion matrices
- Comparison of different approaches
- Impact of augmentation
- Conclusions and future work

## Need Help?

Check:
1. README.md for detailed documentation
2. verify_implementation.py for system check
3. Individual script comments for implementation details
