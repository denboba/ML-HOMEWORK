# ML-HOMEWORK

## Tema 2 - Învățare Automată

This repository contains the implementation of Homework 2 for the Machine Learning course, covering:
1. Image Classification (Imagebits and Land Patches datasets)
2. Romanian Sentiment Analysis (ro_sent dataset)

## Project Structure

```
ML-HOMEWORK/
├── data/
│   └── ro_sent/          # Romanian sentiment dataset
├── imagebits/            # Image classification dataset 1
│   ├── train/
│   └── test/
├── land_patches/         # Image classification dataset 2
│   ├── train/
│   └── test/
├── src/
│   ├── part1_image_classification/
│   │   ├── data_exploration.py      # Data analysis and visualization
│   │   ├── data_loader.py           # Data loading utilities
│   │   ├── mlp_model.py             # MLP model implementation
│   │   ├── cnn_model.py             # CNN model implementation
│   │   ├── train.py                 # Training script
│   │   └── run_experiments.py       # Run all experiments
│   └── part2_sentiment_analysis/
│       ├── data_exploration.py      # Data analysis for text
│       ├── text_preprocessing.py    # Text preprocessing and tokenization
│       ├── rnn_models.py            # RNN and LSTM models
│       ├── train.py                 # Training script
│       └── run_experiments.py       # Run all experiments
├── results/
│   ├── part1/           # Results for image classification
│   └── part2/           # Results for sentiment analysis
├── requirements.txt
└── README.md
```

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Part 1: Image Classification

#### Data Exploration
```bash
cd /path/to/ML-HOMEWORK
python src/part1_image_classification/data_exploration.py
```

This will:
- Analyze class distributions
- Generate sample visualizations
- Save results to `results/part1/`

#### Training Models

To train a single model:
```bash
python src/part1_image_classification/train.py
```

To run all experiments:
```bash
python src/part1_image_classification/run_experiments.py
```

This will train:
- MLP models (basic and with augmentation)
- CNN models (basic and improved, with/without augmentation)
- Models on both Imagebits and Land Patches datasets

### Part 2: Romanian Sentiment Analysis

#### Data Exploration
```bash
python src/part2_sentiment_analysis/data_exploration.py
```

This will:
- Analyze sentiment distribution
- Visualize text length statistics
- Show word frequency distributions
- Save results to `results/part2/`

#### Training Models

To train a single model:
```bash
python src/part2_sentiment_analysis/train.py
```

To run all experiments:
```bash
python src/part2_sentiment_analysis/run_experiments.py
```

This will train:
- Simple RNN model
- LSTM (unidirectional and bidirectional)
- Improved LSTM with attention mechanism

## Model Architectures

### Part 1: Image Classification

#### MLP (Multi-Layer Perceptron)
- Flattens input images
- Multiple fully connected layers with BatchNorm and Dropout
- Configurable hidden layer sizes

#### CNN (Convolutional Neural Network)
- Multiple convolutional blocks (Conv -> BatchNorm -> ReLU -> MaxPool)
- Global average pooling
- Fully connected classification head
- Supports data augmentation using Albumentations

### Part 2: Sentiment Analysis

#### Simple RNN
- Embedding layer
- Multiple RNN layers
- Fully connected classifier

#### LSTM
- Embedding layer
- LSTM layers (unidirectional or bidirectional)
- Fully connected classifier

#### Improved LSTM
- Embedding layer
- Bidirectional LSTM
- Attention mechanism
- Multi-layer classifier

## Data Augmentation

### Part 1 (Images)
- Horizontal flip
- Random brightness and contrast
- Random shift, scale, and rotation
- Coarse dropout

### Part 2 (Text)
- Can be extended with:
  - Random word swap/delete/insert
  - Back-translation
  - Contextual word embeddings

## Results

Results for each experiment are saved in the respective directories:
- `results/part1/` - Image classification results
- `results/part2/` - Sentiment analysis results

Each experiment folder contains:
- `training_history.png` - Loss and accuracy curves
- `confusion_matrix.png` - Confusion matrix
- `classification_report.txt` - Detailed metrics
- `config.json` - Training configuration
- `history.json` - Training history data
- `best_model.pth` - Best model weights

## Requirements

See `requirements.txt` for full list of dependencies.

Key libraries:
- PyTorch
- Albumentations (image augmentation)
- scikit-learn
- pandas
- matplotlib
- seaborn

## Notes

- Training can take significant time depending on hardware (GPU recommended)
- Adjust batch sizes and epochs in configuration if needed
- All models save their best weights based on validation accuracy
- Learning rate scheduling is applied automatically
