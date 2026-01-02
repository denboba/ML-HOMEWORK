"""
Text preprocessing and embedding utilities for Romanian sentiment analysis
"""

import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import pickle


class SimpleTokenizer:
    """
    Simple tokenizer for Romanian text
    (Alternative to spacy for simplicity)
    """
    
    def __init__(self, vocab_size=10000, max_length=200):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()
        
    def clean_text(self, text):
        """Clean and normalize text"""
        text = str(text).lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Remove special characters but keep Romanian letters
        text = re.sub(r'[^a-zăâîșțĂÂÎȘȚ\s]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def tokenize(self, text):
        """Tokenize text into words"""
        text = self.clean_text(text)
        return text.split()
    
    def fit(self, texts):
        """Build vocabulary from texts"""
        print("Building vocabulary...")
        
        # Count words
        for text in texts:
            words = self.tokenize(text)
            self.word_counts.update(words)
        
        # Create vocabulary with most common words
        # Reserve 0 for padding, 1 for unknown
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        
        most_common = self.word_counts.most_common(self.vocab_size - 2)
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
        
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        print(f"Vocabulary size: {len(self.word2idx)}")
        print(f"Most common words: {most_common[:10]}")
    
    def texts_to_sequences(self, texts, max_length=None):
        """Convert texts to sequences of indices"""
        if max_length is None:
            max_length = self.max_length
        
        sequences = []
        for text in texts:
            words = self.tokenize(text)
            # Convert words to indices
            seq = [self.word2idx.get(word, 1) for word in words]  # 1 is <UNK>
            
            # Truncate or pad
            if len(seq) > max_length:
                seq = seq[:max_length]
            else:
                seq = seq + [0] * (max_length - len(seq))  # 0 is <PAD>
            
            sequences.append(seq)
        
        return np.array(sequences)
    
    def save(self, path):
        """Save tokenizer"""
        with open(path, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_counts': self.word_counts,
                'vocab_size': self.vocab_size,
                'max_length': self.max_length
            }, f)
    
    def load(self, path):
        """Load tokenizer"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.word2idx = data['word2idx']
            self.idx2word = data['idx2word']
            self.word_counts = data['word_counts']
            self.vocab_size = data['vocab_size']
            self.max_length = data['max_length']


class SentimentDataset(Dataset):
    """Dataset for sentiment analysis"""
    
    def __init__(self, texts, labels, tokenizer, max_length=200):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Convert texts to sequences
        self.sequences = self.tokenizer.texts_to_sequences(texts, max_length)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        sequence = torch.LongTensor(self.sequences[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        return sequence, label


def load_and_preprocess_data(train_path, test_path, vocab_size=10000, max_length=200):
    """
    Load and preprocess the sentiment dataset
    
    Args:
        train_path: Path to training CSV
        test_path: Path to test CSV
        vocab_size: Maximum vocabulary size
        max_length: Maximum sequence length
    
    Returns:
        train_dataset, test_dataset, tokenizer, num_classes
    """
    # Load data
    print("Loading datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Get column names
    label_col = 'label'
    text_col = 'text'
    
    # Extract texts and labels
    train_texts = train_df[text_col].values
    train_labels = train_df[label_col].values
    
    test_texts = test_df[text_col].values
    test_labels = test_df[label_col].values
    
    num_classes = len(np.unique(train_labels))
    
    print(f"Train samples: {len(train_texts)}")
    print(f"Test samples: {len(test_texts)}")
    print(f"Number of classes: {num_classes}")
    
    # Create and fit tokenizer
    tokenizer = SimpleTokenizer(vocab_size=vocab_size, max_length=max_length)
    tokenizer.fit(train_texts)
    
    # Create datasets
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_length)
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, max_length)
    
    return train_dataset, test_dataset, tokenizer, num_classes


def get_data_loaders(train_dataset, test_dataset, batch_size=32, num_workers=2):
    """Create data loaders"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Test preprocessing
    train_dataset, test_dataset, tokenizer, num_classes = load_and_preprocess_data(
        'data/ro_sent/train.csv',
        'data/ro_sent/test.csv',
        vocab_size=10000,
        max_length=200
    )
    
    print(f"\nDataset created successfully!")
    print(f"Vocabulary size: {len(tokenizer.word2idx)}")
    print(f"Number of classes: {num_classes}")
    
    # Test data loader
    train_loader, test_loader = get_data_loaders(train_dataset, test_dataset, batch_size=32)
    
    # Get a batch
    sequences, labels = next(iter(train_loader))
    print(f"\nBatch shape:")
    print(f"  Sequences: {sequences.shape}")
    print(f"  Labels: {labels.shape}")
