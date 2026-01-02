"""
Data Exploration for Romanian Sentiment Analysis
Analyzes the ro_sent dataset
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

def load_dataset():
    """Load the ro_sent dataset"""
    train_df = pd.read_csv('data/ro_sent/train.csv')
    test_df = pd.read_csv('data/ro_sent/test.csv')
    
    return train_df, test_df

def explore_dataset(train_df, test_df):
    """
    Explore and visualize dataset characteristics
    """
    print(f"\n{'='*60}")
    print("Romanian Sentiment Analysis Dataset Exploration")
    print(f"{'='*60}")
    
    # Basic statistics
    print(f"\nDataset Size:")
    print(f"  Train samples: {len(train_df):,}")
    print(f"  Test samples: {len(test_df):,}")
    print(f"  Total samples: {len(train_df) + len(test_df):,}")
    
    print(f"\nDataset Columns:")
    print(f"  Train: {train_df.columns.tolist()}")
    print(f"  Test: {test_df.columns.tolist()}")
    
    # Show sample data
    print(f"\nSample Training Data:")
    print(train_df.head())
    
    # Get sentiment column name (might be 'label', 'sentiment', etc.)
    label_col = [col for col in train_df.columns if col.lower() in ['label', 'sentiment', 'polarity']][0]
    text_col = [col for col in train_df.columns if col.lower() in ['text', 'sentence', 'review']][0]
    
    print(f"\nUsing columns: text='{text_col}', label='{label_col}'")
    
    # 1. Class Balance Analysis
    print(f"\n{'='*60}")
    print("1. Class Balance Analysis")
    print(f"{'='*60}")
    
    train_label_counts = train_df[label_col].value_counts()
    test_label_counts = test_df[label_col].value_counts()
    
    print(f"\nTrain set class distribution:")
    for label, count in train_label_counts.items():
        print(f"  {label}: {count:,} ({100*count/len(train_df):.2f}%)")
    
    print(f"\nTest set class distribution:")
    for label, count in test_label_counts.items():
        print(f"  {label}: {count:,} ({100*count/len(test_df):.2f}%)")
    
    # Plot class distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Train distribution
    train_label_counts.plot(kind='bar', ax=axes[0], color=['#FF6B6B', '#4ECDC4'])
    axes[0].set_title('Train Set - Sentiment Distribution', fontsize=14)
    axes[0].set_xlabel('Sentiment', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Test distribution
    test_label_counts.plot(kind='bar', ax=axes[1], color=['#FF6B6B', '#4ECDC4'])
    axes[1].set_title('Test Set - Sentiment Distribution', fontsize=14)
    axes[1].set_xlabel('Sentiment', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/part2/sentiment_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Text Length Analysis
    print(f"\n{'='*60}")
    print("2. Text Length Analysis")
    print(f"{'='*60}")
    
    # Calculate text lengths (in words)
    train_df['text_length_words'] = train_df[text_col].apply(lambda x: len(str(x).split()))
    test_df['text_length_words'] = test_df[text_col].apply(lambda x: len(str(x).split()))
    
    # Calculate text lengths (in characters)
    train_df['text_length_chars'] = train_df[text_col].apply(lambda x: len(str(x)))
    test_df['text_length_chars'] = test_df[text_col].apply(lambda x: len(str(x)))
    
    print(f"\nText Length Statistics (words):")
    print(f"  Train - Mean: {train_df['text_length_words'].mean():.2f}, Median: {train_df['text_length_words'].median():.0f}, "
          f"Max: {train_df['text_length_words'].max():.0f}")
    print(f"  Test - Mean: {test_df['text_length_words'].mean():.2f}, Median: {test_df['text_length_words'].median():.0f}, "
          f"Max: {test_df['text_length_words'].max():.0f}")
    
    # Text length distribution by sentiment
    print(f"\nText Length by Sentiment (words):")
    for label in train_df[label_col].unique():
        train_subset = train_df[train_df[label_col] == label]['text_length_words']
        print(f"  {label} - Mean: {train_subset.mean():.2f}, Median: {train_subset.median():.0f}")
    
    # Plot text length distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Histogram of text length (all data)
    axes[0, 0].hist(train_df['text_length_words'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Train Set - Text Length Distribution (Words)', fontsize=12)
    axes[0, 0].set_xlabel('Number of Words', fontsize=10)
    axes[0, 0].set_ylabel('Frequency', fontsize=10)
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].axvline(train_df['text_length_words'].mean(), color='red', linestyle='--', label='Mean')
    axes[0, 0].axvline(train_df['text_length_words'].median(), color='green', linestyle='--', label='Median')
    axes[0, 0].legend()
    
    # Text length by sentiment
    for label in train_df[label_col].unique():
        subset = train_df[train_df[label_col] == label]['text_length_words']
        axes[0, 1].hist(subset, bins=30, alpha=0.6, label=str(label))
    axes[0, 1].set_title('Train Set - Text Length by Sentiment', fontsize=12)
    axes[0, 1].set_xlabel('Number of Words', fontsize=10)
    axes[0, 1].set_ylabel('Frequency', fontsize=10)
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Character length distribution
    axes[1, 0].hist(train_df['text_length_chars'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1, 0].set_title('Train Set - Character Length Distribution', fontsize=12)
    axes[1, 0].set_xlabel('Number of Characters', fontsize=10)
    axes[1, 0].set_ylabel('Frequency', fontsize=10)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Box plot
    data_to_plot = [train_df[train_df[label_col] == label]['text_length_words'].values 
                    for label in sorted(train_df[label_col].unique())]
    axes[1, 1].boxplot(data_to_plot, labels=sorted(train_df[label_col].unique()))
    axes[1, 1].set_title('Text Length Distribution by Sentiment (Box Plot)', fontsize=12)
    axes[1, 1].set_xlabel('Sentiment', fontsize=10)
    axes[1, 1].set_ylabel('Number of Words', fontsize=10)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/part2/text_length_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Word Frequency Analysis
    print(f"\n{'='*60}")
    print("3. Most Frequent Words Analysis")
    print(f"{'='*60}")
    
    # Get most common words per class
    for label in train_df[label_col].unique():
        subset = train_df[train_df[label_col] == label]
        
        # Tokenize and count words
        all_words = []
        for text in subset[text_col]:
            words = str(text).lower().split()
            all_words.extend(words)
        
        word_freq = Counter(all_words)
        most_common = word_freq.most_common(20)
        
        print(f"\nTop 20 words for sentiment '{label}':")
        for word, count in most_common[:10]:
            print(f"  '{word}': {count:,}")
    
    # Plot word frequency for each sentiment
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, label in enumerate(sorted(train_df[label_col].unique())):
        subset = train_df[train_df[label_col] == label]
        
        # Get all words
        all_words = []
        for text in subset[text_col]:
            words = str(text).lower().split()
            all_words.extend(words)
        
        word_freq = Counter(all_words)
        most_common = word_freq.most_common(15)
        
        words, counts = zip(*most_common)
        
        axes[idx].barh(range(len(words)), counts, color='skyblue' if idx == 0 else 'lightcoral')
        axes[idx].set_yticks(range(len(words)))
        axes[idx].set_yticklabels(words)
        axes[idx].set_xlabel('Frequency', fontsize=10)
        axes[idx].set_title(f'Top 15 Words - Sentiment: {label}', fontsize=12)
        axes[idx].invert_yaxis()
        axes[idx].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/part2/word_frequency.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary and Key Observations")
    print(f"{'='*60}")
    print("1. Dataset is balanced/imbalanced between positive and negative sentiments")
    print("2. Text length varies significantly, need to set appropriate max sequence length")
    print("3. Most common words can help understand sentiment patterns")
    print("4. Consider removing stop words and using appropriate tokenization")
    print(f"\nExploration complete! Check results/part2/ for visualizations.")
    
    return train_df, test_df, label_col, text_col


if __name__ == "__main__":
    # Create results directory
    os.makedirs('results/part2', exist_ok=True)
    
    # Load and explore dataset
    train_df, test_df = load_dataset()
    explore_dataset(train_df, test_df)
