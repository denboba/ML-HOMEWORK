"""
Generate summary report for all experiments
"""

import os
import json
import pandas as pd
from pathlib import Path


def generate_summary_report():
    """Generate a comprehensive summary report"""
    
    print("\n" + "="*80)
    print("TEMA 2 - ÎNVĂȚARE AUTOMATĂ - SUMMARY REPORT")
    print("="*80)
    
    # Part 1: Image Classification
    print("\n" + "="*80)
    print("PART 1: IMAGE CLASSIFICATION")
    print("="*80)
    
    part1_results = []
    part1_dir = Path('results/part1')
    
    if part1_dir.exists():
        for exp_dir in part1_dir.iterdir():
            if exp_dir.is_dir():
                config_file = exp_dir / 'config.json'
                history_file = exp_dir / 'history.json'
                
                if config_file.exists() and history_file.exists():
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    with open(history_file, 'r') as f:
                        history = json.load(f)
                    
                    # Get best validation metrics
                    best_val_acc = max(history['val_acc'])
                    best_val_f1 = max(history['val_f1'])
                    
                    part1_results.append({
                        'Experiment': exp_dir.name,
                        'Dataset': config.get('dataset_path', 'N/A'),
                        'Model': config.get('model_arch', 'N/A'),
                        'Type': config.get('model_type', 'N/A'),
                        'Augmentation': 'Yes' if config.get('use_augmentation', False) else 'No',
                        'Best Val Acc (%)': f"{best_val_acc:.2f}",
                        'Best Val F1': f"{best_val_f1:.4f}",
                        'Epochs': config.get('epochs', 'N/A'),
                        'Batch Size': config.get('batch_size', 'N/A'),
                        'Learning Rate': config.get('learning_rate', 'N/A')
                    })
    
    if part1_results:
        df1 = pd.DataFrame(part1_results)
        print("\nImage Classification Results:")
        print(df1.to_string(index=False))
        
        # Save to CSV
        df1.to_csv('results/part1_summary.csv', index=False)
        print("\n✓ Part 1 summary saved to results/part1_summary.csv")
    else:
        print("\n⚠ No Part 1 results found. Run experiments first.")
    
    # Part 2: Sentiment Analysis
    print("\n" + "="*80)
    print("PART 2: SENTIMENT ANALYSIS")
    print("="*80)
    
    part2_results = []
    part2_dir = Path('results/part2')
    
    if part2_dir.exists():
        for exp_dir in part2_dir.iterdir():
            if exp_dir.is_dir():
                config_file = exp_dir / 'config.json'
                history_file = exp_dir / 'history.json'
                
                if config_file.exists() and history_file.exists():
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    with open(history_file, 'r') as f:
                        history = json.load(f)
                    
                    # Get best validation metrics
                    best_val_acc = max(history['val_acc'])
                    best_val_f1 = max(history['val_f1'])
                    
                    part2_results.append({
                        'Experiment': exp_dir.name,
                        'Model': config.get('model_type', 'N/A'),
                        'Vocab Size': config.get('vocab_size', 'N/A'),
                        'Max Length': config.get('max_length', 'N/A'),
                        'Embedding Dim': config.get('embedding_dim', 'N/A'),
                        'Hidden Dim': config.get('hidden_dim', 'N/A'),
                        'Best Val Acc (%)': f"{best_val_acc:.2f}",
                        'Best Val F1': f"{best_val_f1:.4f}",
                        'Epochs': config.get('epochs', 'N/A'),
                        'Batch Size': config.get('batch_size', 'N/A'),
                        'Learning Rate': config.get('learning_rate', 'N/A')
                    })
    
    if part2_results:
        df2 = pd.DataFrame(part2_results)
        print("\nSentiment Analysis Results:")
        print(df2.to_string(index=False))
        
        # Save to CSV
        df2.to_csv('results/part2_summary.csv', index=False)
        print("\n✓ Part 2 summary saved to results/part2_summary.csv")
    else:
        print("\n⚠ No Part 2 results found. Run experiments first.")
    
    # Overall Summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    
    print("\n📊 Visualizations and Detailed Results:")
    print("  - Part 1: results/part1/")
    print("    • Class distributions (Imagebits_class_distribution.png)")
    print("    • Sample images (Imagebits_sample_images.png)")
    print("    • Training curves for each model")
    print("    • Confusion matrices")
    
    print("\n  - Part 2: results/part2/")
    print("    • Sentiment distribution (sentiment_distribution.png)")
    print("    • Text length analysis (text_length_analysis.png)")
    print("    • Word frequency (word_frequency.png)")
    print("    • Training curves for each model")
    print("    • Confusion matrices")
    
    print("\n📝 Key Observations:")
    print("  1. Data Exploration completed for both parts")
    print("  2. Multiple model architectures implemented and trained")
    print("  3. Augmentation effects analyzed")
    print("  4. All results saved with configurations for reproducibility")
    
    print("\n✅ Summary report generation complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    generate_summary_report()
