"""
Run all experiments for Part 2 - Sentiment Analysis
"""

import os
import sys
from train import train_model


def run_all_experiments():
    """Run all required experiments"""
    
    experiments = [
        # Simple RNN
        {
            'name': 'Simple RNN',
            'config': {
                'train_path': 'data/ro_sent/train.csv',
                'test_path': 'data/ro_sent/test.csv',
                'model_type': 'simple_rnn',
                'vocab_size': 10000,
                'max_length': 200,
                'embedding_dim': 100,
                'hidden_dim': 128,
                'num_layers': 2,
                'dropout': 0.5,
                'batch_size': 64,
                'learning_rate': 0.001,
                'optimizer': 'adam',
                'epochs': 15,
                'num_workers': 2,
                'save_dir': 'results/part2/simple_rnn'
            }
        },
        # LSTM Unidirectional
        {
            'name': 'LSTM Unidirectional',
            'config': {
                'train_path': 'data/ro_sent/train.csv',
                'test_path': 'data/ro_sent/test.csv',
                'model_type': 'lstm',
                'vocab_size': 10000,
                'max_length': 200,
                'embedding_dim': 128,
                'hidden_dim': 128,
                'num_layers': 2,
                'dropout': 0.5,
                'batch_size': 64,
                'learning_rate': 0.001,
                'optimizer': 'adam',
                'epochs': 15,
                'num_workers': 2,
                'save_dir': 'results/part2/lstm_uni'
            }
        },
        # LSTM Bidirectional
        {
            'name': 'LSTM Bidirectional',
            'config': {
                'train_path': 'data/ro_sent/train.csv',
                'test_path': 'data/ro_sent/test.csv',
                'model_type': 'lstm_bidirectional',
                'vocab_size': 10000,
                'max_length': 200,
                'embedding_dim': 128,
                'hidden_dim': 128,
                'num_layers': 2,
                'dropout': 0.5,
                'batch_size': 64,
                'learning_rate': 0.001,
                'optimizer': 'adam',
                'epochs': 15,
                'num_workers': 2,
                'save_dir': 'results/part2/lstm_bi'
            }
        },
        # Improved LSTM with Attention
        {
            'name': 'Improved LSTM with Attention',
            'config': {
                'train_path': 'data/ro_sent/train.csv',
                'test_path': 'data/ro_sent/test.csv',
                'model_type': 'improved_lstm',
                'vocab_size': 15000,
                'max_length': 200,
                'embedding_dim': 200,
                'hidden_dim': 256,
                'num_layers': 2,
                'dropout': 0.3,
                'batch_size': 32,
                'learning_rate': 0.0005,
                'optimizer': 'adam',
                'epochs': 20,
                'num_workers': 2,
                'save_dir': 'results/part2/lstm_attention'
            }
        },
    ]
    
    results = []
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n{'='*70}")
        print(f"Running Experiment {i}/{len(experiments)}: {exp['name']}")
        print(f"{'='*70}")
        
        try:
            history, val_acc, val_f1 = train_model(exp['config'])
            results.append({
                'name': exp['name'],
                'config': exp['config'],
                'val_acc': val_acc,
                'val_f1': val_f1,
                'status': 'success'
            })
            print(f"\n✓ Experiment completed successfully!")
            print(f"  Val Accuracy: {val_acc:.2f}%")
            print(f"  Val F1: {val_f1:.4f}")
        except Exception as e:
            print(f"\n✗ Experiment failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append({
                'name': exp['name'],
                'config': exp['config'],
                'status': 'failed',
                'error': str(e)
            })
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY OF ALL EXPERIMENTS")
    print(f"{'='*70}")
    
    for result in results:
        print(f"\n{result['name']}")
        if result['status'] == 'success':
            print(f"  ✓ Val Acc: {result['val_acc']:.2f}% | Val F1: {result['val_f1']:.4f}")
        else:
            print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
    
    print(f"\n{'='*70}")
    print("All experiments completed!")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_all_experiments()
