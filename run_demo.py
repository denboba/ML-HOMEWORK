"""
Complete workflow example - Run a quick demo of the entire pipeline
This runs a shortened version of each component for demonstration
"""

import os
import sys

def run_demo():
    """Run a complete demo of all components"""
    
    print("\n" + "="*70)
    print("TEMA 2 - MACHINE LEARNING HOMEWORK - DEMO")
    print("="*70)
    print("\nThis demo will run a shortened version of the complete pipeline.")
    print("For full training, use the individual run_experiments.py scripts.")
    
    input("\nPress Enter to continue...")
    
    # Step 1: Verify implementation
    print("\n" + "="*70)
    print("STEP 1: Verifying Implementation")
    print("="*70)
    
    import verify_implementation
    if verify_implementation.main() != 0:
        print("\n❌ Verification failed! Please fix the issues.")
        return 1
    
    # Step 2: Data Exploration Part 1
    print("\n" + "="*70)
    print("STEP 2: Data Exploration - Image Classification")
    print("="*70)
    
    try:
        import src.part1_image_classification.data_exploration as exp1
        print("✓ Data exploration completed!")
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1
    
    # Step 3: Data Exploration Part 2
    print("\n" + "="*70)
    print("STEP 3: Data Exploration - Sentiment Analysis")
    print("="*70)
    
    try:
        import src.part2_sentiment_analysis.data_exploration as exp2
        print("✓ Data exploration completed!")
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1
    
    # Step 4: Quick Model Training Demo (Part 1)
    print("\n" + "="*70)
    print("STEP 4: Quick Training Demo - Image Classification")
    print("="*70)
    print("\nTraining a small CNN model on Imagebits (5 epochs - demo only)...")
    
    try:
        from src.part1_image_classification.train import train_model
        
        config = {
            'dataset_path': 'imagebits',
            'model_arch': 'cnn',
            'model_type': 'basic',
            'batch_size': 64,
            'image_size': 96,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'weight_decay': 0.0001,
            'epochs': 5,  # Short demo
            'use_augmentation': False,
            'num_workers': 2,
            'save_dir': 'results/demo/cnn_demo'
        }
        
        history, val_acc, val_f1 = train_model(config)
        print(f"\n✓ Demo training completed!")
        print(f"  Final Val Accuracy: {val_acc:.2f}%")
        print(f"  Final Val F1: {val_f1:.4f}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 5: Quick Model Training Demo (Part 2)
    print("\n" + "="*70)
    print("STEP 5: Quick Training Demo - Sentiment Analysis")
    print("="*70)
    print("\nTraining a simple RNN model (5 epochs - demo only)...")
    
    try:
        from src.part2_sentiment_analysis.train import train_model as train_model_sent
        
        config = {
            'train_path': 'data/ro_sent/train.csv',
            'test_path': 'data/ro_sent/test.csv',
            'model_type': 'simple_rnn',
            'vocab_size': 5000,  # Smaller vocab for demo
            'max_length': 150,
            'embedding_dim': 50,  # Smaller embedding
            'hidden_dim': 64,     # Smaller hidden dim
            'num_layers': 1,      # Fewer layers
            'dropout': 0.5,
            'batch_size': 128,    # Larger batch for speed
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'epochs': 5,          # Short demo
            'num_workers': 2,
            'save_dir': 'results/demo/rnn_demo'
        }
        
        history, val_acc, val_f1 = train_model_sent(config)
        print(f"\n✓ Demo training completed!")
        print(f"  Final Val Accuracy: {val_acc:.2f}%")
        print(f"  Final Val F1: {val_f1:.4f}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 6: Summary
    print("\n" + "="*70)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    print("\n📊 Results saved to:")
    print("  - results/demo/cnn_demo/")
    print("  - results/demo/rnn_demo/")
    
    print("\n📝 Next Steps:")
    print("  1. Review the demo results in results/demo/")
    print("  2. Run full experiments:")
    print("     - python src/part1_image_classification/run_experiments.py")
    print("     - python src/part2_sentiment_analysis/run_experiments.py")
    print("  3. Generate comprehensive report:")
    print("     - python generate_report.py")
    
    print("\n✅ All components are working correctly!")
    print("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(run_demo())
    except KeyboardInterrupt:
        print("\n\n⚠ Demo interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
