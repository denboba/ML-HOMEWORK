"""
Quick test to verify all components are working
"""

import sys
import os

def test_imports():
    """Test if all required libraries are installed"""
    print("\n" + "="*60)
    print("Testing imports...")
    print("="*60)
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch: {e}")
        return False
    
    try:
        import torchvision
        print(f"✓ TorchVision {torchvision.__version__}")
    except ImportError as e:
        print(f"✗ TorchVision: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✓ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"✗ Pandas: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib: {e}")
        return False
    
    try:
        import seaborn
        print(f"✓ Seaborn {seaborn.__version__}")
    except ImportError as e:
        print(f"✗ Seaborn: {e}")
        return False
    
    try:
        import sklearn
        print(f"✓ Scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"✗ Scikit-learn: {e}")
        return False
    
    try:
        import albumentations
        print(f"✓ Albumentations {albumentations.__version__}")
    except ImportError as e:
        print(f"✗ Albumentations: {e}")
        return False
    
    print("\n✅ All required libraries are installed!")
    return True


def test_datasets():
    """Test if datasets are available"""
    print("\n" + "="*60)
    print("Testing datasets...")
    print("="*60)
    
    all_ok = True
    
    # Check imagebits
    if os.path.exists('imagebits/train') and os.path.exists('imagebits/test'):
        train_count = sum([len(files) for r, d, files in os.walk('imagebits/train')])
        test_count = sum([len(files) for r, d, files in os.walk('imagebits/test')])
        print(f"✓ Imagebits dataset: {train_count} train, {test_count} test images")
    else:
        print("✗ Imagebits dataset not found")
        all_ok = False
    
    # Check land_patches
    if os.path.exists('land_patches/train') and os.path.exists('land_patches/test'):
        train_count = sum([len(files) for r, d, files in os.walk('land_patches/train')])
        test_count = sum([len(files) for r, d, files in os.walk('land_patches/test')])
        print(f"✓ Land Patches dataset: {train_count} train, {test_count} test images")
    else:
        print("✗ Land Patches dataset not found")
        all_ok = False
    
    # Check ro_sent
    if os.path.exists('data/ro_sent/train.csv') and os.path.exists('data/ro_sent/test.csv'):
        import pandas as pd
        train_df = pd.read_csv('data/ro_sent/train.csv')
        test_df = pd.read_csv('data/ro_sent/test.csv')
        print(f"✓ Romanian Sentiment dataset: {len(train_df)} train, {len(test_df)} test samples")
    else:
        print("✗ Romanian Sentiment dataset not found")
        all_ok = False
    
    if all_ok:
        print("\n✅ All datasets are available!")
    else:
        print("\n⚠ Some datasets are missing!")
    
    return all_ok


def test_models():
    """Test if models can be instantiated"""
    print("\n" + "="*60)
    print("Testing models...")
    print("="*60)
    
    all_ok = True
    
    # Test MLP
    try:
        from src.part1_image_classification.mlp_model import MLP
        model = MLP(input_size=96*96*3, num_classes=10)
        print(f"✓ MLP model: {sum(p.numel() for p in model.parameters()):,} parameters")
    except Exception as e:
        print(f"✗ MLP model: {e}")
        all_ok = False
    
    # Test CNN
    try:
        from src.part1_image_classification.cnn_model import CNN
        model = CNN(num_classes=10)
        print(f"✓ CNN model: {sum(p.numel() for p in model.parameters()):,} parameters")
    except Exception as e:
        print(f"✗ CNN model: {e}")
        all_ok = False
    
    # Test RNN
    try:
        from src.part2_sentiment_analysis.rnn_models import SimpleRNN
        model = SimpleRNN(vocab_size=10000, num_classes=2)
        print(f"✓ Simple RNN model: {sum(p.numel() for p in model.parameters()):,} parameters")
    except Exception as e:
        print(f"✗ Simple RNN model: {e}")
        all_ok = False
    
    # Test LSTM
    try:
        from src.part2_sentiment_analysis.rnn_models import LSTMModel
        model = LSTMModel(vocab_size=10000, num_classes=2, bidirectional=True)
        print(f"✓ LSTM model: {sum(p.numel() for p in model.parameters()):,} parameters")
    except Exception as e:
        print(f"✗ LSTM model: {e}")
        all_ok = False
    
    if all_ok:
        print("\n✅ All models can be instantiated!")
    else:
        print("\n⚠ Some models failed to instantiate!")
    
    return all_ok


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("HOMEWORK IMPLEMENTATION VERIFICATION")
    print("="*60)
    
    imports_ok = test_imports()
    datasets_ok = test_datasets()
    models_ok = test_models()
    
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    print(f"Imports: {'✅' if imports_ok else '❌'}")
    print(f"Datasets: {'✅' if datasets_ok else '❌'}")
    print(f"Models: {'✅' if models_ok else '❌'}")
    
    if imports_ok and datasets_ok and models_ok:
        print("\n✅ ALL CHECKS PASSED! Implementation is ready.")
        print("\nNext steps:")
        print("  1. Run data exploration: python src/part1_image_classification/data_exploration.py")
        print("  2. Run Part 1 experiments: python src/part1_image_classification/run_experiments.py")
        print("  3. Run Part 2 experiments: python src/part2_sentiment_analysis/run_experiments.py")
        print("  4. Generate report: python generate_report.py")
        return 0
    else:
        print("\n❌ SOME CHECKS FAILED! Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
