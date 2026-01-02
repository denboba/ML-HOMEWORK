"""
Run all experiments for Part 1 - Image Classification
"""

import os
import sys
from train import train_model


def run_all_experiments():
    """Run all required experiments"""
    
    experiments = [
        # MLP on Imagebits
        {
            'name': 'MLP Basic on Imagebits',
            'config': {
                'dataset_path': 'imagebits',
                'model_arch': 'mlp',
                'model_type': 'basic',
                'batch_size': 64,
                'image_size': 96,
                'learning_rate': 0.001,
                'optimizer': 'adam',
                'weight_decay': 0.0001,
                'epochs': 30,
                'use_augmentation': False,
                'num_workers': 2,
                'save_dir': 'results/part1/mlp_imagebits_basic'
            }
        },
        # MLP on Imagebits with augmentation
        {
            'name': 'MLP Basic on Imagebits (with augmentation)',
            'config': {
                'dataset_path': 'imagebits',
                'model_arch': 'mlp',
                'model_type': 'basic',
                'batch_size': 64,
                'image_size': 96,
                'learning_rate': 0.001,
                'optimizer': 'adam',
                'weight_decay': 0.0001,
                'epochs': 30,
                'use_augmentation': True,
                'num_workers': 2,
                'save_dir': 'results/part1/mlp_imagebits_aug'
            }
        },
        # CNN on Imagebits
        {
            'name': 'CNN Basic on Imagebits',
            'config': {
                'dataset_path': 'imagebits',
                'model_arch': 'cnn',
                'model_type': 'basic',
                'batch_size': 64,
                'image_size': 96,
                'learning_rate': 0.001,
                'optimizer': 'adam',
                'weight_decay': 0.0001,
                'epochs': 40,
                'use_augmentation': False,
                'num_workers': 2,
                'save_dir': 'results/part1/cnn_imagebits_basic'
            }
        },
        # CNN on Imagebits with augmentation
        {
            'name': 'CNN Basic on Imagebits (with augmentation)',
            'config': {
                'dataset_path': 'imagebits',
                'model_arch': 'cnn',
                'model_type': 'basic',
                'batch_size': 64,
                'image_size': 96,
                'learning_rate': 0.001,
                'optimizer': 'adam',
                'weight_decay': 0.0001,
                'epochs': 40,
                'use_augmentation': True,
                'num_workers': 2,
                'save_dir': 'results/part1/cnn_imagebits_aug'
            }
        },
        # Improved CNN on Imagebits with augmentation
        {
            'name': 'CNN Improved on Imagebits (with augmentation)',
            'config': {
                'dataset_path': 'imagebits',
                'model_arch': 'cnn',
                'model_type': 'improved',
                'batch_size': 32,
                'image_size': 96,
                'learning_rate': 0.0005,
                'optimizer': 'adam',
                'weight_decay': 0.0001,
                'epochs': 50,
                'use_augmentation': True,
                'num_workers': 2,
                'save_dir': 'results/part1/cnn_imagebits_improved'
            }
        },
        # MLP on Land Patches
        {
            'name': 'MLP Basic on Land Patches',
            'config': {
                'dataset_path': 'land_patches',
                'model_arch': 'mlp',
                'model_type': 'basic',
                'batch_size': 32,
                'image_size': 64,
                'learning_rate': 0.001,
                'optimizer': 'adam',
                'weight_decay': 0.0001,
                'epochs': 30,
                'use_augmentation': False,
                'num_workers': 2,
                'save_dir': 'results/part1/mlp_landpatches_basic'
            }
        },
        # CNN on Land Patches
        {
            'name': 'CNN Basic on Land Patches',
            'config': {
                'dataset_path': 'land_patches',
                'model_arch': 'cnn',
                'model_type': 'basic',
                'batch_size': 32,
                'image_size': 64,
                'learning_rate': 0.001,
                'optimizer': 'adam',
                'weight_decay': 0.0001,
                'epochs': 40,
                'use_augmentation': False,
                'num_workers': 2,
                'save_dir': 'results/part1/cnn_landpatches_basic'
            }
        },
        # CNN on Land Patches with augmentation
        {
            'name': 'CNN Basic on Land Patches (with augmentation)',
            'config': {
                'dataset_path': 'land_patches',
                'model_arch': 'cnn',
                'model_type': 'basic',
                'batch_size': 32,
                'image_size': 64,
                'learning_rate': 0.001,
                'optimizer': 'adam',
                'weight_decay': 0.0001,
                'epochs': 40,
                'use_augmentation': True,
                'num_workers': 2,
                'save_dir': 'results/part1/cnn_landpatches_aug'
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
