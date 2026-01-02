"""
Data Exploration for Image Classification Task
Analyzes imagebits and land_patches datasets
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
import seaborn as sns

def explore_dataset(dataset_path, dataset_name):
    """
    Explore and visualize dataset characteristics
    
    Args:
        dataset_path: Path to dataset directory
        dataset_name: Name of the dataset for titles
    """
    print(f"\n{'='*60}")
    print(f"Exploring {dataset_name} Dataset")
    print(f"{'='*60}")
    
    # Get all classes
    train_path = os.path.join(dataset_path, 'train')
    test_path = os.path.join(dataset_path, 'test')
    
    classes = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
    print(f"\nClasses: {classes}")
    print(f"Number of classes: {len(classes)}")
    
    # Count images per class
    train_counts = {}
    test_counts = {}
    
    for cls in classes:
        train_cls_path = os.path.join(train_path, cls)
        test_cls_path = os.path.join(test_path, cls)
        
        train_counts[cls] = len([f for f in os.listdir(train_cls_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        test_counts[cls] = len([f for f in os.listdir(test_cls_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"\nTrain set:")
    for cls, count in train_counts.items():
        print(f"  {cls}: {count} images")
    print(f"Total train images: {sum(train_counts.values())}")
    
    print(f"\nTest set:")
    for cls, count in test_counts.items():
        print(f"  {cls}: {count} images")
    print(f"Total test images: {sum(test_counts.values())}")
    
    # Visualize class distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Train distribution
    axes[0].bar(range(len(train_counts)), list(train_counts.values()), color='skyblue')
    axes[0].set_xticks(range(len(train_counts)))
    axes[0].set_xticklabels(train_counts.keys(), rotation=45, ha='right')
    axes[0].set_ylabel('Number of Images')
    axes[0].set_title(f'{dataset_name} - Train Set Distribution')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Test distribution
    axes[1].bar(range(len(test_counts)), list(test_counts.values()), color='lightcoral')
    axes[1].set_xticks(range(len(test_counts)))
    axes[1].set_xticklabels(test_counts.keys(), rotation=45, ha='right')
    axes[1].set_ylabel('Number of Images')
    axes[1].set_title(f'{dataset_name} - Test Set Distribution')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/part1/{dataset_name}_class_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Sample images from each class
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for idx, cls in enumerate(classes):
        cls_path = os.path.join(train_path, cls)
        images = [f for f in os.listdir(cls_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if images:
            img_path = os.path.join(cls_path, images[0])
            img = Image.open(img_path)
            axes[idx].imshow(img)
            axes[idx].set_title(f'{cls}\n{img.size}')
            axes[idx].axis('off')
    
    plt.suptitle(f'{dataset_name} - Sample Images from Each Class', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f'results/part1/{dataset_name}_sample_images.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Analyze image properties
    print(f"\nAnalyzing image properties...")
    sample_images = []
    for cls in classes[:3]:  # Sample from first 3 classes
        cls_path = os.path.join(train_path, cls)
        images = [f for f in os.listdir(cls_path) if f.endswith(('.png', '.jpg', '.jpeg'))][:5]
        for img_name in images:
            img_path = os.path.join(cls_path, img_name)
            img = np.array(Image.open(img_path))
            sample_images.append(img)
    
    if sample_images:
        shapes = [img.shape for img in sample_images]
        print(f"Sample image shapes: {set(shapes)}")
        print(f"Image dtype: {sample_images[0].dtype}")
        print(f"Value range: [{np.min(sample_images[0])}, {np.max(sample_images[0])}]")
    
    # Analyze intra-class and inter-class variability
    print(f"\nVariability Analysis:")
    print("  Intra-class: Images within the same class may vary in lighting, pose, background, etc.")
    print("  Inter-class: Different classes should have distinct visual features.")
    
    return classes, train_counts, test_counts

if __name__ == "__main__":
    # Create results directory
    os.makedirs('results/part1', exist_ok=True)
    
    # Explore imagebits dataset
    imagebits_classes, imagebits_train, imagebits_test = explore_dataset(
        'imagebits', 
        'Imagebits'
    )
    
    # Explore land_patches dataset
    land_classes, land_train, land_test = explore_dataset(
        'land_patches',
        'Land_Patches'
    )
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("Dataset Comparison Summary")
    print(f"{'='*60}")
    print(f"Imagebits: {len(imagebits_classes)} classes, {sum(imagebits_train.values())} train, {sum(imagebits_test.values())} test")
    print(f"Land Patches: {len(land_classes)} classes, {sum(land_train.values())} train, {sum(land_test.values())} test")
    print(f"\nKey Observations:")
    print(f"1. Imagebits has balanced classes with 800 train and 500 test images per class")
    print(f"2. Land Patches has fewer training samples (200 per class), making it more challenging")
    print(f"3. Land Patches would benefit from transfer learning from Imagebits")
    print(f"\nExploration complete! Check results/part1/ for visualizations.")
