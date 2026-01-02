"""
Training script for image classification models
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns

from data_loader import get_data_loaders
from mlp_model import get_mlp_model
from cnn_model import get_cnn_model


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(train_loader, desc='Training', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion, device):
    """Evaluate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return epoch_loss, epoch_acc, f1, all_preds, all_labels


def plot_training_history(history, save_path):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(labels, predictions, class_names, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def train_model(config):
    """
    Main training function
    
    Args:
        config: Dictionary with training configuration
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create results directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Get data loaders
    print(f"\nLoading data from {config['dataset_path']}...")
    train_loader, test_loader, num_classes = get_data_loaders(
        config['dataset_path'],
        batch_size=config['batch_size'],
        image_size=config['image_size'],
        use_augmentation=config['use_augmentation'],
        num_workers=config.get('num_workers', 4)
    )
    
    # Get class names
    train_dir = os.path.join(config['dataset_path'], 'train')
    class_names = sorted([d for d in os.listdir(train_dir) 
                         if os.path.isdir(os.path.join(train_dir, d))])
    
    print(f"Number of classes: {num_classes}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    print(f"\nCreating {config['model_arch']} model (type: {config['model_type']})...")
    if config['model_arch'] == 'mlp':
        input_size = config['image_size'] * config['image_size'] * 3
        model = get_mlp_model(
            model_type=config['model_type'],
            input_size=input_size,
            num_classes=num_classes
        )
    elif config['model_arch'] == 'cnn':
        model = get_cnn_model(
            model_type=config['model_type'],
            num_classes=num_classes,
            input_channels=3
        )
    else:
        raise ValueError(f"Unknown architecture: {config['model_arch']}")
    
    model = model.to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], 
                              weight_decay=config.get('weight_decay', 0.0))
    elif config['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], 
                             momentum=0.9, weight_decay=config.get('weight_decay', 0.0))
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    
    best_val_acc = 0.0
    
    # Training loop
    print(f"\nStarting training for {config['epochs']} epochs...")
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        val_loss, val_acc, val_f1, _, _ = evaluate(model, test_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Scheduler step
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'config': config
            }, os.path.join(config['save_dir'], 'best_model.pth'))
            print(f"Saved best model with val_acc: {val_acc:.2f}%")
    
    # Save final model
    torch.save({
        'epoch': config['epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'config': config
    }, os.path.join(config['save_dir'], 'final_model.pth'))
    
    # Plot training history
    plot_training_history(
        history,
        os.path.join(config['save_dir'], 'training_history.png')
    )
    
    # Final evaluation with confusion matrix
    print("\nFinal evaluation on test set...")
    model.load_state_dict(torch.load(os.path.join(config['save_dir'], 'best_model.pth'))['model_state_dict'])
    val_loss, val_acc, val_f1, predictions, labels = evaluate(model, test_loader, criterion, device)
    
    print(f"Best Model - Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f}")
    
    # Confusion matrix
    plot_confusion_matrix(
        labels,
        predictions,
        class_names,
        os.path.join(config['save_dir'], 'confusion_matrix.png')
    )
    
    # Classification report
    report = classification_report(labels, predictions, target_names=class_names)
    with open(os.path.join(config['save_dir'], 'classification_report.txt'), 'w') as f:
        f.write(report)
    print("\nClassification Report:")
    print(report)
    
    # Save history
    with open(os.path.join(config['save_dir'], 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save config
    with open(os.path.join(config['save_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nTraining complete! Results saved to {config['save_dir']}")
    
    return history, val_acc, val_f1


if __name__ == "__main__":
    # Example configuration
    config = {
        'dataset_path': 'imagebits',
        'model_arch': 'mlp',
        'model_type': 'basic',
        'batch_size': 64,
        'image_size': 96,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'weight_decay': 0.0001,
        'epochs': 50,
        'use_augmentation': False,
        'num_workers': 4,
        'save_dir': 'results/part1/mlp_imagebits_basic'
    }
    
    train_model(config)
