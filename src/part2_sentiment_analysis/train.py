"""
Training script for sentiment analysis models
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

from text_preprocessing import load_and_preprocess_data, get_data_loaders
from rnn_models import get_rnn_model


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for sequences, labels in tqdm(train_loader, desc='Training', leave=False):
        sequences, labels = sequences.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        running_loss += loss.item() * sequences.size(0)
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
        for sequences, labels in tqdm(test_loader, desc='Evaluating', leave=False):
            sequences, labels = sequences.to(device), labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * sequences.size(0)
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
    
    plt.figure(figsize=(8, 6))
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
    
    # Load and preprocess data
    print(f"\nLoading and preprocessing data...")
    train_dataset, test_dataset, tokenizer, num_classes = load_and_preprocess_data(
        config['train_path'],
        config['test_path'],
        vocab_size=config['vocab_size'],
        max_length=config['max_length']
    )
    
    # Save tokenizer
    tokenizer.save(os.path.join(config['save_dir'], 'tokenizer.pkl'))
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders(
        train_dataset,
        test_dataset,
        batch_size=config['batch_size'],
        num_workers=config.get('num_workers', 2)
    )
    
    print(f"Vocabulary size: {len(tokenizer.word2idx)}")
    print(f"Number of classes: {num_classes}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    print(f"\nCreating {config['model_type']} model...")
    model = get_rnn_model(
        config['model_type'],
        vocab_size=len(tokenizer.word2idx),
        num_classes=num_classes,
        embedding_dim=config.get('embedding_dim', 100),
        hidden_dim=config.get('hidden_dim', 128),
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.5)
    )
    
    model = model.to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    elif config['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
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
    
    # Final evaluation
    print("\nFinal evaluation on test set...")
    model.load_state_dict(torch.load(os.path.join(config['save_dir'], 'best_model.pth'))['model_state_dict'])
    val_loss, val_acc, val_f1, predictions, labels = evaluate(model, test_loader, criterion, device)
    
    print(f"Best Model - Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f}")
    
    # Confusion matrix
    class_names = ['Negative', 'Positive']
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
        'epochs': 20,
        'num_workers': 2,
        'save_dir': 'results/part2/simple_rnn'
    }
    
    train_model(config)
