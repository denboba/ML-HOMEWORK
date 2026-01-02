"""
Data loading utilities for image classification
"""

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ImageDataset(Dataset):
    """Custom dataset for loading images"""
    
    def __init__(self, root_dir, transform=None, augment=None):
        """
        Args:
            root_dir: Directory with all the images organized in class folders
            transform: Basic transformations (normalization, etc.)
            augment: Albumentations augmentation pipeline
        """
        self.root_dir = root_dir
        self.transform = transform
        self.augment = augment
        
        # Get all classes (folder names)
        self.classes = sorted([d for d in os.listdir(root_dir) 
                              if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Get all image paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Apply albumentations augmentation
        if self.augment is not None:
            augmented = self.augment(image=image)
            image = augmented['image']
        
        # Apply basic transforms
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label


def get_data_loaders(dataset_path, batch_size=32, image_size=96, 
                    use_augmentation=False, num_workers=4):
    """
    Create data loaders for train and test sets
    
    Args:
        dataset_path: Path to dataset directory
        batch_size: Batch size for data loaders
        image_size: Size to resize images to
        use_augmentation: Whether to use data augmentation
        num_workers: Number of worker processes for data loading
    
    Returns:
        train_loader, test_loader, num_classes
    """
    train_dir = os.path.join(dataset_path, 'train')
    test_dir = os.path.join(dataset_path, 'test')
    
    # Get number of classes
    classes = sorted([d for d in os.listdir(train_dir) 
                     if os.path.isdir(os.path.join(train_dir, d))])
    num_classes = len(classes)
    
    # Define augmentations using albumentations
    train_augment = None
    if use_augmentation:
        train_augment = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.CoarseDropout(max_holes=1, max_height=16, max_width=16, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    # Basic transforms (no augmentation)
    basic_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    if use_augmentation:
        train_dataset = ImageDataset(train_dir, transform=None, augment=train_augment)
    else:
        train_dataset = ImageDataset(train_dir, transform=basic_transform, augment=None)
    
    test_dataset = ImageDataset(test_dir, transform=test_transform, augment=None)
    
    # Create data loaders
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
    
    return train_loader, test_loader, num_classes
