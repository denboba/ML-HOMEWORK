"""
MLP (Multi-Layer Perceptron) model for image classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for image classification
    
    Architecture:
    - Flatten input images
    - Multiple fully connected layers with ReLU activation
    - Batch normalization and dropout for regularization
    - Final classification layer
    """
    
    def __init__(self, input_size=96*96*3, hidden_sizes=[512, 256, 128], 
                 num_classes=10, dropout=0.5):
        """
        Args:
            input_size: Size of flattened input (height * width * channels)
            hidden_sizes: List of hidden layer sizes
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super(MLP, self).__init__()
        
        self.input_size = input_size
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # Flatten input: (batch, channels, height, width) -> (batch, features)
        x = x.view(x.size(0), -1)
        return self.network(x)


class ImprovedMLP(nn.Module):
    """
    Improved MLP with residual connections
    """
    
    def __init__(self, input_size=96*96*3, hidden_sizes=[1024, 512, 256, 128], 
                 num_classes=10, dropout=0.3):
        super(ImprovedMLP, self).__init__()
        
        self.input_size = input_size
        
        # Input projection
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_sizes[i], hidden_sizes[i+1]),
                    nn.BatchNorm1d(hidden_sizes[i+1]),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], num_classes)
    
    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        
        # Input layer
        x = self.input_layer(x)
        
        # Hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
        
        # Output layer
        x = self.output_layer(x)
        return x


def get_mlp_model(model_type='basic', input_size=96*96*3, num_classes=10):
    """
    Factory function to create MLP models
    
    Args:
        model_type: 'basic' or 'improved'
        input_size: Size of flattened input
        num_classes: Number of output classes
    
    Returns:
        MLP model
    """
    if model_type == 'basic':
        return MLP(input_size=input_size, num_classes=num_classes)
    elif model_type == 'improved':
        return ImprovedMLP(input_size=input_size, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test the model
    model = MLP(input_size=96*96*3, num_classes=10)
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(4, 3, 96, 96)  # Batch of 4 images
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
