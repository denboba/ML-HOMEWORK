"""
RNN and LSTM models for sentiment analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleRNN(nn.Module):
    """
    Simple RNN model for sentiment analysis
    
    Architecture:
    - Embedding layer
    - RNN layers
    - Fully connected classifier
    """
    
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, 
                 num_layers=2, num_classes=2, dropout=0.5):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Hidden dimension of RNN
            num_layers: Number of RNN layers
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super(SimpleRNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # RNN layer
        self.rnn = nn.RNN(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Classifier
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # Embedding: (batch, seq_len) -> (batch, seq_len, embedding_dim)
        embedded = self.embedding(x)
        
        # RNN: (batch, seq_len, embedding_dim) -> (batch, seq_len, hidden_dim)
        rnn_out, hidden = self.rnn(embedded)
        
        # Use last hidden state
        # hidden: (num_layers, batch, hidden_dim)
        last_hidden = hidden[-1]  # (batch, hidden_dim)
        
        # Dropout and classification
        out = self.dropout(last_hidden)
        out = self.fc(out)
        
        return out


class LSTMModel(nn.Module):
    """
    LSTM model for sentiment analysis
    
    Architecture:
    - Embedding layer
    - LSTM layers (can be bidirectional)
    - Fully connected classifier
    """
    
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, 
                 num_layers=2, num_classes=2, dropout=0.5, bidirectional=False):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Classifier
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, num_classes)
    
    def forward(self, x):
        # Embedding: (batch, seq_len) -> (batch, seq_len, embedding_dim)
        embedded = self.embedding(x)
        
        # LSTM: (batch, seq_len, embedding_dim) -> (batch, seq_len, hidden_dim * num_directions)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use last hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            # hidden: (num_layers * 2, batch, hidden_dim)
            hidden_forward = hidden[-2]  # Forward direction
            hidden_backward = hidden[-1]  # Backward direction
            last_hidden = torch.cat([hidden_forward, hidden_backward], dim=1)
        else:
            # hidden: (num_layers, batch, hidden_dim)
            last_hidden = hidden[-1]
        
        # Dropout and classification
        out = self.dropout(last_hidden)
        out = self.fc(out)
        
        return out


class ImprovedLSTM(nn.Module):
    """
    Improved LSTM with attention mechanism
    """
    
    def __init__(self, vocab_size, embedding_dim=200, hidden_dim=256, 
                 num_layers=2, num_classes=2, dropout=0.3, bidirectional=True):
        super(ImprovedLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention layer
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = nn.Linear(lstm_output_dim, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Classifier with additional hidden layer
        self.fc1 = nn.Linear(lstm_output_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def attention_net(self, lstm_output):
        """
        Attention mechanism
        
        Args:
            lstm_output: (batch, seq_len, hidden_dim)
        
        Returns:
            output: (batch, hidden_dim)
        """
        # Calculate attention weights
        # (batch, seq_len, hidden_dim) -> (batch, seq_len, 1)
        attention_weights = torch.tanh(self.attention(lstm_output))
        
        # (batch, seq_len, 1) -> (batch, seq_len)
        attention_weights = attention_weights.squeeze(-1)
        
        # Softmax
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention weights
        # (batch, seq_len, 1) * (batch, seq_len, hidden_dim) -> (batch, seq_len, hidden_dim)
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_output)
        
        # (batch, 1, hidden_dim) -> (batch, hidden_dim)
        context = context.squeeze(1)
        
        return context
    
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Apply attention
        context = self.attention_net(lstm_out)
        
        # Classifier
        out = self.dropout(context)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


def get_rnn_model(model_type, vocab_size, num_classes=2, **kwargs):
    """
    Factory function to create RNN models
    
    Args:
        model_type: 'simple_rnn', 'lstm', 'lstm_bidirectional', or 'improved_lstm'
        vocab_size: Size of vocabulary
        num_classes: Number of output classes
        **kwargs: Additional model parameters
    
    Returns:
        RNN model
    """
    if model_type == 'simple_rnn':
        return SimpleRNN(vocab_size, num_classes=num_classes, **kwargs)
    elif model_type == 'lstm':
        return LSTMModel(vocab_size, num_classes=num_classes, bidirectional=False, **kwargs)
    elif model_type == 'lstm_bidirectional':
        return LSTMModel(vocab_size, num_classes=num_classes, bidirectional=True, **kwargs)
    elif model_type == 'improved_lstm':
        return ImprovedLSTM(vocab_size, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test models
    vocab_size = 10000
    batch_size = 4
    seq_length = 200
    
    # Create dummy input
    x = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # Test Simple RNN
    print("Testing Simple RNN...")
    model = SimpleRNN(vocab_size)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test LSTM
    print("\nTesting LSTM...")
    model = LSTMModel(vocab_size, bidirectional=False)
    output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test Bidirectional LSTM
    print("\nTesting Bidirectional LSTM...")
    model = LSTMModel(vocab_size, bidirectional=True)
    output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test Improved LSTM
    print("\nTesting Improved LSTM with Attention...")
    model = ImprovedLSTM(vocab_size)
    output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
