"""RNN-based network architectures for QPLEX algorithm."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from .base_networks import MLP, RNN, Attention


class RNNQNetwork(nn.Module):
    """RNN-based Q-network for individual agents with memory."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, rnn_type: str = 'lstm', dropout: float = 0.0):
        super(RNNQNetwork, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        
        # Input projection
        self.input_proj = nn.Linear(obs_dim, hidden_dim)
        
        # RNN layer
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers, 
                              dropout=dropout if num_layers > 1 else 0,
                              batch_first=True)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers,
                             dropout=dropout if num_layers > 1 else 0,
                             batch_first=True)
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        # Output layers
        self.q_head = nn.Linear(hidden_dim, action_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, obs: torch.Tensor, hidden: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        """
        Args:
            obs: Observation tensor of shape (batch_size, seq_len, obs_dim) or (batch_size, obs_dim)
            hidden: Hidden state tuple (h_0, c_0) for LSTM or h_0 for GRU
        
        Returns:
            q_values: Q-values of shape (batch_size, seq_len, action_dim) or (batch_size, action_dim)
            hidden: Updated hidden state
        """
        if len(obs.shape) == 2:
            # Single timestep
            obs = obs.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Input projection
        x = self.input_proj(obs)
        x = self.dropout(x)
        
        # RNN forward pass
        rnn_out, hidden = self.rnn(x, hidden)
        
        # Q-value prediction
        q_values = self.q_head(rnn_out)
        
        if squeeze_output:
            q_values = q_values.squeeze(1)
        
        return q_values, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple:
        """Initialize hidden state."""
        if self.rnn_type == 'lstm':
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
            return (h_0, c_0)
        else:  # GRU
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
            return (h_0,)


class AttentionRNNQNetwork(nn.Module):
    """RNN-based Q-network with attention mechanism."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, rnn_type: str = 'lstm', num_heads: int = 4,
                 dropout: float = 0.0):
        super(AttentionRNNQNetwork, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        
        # Input projection
        self.input_proj = nn.Linear(obs_dim, hidden_dim)
        
        # RNN layer
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers, 
                              dropout=dropout if num_layers > 1 else 0,
                              batch_first=True)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers,
                             dropout=dropout if num_layers > 1 else 0,
                             batch_first=True)
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        # Attention mechanism
        self.attention = Attention(hidden_dim, num_heads, dropout)
        
        # Output layers
        self.q_head = nn.Linear(hidden_dim, action_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, obs: torch.Tensor, hidden: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        """
        Args:
            obs: Observation tensor of shape (batch_size, seq_len, obs_dim) or (batch_size, obs_dim)
            hidden: Hidden state tuple (h_0, c_0) for LSTM or h_0 for GRU
        
        Returns:
            q_values: Q-values of shape (batch_size, seq_len, action_dim) or (batch_size, action_dim)
            hidden: Updated hidden state
        """
        if len(obs.shape) == 2:
            # Single timestep
            obs = obs.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Input projection
        x = self.input_proj(obs)
        x = self.dropout(x)
        
        # RNN forward pass
        rnn_out, hidden = self.rnn(x, hidden)
        
        # Self-attention
        attn_out = self.attention(rnn_out, rnn_out, rnn_out)
        
        # Residual connection
        rnn_out = rnn_out + attn_out
        
        # Q-value prediction
        q_values = self.q_head(rnn_out)
        
        if squeeze_output:
            q_values = q_values.squeeze(1)
        
        return q_values, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple:
        """Initialize hidden state."""
        if self.rnn_type == 'lstm':
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
            return (h_0, c_0)
        else:  # GRU
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
            return (h_0,)


class BiRNNQNetwork(nn.Module):
    """Bidirectional RNN-based Q-network."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, rnn_type: str = 'lstm', dropout: float = 0.0):
        super(BiRNNQNetwork, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        
        # Input projection
        self.input_proj = nn.Linear(obs_dim, hidden_dim)
        
        # Bidirectional RNN layer
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers, 
                              dropout=dropout if num_layers > 1 else 0,
                              batch_first=True, bidirectional=True)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers,
                             dropout=dropout if num_layers > 1 else 0,
                             batch_first=True, bidirectional=True)
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        # Output projection (from bidirectional output)
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Q-value head
        self.q_head = nn.Linear(hidden_dim, action_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, obs: torch.Tensor, hidden: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        """
        Args:
            obs: Observation tensor of shape (batch_size, seq_len, obs_dim) or (batch_size, obs_dim)
            hidden: Hidden state tuple (h_0, c_0) for LSTM or h_0 for GRU
        
        Returns:
            q_values: Q-values of shape (batch_size, seq_len, action_dim) or (batch_size, action_dim)
            hidden: Updated hidden state
        """
        if len(obs.shape) == 2:
            # Single timestep
            obs = obs.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Input projection
        x = self.input_proj(obs)
        x = self.dropout(x)
        
        # Bidirectional RNN forward pass
        rnn_out, hidden = self.rnn(x, hidden)
        
        # Project back to original hidden dimension
        rnn_out = self.output_proj(rnn_out)
        
        # Q-value prediction
        q_values = self.q_head(rnn_out)
        
        if squeeze_output:
            q_values = q_values.squeeze(1)
        
        return q_values, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple:
        """Initialize hidden state for bidirectional RNN."""
        if self.rnn_type == 'lstm':
            h_0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim, device=device)
            c_0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim, device=device)
            return (h_0, c_0)
        else:  # GRU
            h_0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim, device=device)
            return (h_0,)


class HierarchicalRNNQNetwork(nn.Module):
    """Hierarchical RNN-based Q-network with different time scales."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, rnn_type: str = 'lstm', dropout: float = 0.0):
        super(HierarchicalRNNQNetwork, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        
        # Input projection
        self.input_proj = nn.Linear(obs_dim, hidden_dim)
        
        # Fast RNN (short-term memory)
        if self.rnn_type == 'lstm':
            self.fast_rnn = nn.LSTM(hidden_dim, hidden_dim // 2, num_layers, 
                                   dropout=dropout if num_layers > 1 else 0,
                                   batch_first=True)
            self.slow_rnn = nn.LSTM(hidden_dim // 2, hidden_dim // 2, num_layers,
                                   dropout=dropout if num_layers > 1 else 0,
                                   batch_first=True)
        elif self.rnn_type == 'gru':
            self.fast_rnn = nn.GRU(hidden_dim, hidden_dim // 2, num_layers,
                                  dropout=dropout if num_layers > 1 else 0,
                                  batch_first=True)
            self.slow_rnn = nn.GRU(hidden_dim // 2, hidden_dim // 2, num_layers,
                                  dropout=dropout if num_layers > 1 else 0,
                                  batch_first=True)
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        # Fusion layer
        self.fusion = nn.Linear(hidden_dim, hidden_dim)
        
        # Q-value head
        self.q_head = nn.Linear(hidden_dim, action_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, obs: torch.Tensor, hidden: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        """
        Args:
            obs: Observation tensor of shape (batch_size, seq_len, obs_dim) or (batch_size, obs_dim)
            hidden: Hidden state tuple (fast_hidden, slow_hidden)
        
        Returns:
            q_values: Q-values of shape (batch_size, seq_len, action_dim) or (batch_size, action_dim)
            hidden: Updated hidden state
        """
        if len(obs.shape) == 2:
            # Single timestep
            obs = obs.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Input projection
        x = self.input_proj(obs)
        x = self.dropout(x)
        
        # Separate hidden states
        if hidden is None:
            fast_hidden = None
            slow_hidden = None
        else:
            fast_hidden, slow_hidden = hidden
        
        # Fast RNN (processes every timestep)
        fast_out, fast_hidden = self.fast_rnn(x, fast_hidden)
        
        # Slow RNN (processes every other timestep)
        if x.size(1) > 1:
            slow_input = fast_out[:, ::2, :]  # Take every other timestep
            slow_out, slow_hidden = self.slow_rnn(slow_input, slow_hidden)
            
            # Upsample slow output to match fast output
            slow_out_upsampled = F.interpolate(
                slow_out.transpose(1, 2), 
                size=x.size(1), 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
        else:
            slow_out_upsampled = fast_out
            slow_hidden = fast_hidden
        
        # Fusion
        combined = torch.cat([fast_out, slow_out_upsampled], dim=-1)
        fused = self.fusion(combined)
        
        # Q-value prediction
        q_values = self.q_head(fused)
        
        if squeeze_output:
            q_values = q_values.squeeze(1)
        
        return q_values, (fast_hidden, slow_hidden)
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple:
        """Initialize hidden state."""
        if self.rnn_type == 'lstm':
            fast_h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim // 2, device=device)
            fast_c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim // 2, device=device)
            slow_h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim // 2, device=device)
            slow_c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim // 2, device=device)
            return ((fast_h_0, fast_c_0), (slow_h_0, slow_c_0))
        else:  # GRU
            fast_h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim // 2, device=device)
            slow_h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim // 2, device=device)
            return ((fast_h_0,), (slow_h_0,))
