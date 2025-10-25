"""Base network architectures for QPLEX algorithm."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class MLP(nn.Module):
    """Multi-layer perceptron network."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, 
                 activation: str = 'relu', dropout: float = 0.0):
        super(MLP, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class RNN(nn.Module):
    """Recurrent neural network with LSTM or GRU."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1,
                 rnn_type: str = 'lstm', dropout: float = 0.0):
        super(RNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, 
                              dropout=dropout if num_layers > 1 else 0,
                              batch_first=True)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers,
                             dropout=dropout if num_layers > 1 else 0,
                             batch_first=True)
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
    
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            hidden: Hidden state tuple (h_0, c_0) for LSTM or h_0 for GRU
        
        Returns:
            output: Output tensor of shape (batch_size, seq_len, hidden_dim)
            hidden: Updated hidden state
        """
        return self.rnn(x, hidden)
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple:
        """Initialize hidden state."""
        if self.rnn_type == 'lstm':
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
            return (h_0, c_0)
        else:  # GRU
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
            return (h_0,)


class Attention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, input_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super(Attention, self).__init__()
        
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        self.q_linear = nn.Linear(input_dim, input_dim)
        self.k_linear = nn.Linear(input_dim, input_dim)
        self.v_linear = nn.Linear(input_dim, input_dim)
        self.out_linear = nn.Linear(input_dim, input_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: Query tensor of shape (batch_size, seq_len, input_dim)
            key: Key tensor of shape (batch_size, seq_len, input_dim)
            value: Value tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional mask tensor of shape (batch_size, seq_len, seq_len)
        
        Returns:
            output: Attention output of shape (batch_size, seq_len, input_dim)
        """
        batch_size, seq_len, _ = query.size()
        
        # Linear projections
        Q = self.q_linear(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.input_dim
        )
        
        # Final linear projection
        output = self.out_linear(attn_output)
        
        return output


class QNetwork(nn.Module):
    """Q-network for individual agents."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256],
                 use_rnn: bool = False, rnn_hidden_dim: int = 128, rnn_layers: int = 1,
                 use_attention: bool = False, num_attention_heads: int = 4):
        super(QNetwork, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.use_rnn = use_rnn
        self.use_attention = use_attention
        
        # Feature extraction
        if use_rnn:
            self.rnn = RNN(obs_dim, rnn_hidden_dim, rnn_layers)
            feature_dim = rnn_hidden_dim
        else:
            self.feature_extractor = MLP(obs_dim, hidden_dims[:-1], hidden_dims[-1])
            feature_dim = hidden_dims[-1]
        
        # Attention mechanism
        if use_attention:
            self.attention = Attention(feature_dim, num_attention_heads)
        
        # Q-value head
        self.q_head = nn.Linear(feature_dim, action_dim)
    
    def forward(self, obs: torch.Tensor, hidden: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        """
        Args:
            obs: Observation tensor of shape (batch_size, seq_len, obs_dim) or (batch_size, obs_dim)
            hidden: Hidden state for RNN
        
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
        
        if self.use_rnn:
            features, hidden = self.rnn(obs, hidden)
        else:
            features = self.feature_extractor(obs)
            hidden = None
        
        if self.use_attention:
            features = self.attention(features, features, features)
        
        q_values = self.q_head(features)
        
        if squeeze_output:
            q_values = q_values.squeeze(1)
        
        return q_values, hidden


class MixingNetwork(nn.Module):
    """Mixing network for QPLEX algorithm."""
    
    def __init__(self, state_dim: int, n_agents: int, hidden_dims: List[int] = [256, 256],
                 use_hypernet: bool = True, hidden_dim=32):
        super(MixingNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.use_hypernet = use_hypernet
        self.hidden_dim = hidden_dim

        
        if use_hypernet:
            # Hypernetwork approach
            self.hyper_w1 = nn.Linear(state_dim, hidden_dim * n_agents)
            self.hyper_w2 = nn.Linear(state_dim, hidden_dim)
            self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
            self.hyper_b2 = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )            
            self.hyper_w_final = nn.Linear(state_dim, hidden_dims[1])
        else:
            # Standard mixing network
            self.mixing_net = MLP(state_dim, hidden_dims, 1)
    
    def forward(self, q_values, state):
        batch_size = q_values.size(0)
        n_agents = q_values.size(1)
        hidden_dim = self.hidden_dim

        # Nếu q_values là 2D → thêm chiều giữa
        if q_values.dim() == 2:
            q_values = q_values.unsqueeze(1)  # [B, 1, n_agents]

        # Tạo weight và bias
        w1 = torch.abs(self.hyper_w1(state))
        b1 = self.hyper_b1(state)

        # Reshape
        w1 = w1.view(batch_size, n_agents, hidden_dim)
        b1 = b1.view(batch_size, 1, hidden_dim)

        # Tầng 1
        hidden = F.elu(torch.bmm(q_values, w1) + b1)

        # Tầng 2
        w2 = torch.abs(self.hyper_w2(state)).view(batch_size, hidden_dim, 1)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)
        # print(">>> q_values:", q_values.shape)
        # print(">>> w1:", w1.shape)

        q_total = torch.bmm(hidden, w2) + b2
        return q_total.view(batch_size, -1)



class QPLEXNetwork(nn.Module):
    """Complete QPLEX network combining individual Q-networks and mixing network."""
    
    def __init__(self, obs_dim: int, action_dim: int, state_dim: int, n_agents: int,
                 q_hidden_dims: List[int] = [256, 256], mix_hidden_dims: List[int] = [256, 256],
                 use_rnn: bool = False, rnn_hidden_dim: int = 128, rnn_layers: int = 1,
                 use_attention: bool = False, num_attention_heads: int = 4,
                 use_hypernet: bool = True):
        super(QPLEXNetwork, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.use_rnn = use_rnn
        
        # Individual Q-networks
        self.q_networks = nn.ModuleList([
            QNetwork(obs_dim, action_dim, q_hidden_dims, use_rnn, rnn_hidden_dim, rnn_layers,
                    use_attention, num_attention_heads)
            for _ in range(n_agents)
        ])
        
        # Mixing network
        self.mixing_net = MixingNetwork(state_dim, n_agents, mix_hidden_dims, use_hypernet)
    
    def forward(self, obs: torch.Tensor, state: torch.Tensor, 
                hidden: Optional[List[Tuple]] = None) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple]]:
        """
        Args:
            obs: Observations of shape (batch_size, n_agents, obs_dim) or (batch_size, n_agents, seq_len, obs_dim)
            state: Global state of shape (batch_size, state_dim)
            hidden: List of hidden states for each agent's RNN
        
        Returns:
            q_values: Individual Q-values of shape (batch_size, n_agents, action_dim)
            q_total: Total Q-value of shape (batch_size, 1)
            hidden: Updated hidden states
        """
        batch_size, n_agents = obs.shape[:2]
        
        if hidden is None:
            hidden = [None] * n_agents
        
        # Get individual Q-values
        q_values = []
        new_hidden = []
        
        for i in range(n_agents):
            q_val, h = self.q_networks[i](obs[:, i], hidden[i])
            q_values.append(q_val)
            new_hidden.append(h)
        
        q_values = torch.stack(q_values, dim=1)  # (batch_size, n_agents, action_dim)
        
        # Get total Q-value
        # Ensure q_values shape is (batch, n_agents)
        q_input = q_values.max(dim=-1)[0]

        # Sometimes q_input is (batch, 1, n_agents), fix it:
        if q_input.dim() == 3 and q_input.shape[1] == 1:
            q_input = q_input.squeeze(1)

        q_total = self.mixing_net(q_input, state)

        
        return q_values, q_total, new_hidden
    
    def get_q_values(self, obs: torch.Tensor, hidden: Optional[List[Tuple]] = None) -> Tuple[torch.Tensor, List[Tuple]]:
        """Get only individual Q-values without mixing."""
        batch_size, n_agents = obs.shape[:2]
        
        if hidden is None:
            hidden = [None] * n_agents
        
        q_values = []
        new_hidden = []
        
        for i in range(n_agents):
            q_val, h = self.q_networks[i](obs[:, i], hidden[i])
            q_values.append(q_val)
            new_hidden.append(h)
        
        q_values = torch.stack(q_values, dim=1)
        
        return q_values, new_hidden
