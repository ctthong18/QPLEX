"""Mixing network implementations for QPLEX algorithm."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import math

from networks.base_networks import MixingNetwork, Attention


class QPLEXMixingNetwork(nn.Module):
    """QPLEX-specific mixing network with dueling architecture."""
    
    def __init__(self, state_dim: int, n_agents: int, hidden_dims: List[int] = [256, 256],
                 use_hypernet: bool = True, dueling: bool = True):
        super(QPLEXMixingNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.use_hypernet = use_hypernet
        self.dueling = dueling
        
        if use_hypernet:
            # Hypernetwork approach for QPLEX
            self.hyper_w1 = nn.Linear(state_dim, hidden_dims[0] * n_agents)
            self.hyper_w2 = nn.Linear(state_dim, hidden_dims[1] * hidden_dims[0])
            self.hyper_b1 = nn.Linear(state_dim, hidden_dims[0])
            self.hyper_b2 = nn.Linear(state_dim, hidden_dims[1])
            
            if dueling:
                # Dueling architecture: separate value and advantage streams
                self.hyper_w_value = nn.Linear(state_dim, hidden_dims[1])
                self.hyper_w_advantage = nn.Linear(state_dim, hidden_dims[1] * n_agents)
                self.hyper_b_value = nn.Linear(state_dim, 1)
                self.hyper_b_advantage = nn.Linear(state_dim, n_agents)
            else:
                self.hyper_w_final = nn.Linear(state_dim, hidden_dims[1])
        else:
            # Standard mixing network
            self.mixing_net = nn.Sequential(
                nn.Linear(state_dim, hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[1], 1)
            )
    
    def forward(self, q_values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_values: Individual Q-values of shape (batch_size, n_agents)
            state: Global state of shape (batch_size, state_dim)
        
        Returns:
            q_total: Total Q-value of shape (batch_size, 1)
        """
        if self.use_hypernet:
            batch_size = q_values.size(0)
            
            if self.dueling:
                # Dueling architecture
                # Value stream
                w_value = torch.abs(self.hyper_w_value(state))
                b_value = self.hyper_b_value(state)
                
                # Advantage stream
                w_advantage = torch.abs(self.hyper_w_advantage(state))
                b_advantage = self.hyper_b_advantage(state)
                w_advantage = w_advantage.view(batch_size, self.n_agents, -1)
                
                # Compute advantage
                advantage = torch.bmm(q_values.unsqueeze(1), w_advantage).squeeze(1) + b_advantage
                
                # Compute value
                value = w_value * q_values.sum(dim=1, keepdim=True) + b_value
                
                # Combine value and advantage
                q_total = value + advantage.sum(dim=1, keepdim=True) - advantage.mean(dim=1, keepdim=True)
            else:
                # Standard hypernetwork mixing
                # First layer
                w1 = torch.abs(self.hyper_w1(state))
                b1 = self.hyper_b1(state)
                w1 = w1.view(batch_size, self.n_agents, -1)
                b1 = b1.view(batch_size, 1, -1)
                
                hidden = F.elu(torch.bmm(q_values.unsqueeze(1), w1) + b1)
                
                # Second layer
                w2 = torch.abs(self.hyper_w2(state))
                b2 = self.hyper_b2(state)
                w2 = w2.view(batch_size, -1, 1)
                b2 = b2.view(batch_size, 1, 1)
                
                q_total = torch.bmm(hidden, w2) + b2
                
                # Final layer
                w_final = torch.abs(self.hyper_w_final(state))
                q_total = q_total * w_final
        else:
            # Standard mixing
            q_total = self.mixing_net(state) * q_values.sum(dim=1, keepdim=True)
        
        return q_total


class AttentionMixingNetwork(nn.Module):
    """Mixing network with attention mechanism."""
    
    def __init__(self, state_dim: int, n_agents: int, hidden_dim: int = 256,
                 num_heads: int = 4, dropout: float = 0.0):
        super(AttentionMixingNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        
        # State embedding
        self.state_embedding = nn.Linear(state_dim, hidden_dim)
        
        # Agent embedding
        self.agent_embedding = nn.Linear(1, hidden_dim)  # Q-values as input
        
        # Attention mechanism
        self.attention = Attention(hidden_dim, num_heads, dropout)
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q_values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_values: Individual Q-values of shape (batch_size, n_agents)
            state: Global state of shape (batch_size, state_dim)
        
        Returns:
            q_total: Total Q-value of shape (batch_size, 1)
        """
        batch_size = q_values.size(0)
        
        # Embed state
        state_emb = self.state_embedding(state)  # (batch_size, hidden_dim)
        state_emb = state_emb.unsqueeze(1).repeat(1, self.n_agents, 1)  # (batch_size, n_agents, hidden_dim)
        
        # Embed agent Q-values
        agent_emb = self.agent_embedding(q_values.unsqueeze(-1))  # (batch_size, n_agents, hidden_dim)
        
        # Combine state and agent embeddings
        combined_emb = state_emb + agent_emb
        combined_emb = self.dropout(combined_emb)
        
        # Apply attention
        attn_out = self.attention(combined_emb, combined_emb, combined_emb)
        
        # Residual connection
        attn_out = attn_out + combined_emb
        
        # Global pooling
        pooled = attn_out.mean(dim=1)  # (batch_size, hidden_dim)
        
        # Output
        q_total = self.output_layers(pooled)
        
        return q_total


class MonotonicMixingNetwork(nn.Module):
    """Monotonic mixing network ensuring monotonicity in Q-values."""
    
    def __init__(self, state_dim: int, n_agents: int, hidden_dims: List[int] = [256, 256]):
        super(MonotonicMixingNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.n_agents = n_agents
        
        # Hypernetworks with monotonicity constraints
        self.hyper_w1 = nn.Linear(state_dim, hidden_dims[0] * n_agents)
        self.hyper_w2 = nn.Linear(state_dim, hidden_dims[1] * hidden_dims[0])
        self.hyper_b1 = nn.Linear(state_dim, hidden_dims[0])
        self.hyper_b2 = nn.Linear(state_dim, hidden_dims[1])
        self.hyper_w_final = nn.Linear(state_dim, hidden_dims[1])
    
    def forward(self, q_values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_values: Individual Q-values of shape (batch_size, n_agents)
            state: Global state of shape (batch_size, state_dim)
        
        Returns:
            q_total: Total Q-value of shape (batch_size, 1)
        """
        batch_size = q_values.size(0)
        
        # First layer with monotonicity (positive weights)
        w1 = torch.abs(self.hyper_w1(state))
        b1 = self.hyper_b1(state)
        w1 = w1.view(batch_size, self.n_agents, -1)
        b1 = b1.view(batch_size, 1, -1)
        
        hidden = F.elu(torch.bmm(q_values.unsqueeze(1), w1) + b1)
        
        # Second layer with monotonicity (positive weights)
        w2 = torch.abs(self.hyper_w2(state))
        b2 = self.hyper_b2(state)
        w2 = w2.view(batch_size, -1, 1)
        b2 = b2.view(batch_size, 1, 1)
        
        q_total = torch.bmm(hidden, w2) + b2
        
        # Final layer with monotonicity (positive weights)
        w_final = torch.abs(self.hyper_w_final(state))
        q_total = q_total * w_final
        
        return q_total


class HierarchicalMixingNetwork(nn.Module):
    """Hierarchical mixing network with multiple levels of abstraction."""
    
    def __init__(self, state_dim: int, n_agents: int, hidden_dims: List[int] = [256, 256],
                 num_levels: int = 2):
        super(HierarchicalMixingNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.num_levels = num_levels
        
        # Multiple mixing networks for different levels
        self.mixing_networks = nn.ModuleList([
            QPLEXMixingNetwork(state_dim, n_agents, hidden_dims, use_hypernet=True, dueling=False)
            for _ in range(num_levels)
        ])
        
        # Level weights
        self.level_weights = nn.Linear(state_dim, num_levels)
        
        # Final combination
        self.final_combine = nn.Linear(num_levels, 1)
    
    def forward(self, q_values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_values: Individual Q-values of shape (batch_size, n_agents)
            state: Global state of shape (batch_size, state_dim)
        
        Returns:
            q_total: Total Q-value of shape (batch_size, 1)
        """
        batch_size = q_values.size(0)
        
        # Get outputs from different levels
        level_outputs = []
        for mixing_net in self.mixing_networks:
            level_output = mixing_net(q_values, state)
            level_outputs.append(level_output)
        
        # Stack level outputs
        level_outputs = torch.cat(level_outputs, dim=1)  # (batch_size, num_levels)
        
        # Compute level weights
        level_weights = F.softmax(self.level_weights(state), dim=1)  # (batch_size, num_levels)
        
        # Weighted combination
        weighted_outputs = level_outputs * level_weights
        
        # Final combination
        q_total = self.final_combine(weighted_outputs)
        
        return q_total


class AdaptiveMixingNetwork(nn.Module):
    """Adaptive mixing network that adjusts based on state complexity."""
    
    def __init__(self, state_dim: int, n_agents: int, hidden_dims: List[int] = [256, 256],
                 complexity_threshold: float = 0.5):
        super(AdaptiveMixingNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.complexity_threshold = complexity_threshold
        
        # Complexity estimator
        self.complexity_estimator = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Simple mixing network (for low complexity)
        self.simple_mixer = QPLEXMixingNetwork(state_dim, n_agents, hidden_dims, 
                                              use_hypernet=False, dueling=False)
        
        # Complex mixing network (for high complexity)
        self.complex_mixer = QPLEXMixingNetwork(state_dim, n_agents, hidden_dims, 
                                               use_hypernet=True, dueling=True)
    
    def forward(self, q_values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_values: Individual Q-values of shape (batch_size, n_agents)
            state: Global state of shape (batch_size, state_dim)
        
        Returns:
            q_total: Total Q-value of shape (batch_size, 1)
        """
        # Estimate state complexity
        complexity = self.complexity_estimator(state)  # (batch_size, 1)
        
        # Get outputs from both mixers
        simple_output = self.simple_mixer(q_values, state)
        complex_output = self.complex_mixer(q_values, state)
        
        # Adaptive combination based on complexity
        alpha = torch.sigmoid((complexity - self.complexity_threshold) * 10)
        q_total = alpha * complex_output + (1 - alpha) * simple_output
        
        return q_total
