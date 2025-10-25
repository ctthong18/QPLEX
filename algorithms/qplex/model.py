"""QPLEX Model implementation with various network architectures."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union

from networks.base_networks import QNetwork, MixingNetwork, Attention
from networks.rnn_networks import RNNQNetwork, AttentionRNNQNetwork, BiRNNQNetwork, HierarchicalRNNQNetwork
from .mixer import (QPLEXMixingNetwork, AttentionMixingNetwork, MonotonicMixingNetwork,
                   HierarchicalMixingNetwork, AdaptiveMixingNetwork)


class QPLEXModel(nn.Module):
    """Complete QPLEX model with configurable architectures."""
    
    def __init__(self, obs_dim: int, action_dim: int, state_dim: int, n_agents: int,
                 config: Dict[str, Any]):
        """
        Initialize QPLEX model.
        
        Args:
            obs_dim: Observation dimension for each agent
            action_dim: Action dimension for each agent
            state_dim: Global state dimension
            n_agents: Number of agents
            config: Configuration dictionary
        """
        super(QPLEXModel, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.config = config
        
        # Network configuration
        network_config = config['network']
        q_net_config = network_config['q_network']
        mix_net_config = network_config['mixing_network']
        
        # Individual Q-networks
        self.q_networks = nn.ModuleList([
            self._create_q_network(q_net_config) for _ in range(n_agents)
        ])
        
        # Mixing network
        self.mixing_network = self._create_mixing_network(mix_net_config)
        
        # State encoder (optional)
        if network_config.get('use_state_encoder', False):
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, network_config['state_encoder_hidden']),
                nn.ReLU(),
                nn.Linear(network_config['state_encoder_hidden'], state_dim)
            )
        else:
            self.state_encoder = None
    
    def _create_q_network(self, config: Dict[str, Any]) -> nn.Module:
        """Create individual Q-network based on configuration."""
        network_type = config.get('type', 'mlp')
        
        if network_type == 'mlp':
            return QNetwork(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                hidden_dims=config['hidden_dims'],
                use_rnn=config['use_rnn'],
                rnn_hidden_dim=config['rnn_hidden_dim'],
                rnn_layers=config['rnn_layers'],
                use_attention=config['use_attention'],
                num_attention_heads=config['num_attention_heads']
            )
        elif network_type == 'rnn':
            return RNNQNetwork(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                hidden_dim=config['rnn_hidden_dim'],
                num_layers=config['rnn_layers'],
                rnn_type=config['rnn_type'],
                dropout=config.get('dropout', 0.0)
            )
        elif network_type == 'attention_rnn':
            return AttentionRNNQNetwork(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                hidden_dim=config['rnn_hidden_dim'],
                num_layers=config['rnn_layers'],
                rnn_type=config['rnn_type'],
                num_heads=config['num_attention_heads'],
                dropout=config.get('dropout', 0.0)
            )
        elif network_type == 'bi_rnn':
            return BiRNNQNetwork(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                hidden_dim=config['rnn_hidden_dim'],
                num_layers=config['rnn_layers'],
                rnn_type=config['rnn_type'],
                dropout=config.get('dropout', 0.0)
            )
        elif network_type == 'hierarchical_rnn':
            return HierarchicalRNNQNetwork(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                hidden_dim=config['rnn_hidden_dim'],
                num_layers=config['rnn_layers'],
                rnn_type=config['rnn_type'],
                dropout=config.get('dropout', 0.0)
            )
        else:
            raise ValueError(f"Unsupported Q-network type: {network_type}")
    
    def _create_mixing_network(self, config: Dict[str, Any]) -> nn.Module:
        """Create mixing network based on configuration."""
        mixer_type = config.get('type', 'qplex')
        
        if mixer_type == 'qplex':
            return QPLEXMixingNetwork(
                state_dim=self.state_dim,
                n_agents=self.n_agents,
                hidden_dims=config['hidden_dims'],
                use_hypernet=config['use_hypernet'],
                dueling=config.get('dueling', True)
            )
        elif mixer_type == 'attention':
            return AttentionMixingNetwork(
                state_dim=self.state_dim,
                n_agents=self.n_agents,
                hidden_dim=config['hidden_dims'][0],
                num_heads=config.get('num_attention_heads', 4),
                dropout=config.get('dropout', 0.0)
            )
        elif mixer_type == 'monotonic':
            return MonotonicMixingNetwork(
                state_dim=self.state_dim,
                n_agents=self.n_agents,
                hidden_dims=config['hidden_dims']
            )
        elif mixer_type == 'hierarchical':
            return HierarchicalMixingNetwork(
                state_dim=self.state_dim,
                n_agents=self.n_agents,
                hidden_dims=config['hidden_dims'],
                num_levels=config.get('num_levels', 2)
            )
        elif mixer_type == 'adaptive':
            return AdaptiveMixingNetwork(
                state_dim=self.state_dim,
                n_agents=self.n_agents,
                hidden_dims=config['hidden_dims'],
                complexity_threshold=config.get('complexity_threshold', 0.5)
            )
        else:
            raise ValueError(f"Unsupported mixing network type: {mixer_type}")
    
    def forward(self, obs: torch.Tensor, state: torch.Tensor, 
                hidden: Optional[List[Tuple]] = None) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple]]:
        """
        Forward pass through the model.
        
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
        
        # Encode state if state encoder is used
        if self.state_encoder is not None:
            state = self.state_encoder(state)
        
        # Get individual Q-values
        q_values = []
        new_hidden = []
        
        for i in range(n_agents):
            q_val, h = self.q_networks[i](obs[:, i], hidden[i])
            q_values.append(q_val)
            new_hidden.append(h)
        
        q_values = torch.stack(q_values, dim=1)  # (batch_size, n_agents, action_dim)
        
        # Get total Q-value through mixing network
        q_total = self.mixing_network(q_values.max(dim=-1)[0], state)  # Use max Q-value for mixing
        
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
    
    def get_attention_weights(self, obs: torch.Tensor, state: torch.Tensor) -> Optional[torch.Tensor]:
        """Get attention weights if attention mechanism is used."""
        # This is a placeholder - actual implementation would depend on the specific attention mechanism
        # used in the Q-networks and mixing network
        return None
    
    def get_state_representation(self, state: torch.Tensor) -> torch.Tensor:
        """Get state representation from state encoder."""
        if self.state_encoder is not None:
            return self.state_encoder(state)
        return state
    
    def get_agent_representations(self, obs: torch.Tensor) -> torch.Tensor:
        """Get agent representations from Q-networks."""
        batch_size, n_agents = obs.shape[:2]
        
        representations = []
        for i in range(n_agents):
            # Get intermediate representations from Q-networks
            # This is a placeholder - actual implementation would depend on the specific network architecture
            rep = obs[:, i]  # For now, just return observations
            representations.append(rep)
        
        return torch.stack(representations, dim=1)
    
    def compute_importance_weights(self, obs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Compute importance weights for each agent."""
        # This is a placeholder for computing importance weights
        # Actual implementation would depend on the specific method used
        batch_size, n_agents = obs.shape[:2]
        return torch.ones(batch_size, n_agents, device=obs.device) / n_agents
    
    def get_mixing_weights(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get mixing weights from the mixing network."""
        # This is a placeholder for extracting mixing weights
        # Actual implementation would depend on the specific mixing network architecture
        return {}
    
    def compute_individual_contributions(self, obs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Compute individual agent contributions to the total Q-value."""
        q_values, q_total, _ = self.forward(obs, state)
        
        # Compute individual contributions
        individual_q = q_values.max(dim=-1)[0]  # (batch_size, n_agents)
        total_q = q_total.squeeze(-1)  # (batch_size,)
        
        # Normalize contributions
        contributions = individual_q / (individual_q.sum(dim=1, keepdim=True) + 1e-8)
        
        return contributions
    
    def get_network_parameters(self) -> Dict[str, torch.Tensor]:
        """Get all network parameters."""
        params = {}
        
        # Q-network parameters
        for i, q_net in enumerate(self.q_networks):
            params[f'q_network_{i}'] = dict(q_net.named_parameters())
        
        # Mixing network parameters
        params['mixing_network'] = dict(self.mixing_network.named_parameters())
        
        # State encoder parameters
        if self.state_encoder is not None:
            params['state_encoder'] = dict(self.state_encoder.named_parameters())
        
        return params
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count for each component."""
        counts = {}
        
        # Q-network parameters
        for i, q_net in enumerate(self.q_networks):
            counts[f'q_network_{i}'] = sum(p.numel() for p in q_net.parameters())
        
        # Mixing network parameters
        counts['mixing_network'] = sum(p.numel() for p in self.mixing_network.parameters())
        
        # State encoder parameters
        if self.state_encoder is not None:
            counts['state_encoder'] = sum(p.numel() for p in self.state_encoder.parameters())
        
        # Total parameters
        counts['total'] = sum(counts.values())
        
        return counts
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'state_dim': self.state_dim,
            'n_agents': self.n_agents,
            'config': self.config,
            'parameter_count': self.get_parameter_count(),
            'q_network_types': [type(q_net).__name__ for q_net in self.q_networks],
            'mixing_network_type': type(self.mixing_network).__name__,
            'has_state_encoder': self.state_encoder is not None
        }
