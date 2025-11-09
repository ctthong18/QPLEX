"""Optimized QPLEX Model for Tensor Coverage."""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any

from .base_networks import QNetwork, MixingNetwork
from .rnn_networks import RNNQNetwork, AttentionRNNQNetwork
from qplex.mixer import QPLEXMixingNetwork, AdaptiveMixingNetwork


class OptimizedQPLEX(nn.Module):
    """Optimized QPLEX model với các cải tiến cho tensor coverage."""
    
    def __init__(self, obs_dim: int, action_dim: int, state_dim: int, n_agents: int,
                 config: Dict[str, Any]):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.config = config
        
        # Individual Q-networks với kiến trúc tối ưu
        self.q_networks = nn.ModuleList([
            self._create_optimized_q_network(config) 
            for _ in range(n_agents)
        ])
        
        # Adaptive mixing network
        self.mixing_network = AdaptiveMixingNetwork(
            state_dim=state_dim,
            n_agents=n_agents,
            hidden_dims=[512, 256],
            complexity_threshold=0.6
        )
        
        # State encoder để xử lý global state
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim)
        )
        
        # Gradient monitoring
        self.gradient_norms = []
    
    def _create_optimized_q_network(self, config: Dict[str, Any]) -> nn.Module:
        """Tạo Q-network được tối ưu cho tensor coverage."""
        network_type = config.get('q_network_type', 'attention_rnn')
        
        if network_type == 'attention_rnn':
            return AttentionRNNQNetwork(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                hidden_dim=128,
                num_layers=2,
                rnn_type='lstm',
                num_heads=4,
                dropout=0.1
            )
        elif network_type == 'hierarchical_rnn':
            from .rnn_networks import HierarchicalRNNQNetwork
            return HierarchicalRNNQNetwork(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                hidden_dim=128,
                num_layers=2,
                rnn_type='lstm',
                dropout=0.1
            )
        else:
            return QNetwork(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                hidden_dims=[256, 128],
                use_rnn=True,
                rnn_hidden_dim=128
            )
    
    def forward(self, obs: torch.Tensor, state: torch.Tensor, 
                hidden: Optional[List[Tuple]] = None) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple]]:
        
        batch_size, n_agents = obs.shape[:2]
        
        if hidden is None:
            hidden = [None] * n_agents
        
        # Encode global state
        encoded_state = self.state_encoder(state)
        
        # Get individual Q-values
        q_values = []
        new_hidden = []
        
        for i in range(n_agents):
            q_val, h = self.q_networks[i](obs[:, i], hidden[i])
            q_values.append(q_val)
            new_hidden.append(h)
        
        q_values = torch.stack(q_values, dim=1)
        
        # Get total Q-value through adaptive mixing
        q_individual = q_values.max(dim=-1)[0]  # Use max Q-values for mixing
        q_total = self.mixing_network(q_individual, encoded_state)
        
        return q_values, q_total, new_hidden
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Đếm số lượng parameters."""
        counts = {}
        counts['q_networks'] = sum(p.numel() for p in self.q_networks.parameters())
        counts['mixing_network'] = sum(p.numel() for p in self.mixing_network.parameters())
        counts['state_encoder'] = sum(p.numel() for p in self.state_encoder.parameters())
        counts['total'] = sum(counts.values())
        return counts