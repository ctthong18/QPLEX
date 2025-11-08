"""QPLEX Agent implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import random
from collections import deque

from networks.base_networks import QPLEXNetwork
from networks.rnn_networks import RNNQNetwork, AttentionRNNQNetwork


class QPLEXAgent:
    """QPLEX Agent for multi-agent reinforcement learning."""
    
    def __init__(self, obs_dim: int, action_dim: int, state_dim: int, n_agents: int,
                 config: Dict[str, Any], device: torch.device):
        """
        Initialize QPLEX agent.
        
        Args:
            obs_dim: Observation dimension for each agent
            action_dim: Action dimension for each agent
            state_dim: Global state dimension
            n_agents: Number of agents
            config: Configuration dictionary
            device: PyTorch device
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.config = config
        self.device = device
        
        # Algorithm parameters
        self.learning_rate = config['algorithm']['learning_rate']
        self.gamma = config['algorithm']['gamma']
        self.tau = config['algorithm']['tau']
        self.epsilon_start = config['algorithm']['epsilon_start']
        self.epsilon_end = config['algorithm']['epsilon_end']
        self.epsilon_decay = config['algorithm']['epsilon_decay']
        self.dueling = config['algorithm']['dueling']
        self.double_q = config['algorithm']['double_q']
        
        # Network parameters
        network_config = config['network']
        q_net_config = network_config['q_network']
        mix_net_config = network_config['mixing_network']
        
        # Initialize networks
        self.q_network = QPLEXNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            state_dim=state_dim,
            n_agents=n_agents,
            q_hidden_dims=q_net_config['hidden_dims'],
            mix_hidden_dims=mix_net_config['hidden_dims'],
            use_rnn=q_net_config['use_rnn'],
            rnn_hidden_dim=q_net_config['rnn_hidden_dim'],
            rnn_layers=q_net_config['rnn_layers'],
            use_attention=q_net_config['use_attention'],
            num_attention_heads=q_net_config['num_attention_heads'],
            use_hypernet=mix_net_config['use_hypernet']
        ).to(device)
        
        self.target_q_network = QPLEXNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            state_dim=state_dim,
            n_agents=n_agents,
            q_hidden_dims=q_net_config['hidden_dims'],
            mix_hidden_dims=mix_net_config['hidden_dims'],
            use_rnn=q_net_config['use_rnn'],
            rnn_hidden_dim=q_net_config['rnn_hidden_dim'],
            rnn_layers=q_net_config['rnn_layers'],
            use_attention=q_net_config['use_attention'],
            num_attention_heads=q_net_config['num_attention_heads'],
            use_hypernet=mix_net_config['use_hypernet']
        ).to(device)
        
        # Copy parameters to target network
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Exploration
        self.epsilon = self.epsilon_start
        
        # Hidden states for RNN
        self.hidden_states = [None] * n_agents
        self.target_hidden_states = [None] * n_agents
        
        # Training statistics
        self.training_stats = {
            'loss': deque(maxlen=1000),
            'q_values': deque(maxlen=1000),
            'target_q_values': deque(maxlen=1000),
            'td_errors': deque(maxlen=1000)
        }
    
    def select_action(self, obs: np.ndarray, state: np.ndarray, 
                     evaluate: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Select actions for all agents.
        
        Args:
            obs: Observations of shape (n_agents, obs_dim)
            state: Global state of shape (state_dim,)
            evaluate: Whether in evaluation mode
        
        Returns:
            actions: Selected actions of shape (n_agents, action_dim)
            info: Additional information
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)  # (1, n_agents, obs_dim)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, state_dim)
            
            if len(obs.shape) == 2:  # (n_agents, obs_dim)
                if isinstance(obs, np.ndarray):
                    obs = torch.from_numpy(obs).float().to(self.device)
                obs = obs.unsqueeze(0)  # (1, n_agents, obs_dim)
            if len(state.shape) == 1:  # (state_dim,)
                if isinstance(state, np.ndarray):
                    state = torch.from_numpy(state).float().to(self.device)
                state = state.unsqueeze(0)  # (1, state_dim)

            # Get Q-values
            q_values, _, new_hidden = self.q_network(obs_tensor, state_tensor, self.hidden_states)
            self.hidden_states = new_hidden
            
            # Select actions
            # Select actions
            if evaluate or random.random() > self.epsilon:
                # Greedy or deterministic action selection
                actions = q_values.squeeze(0).cpu().numpy()  # (n_agents, action_dim) or (n_agents,)
    
                # Ensure actions have correct shape (n_agents, action_dim)
                if actions.ndim == 1:  # if (n_agents,)
                    actions = np.expand_dims(actions, axis=-1)
                    if self.action_dim > 1:
                        # Duplicate along last axis if expected dim > 1
                        actions = np.repeat(actions, self.action_dim, axis=-1)
            else:
                # Random actions
                if self.action_dim == 1:
                    actions = np.random.randint(0, self.action_dim, size=self.n_agents)
                else:
                    actions = np.random.uniform(-1, 1, size=(self.n_agents, self.action_dim))

            # Convert to continuous actions if needed
            if self.action_dim > 1:
                # For continuous action spaces, we might need to sample from the action distribution
                # For now, assuming discrete actions
                pass
            
            info = {
                'q_values': q_values.cpu().numpy().squeeze(0),
                'epsilon': self.epsilon,
                'actions': actions
            }
            
            return actions, info
    
    def update_epsilon(self):
        """Update exploration epsilon."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def train(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Train the agent on a batch of experiences.
        
        Args:
            batch: Dictionary containing batch data
        
        Returns:
            training_info: Dictionary with training statistics
        """
        obs = batch['obs']  # (batch_size, n_agents, obs_dim)
        actions = batch['actions']  # (batch_size, n_agents)
        rewards = batch['rewards']  # (batch_size, n_agents)
        next_obs = batch['next_obs']  # (batch_size, n_agents, obs_dim)
        dones = batch['dones']  # (batch_size,)
        state = batch['state']  # (batch_size, state_dim)
        next_state = batch['next_state']  # (batch_size, state_dim)
        
        batch_size = obs.size(0)
        
        # Current Q-values
        current_q_values, current_q_total, _ = self.q_network(obs, state)
        
        # Select actions for next state
        if self.double_q:
            # Double Q-learning: use main network to select actions
            next_q_values, _, _ = self.q_network(next_obs, next_state)
            next_actions = next_q_values.argmax(dim=-1)
            
            # Use target network to evaluate selected actions
            target_q_values, target_q_total, _ = self.target_q_network(next_obs, next_state)
            target_q_values = target_q_values.gather(2, next_actions.unsqueeze(-1)).squeeze(-1)
        else:
            # Standard Q-learning: use target network for both selection and evaluation
            target_q_values, target_q_total, _ = self.target_q_network(next_obs, next_state)
            target_q_values = target_q_values.max(dim=-1)[0]
        
        # Compute target Q-values
        target_q_total = target_q_total.squeeze(-1)  # (batch_size,)
        target_q_total = rewards.sum(dim=1) + self.gamma * target_q_total * (1 - dones.float())
        
        # Compute individual target Q-values
        target_q_individual = rewards + self.gamma * target_q_values * (1 - dones.unsqueeze(1).float())
        # print("current_q_values.shape =", current_q_values.shape)
        # print("actions.shape =", actions.shape)
        # print("actions.unsqueeze(-1).shape =", actions.unsqueeze(-1).shape)
        if actions.dim() == 3 and actions.size(-1) > 1:
            actions = actions.argmax(dim=-1)

        # Current Q-values for selected actions
        current_q_selected = current_q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        current_q_total_selected = current_q_total.squeeze(-1)
        
        # Compute losses
        # Individual Q-loss
        individual_loss = F.mse_loss(current_q_selected, target_q_individual)
        
        # Total Q-loss (mixing network loss)
        total_loss = F.mse_loss(current_q_total_selected, target_q_total)
        
        # Combined loss
        loss = individual_loss + total_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        
        # Update target network
        self._soft_update_target_network()
        
        # Update training statistics
        self.training_stats['loss'].append(loss.item())
        self.training_stats['q_values'].append(current_q_selected.mean().item())
        self.training_stats['target_q_values'].append(target_q_individual.mean().item())
        self.training_stats['td_errors'].append((target_q_individual - current_q_selected).abs().mean().item())
        
        # Update epsilon
        self.update_epsilon()
        
        return {
            'loss': loss.item(),
            'individual_loss': individual_loss.item(),
            'total_loss': total_loss.item(),
            'q_values': current_q_selected.mean().item(),
            'target_q_values': target_q_individual.mean().item(),
            'td_error': (target_q_individual - current_q_selected).abs().mean().item(),
            'epsilon': self.epsilon
        }
    
    def _soft_update_target_network(self):
        """Soft update of target network parameters."""
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def save(self, filepath: str):
        """Save agent state."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_q_network_state_dict': self.target_q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'config': self.config
        }, filepath)
    
    def load(self, filepath: str):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
    
    def reset_hidden_states(self):
        """Reset hidden states for RNN."""
        self.hidden_states = [None] * self.n_agents
        self.target_hidden_states = [None] * self.n_agents
    
    def get_training_stats(self) -> Dict[str, float]:
        """Get current training statistics."""
        stats = {}
        for key, values in self.training_stats.items():
            if len(values) > 0:
                stats[key] = np.mean(values)
        return stats
