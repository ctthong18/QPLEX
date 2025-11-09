"""Optimized QPLEX Agent với các cải tiến training."""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import random

from networks.qplex_model import OptimizedQPLEX


class OptimizedQPLEXAgent:
    """QPLEX Agent được tối ưu cho tensor coverage."""
    
    def __init__(self, obs_dim: int, action_dim: int, state_dim: int, n_agents: int,
                 config: Dict[str, Any], device: torch.device):
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.config = config
        self.device = device
        
        # Training parameters với giá trị tối ưu
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.02)
        self.epsilon_decay = config.get('epsilon_decay', 0.9995)
        
        # Initialize optimized networks
        self.q_network = OptimizedQPLEX(
            obs_dim=obs_dim,
            action_dim=action_dim, 
            state_dim=state_dim,
            n_agents=n_agents,
            config=config
        ).to(device)
        
        self.target_q_network = OptimizedQPLEX(
            obs_dim=obs_dim,
            action_dim=action_dim,
            state_dim=state_dim, 
            n_agents=n_agents,
            config=config
        ).to(device)
        
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer với weight decay
        self.optimizer = torch.optim.AdamW(
            self.q_network.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=100000
        )
        
        # Hidden states
        self.hidden_states = [None] * n_agents
        self.target_hidden_states = [None] * n_agents
        
        # Training statistics
        self.training_step = 0
        
    def select_action(self, obs: np.ndarray, state: np.ndarray, 
                     evaluate: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Lựa chọn action được tối ưu."""
        with torch.no_grad():
            # Convert to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get Q-values
            q_values, _, new_hidden = self.q_network(obs_tensor, state_tensor, self.hidden_states)
            self.hidden_states = new_hidden
            
            # Action selection với exploration improved
            if evaluate or random.random() > self.epsilon:
                # Greedy action selection
                actions = q_values.argmax(dim=-1).squeeze(0).cpu().numpy()
            else:
                # Improved exploration: ưu tiên actions gần với greedy
                q_values_np = q_values.squeeze(0).cpu().numpy()
                actions = self._soft_exploration(q_values_np)
            
            info = {
                'q_values': q_values.cpu().numpy().squeeze(0),
                'epsilon': self.epsilon,
                'actions': actions
            }
            
            return actions, info
    
    def _soft_exploration(self, q_values: np.ndarray) -> np.ndarray:
        """Soft exploration sử dụng Boltzmann distribution."""
        temperature = max(0.1, self.epsilon)  # Adaptive temperature
        exp_q = np.exp(q_values / temperature)
        prob = exp_q / np.sum(exp_q, axis=1, keepdims=True)
        
        actions = []
        for i in range(len(q_values)):
            action = np.random.choice(self.action_dim, p=prob[i])
            actions.append(action)
        
        return np.array(actions)
    
    def update_epsilon(self, episode: int):
        """Cập nhật epsilon với curriculum."""
        # Curriculum: giảm exploration nhanh hơn sau 20k episodes
        if episode < 20000:
            decay = self.epsilon_decay
        else:
            decay = self.epsilon_decay ** 1.5  # Giảm nhanh hơn
            
        self.epsilon = max(self.epsilon_end, self.epsilon * decay)
    
    def train(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Training step được tối ưu."""
        self.training_step += 1
        
        # Unpack batch
        obs, actions, rewards, next_obs, dones, state, next_state = (
            batch['obs'], batch['actions'], batch['rewards'], batch['next_obs'],
            batch['dones'], batch['state'], batch['next_state']
        )
        
        # Compute current Q-values
        current_q_values, current_q_total, _ = self.q_network(obs, state)
        
        # Double Q-learning với target network
        with torch.no_grad():
            next_q_values, _, _ = self.q_network(next_obs, next_state)
            next_actions = next_q_values.argmax(dim=-1)
            
            target_q_values, target_q_total, _ = self.target_q_network(next_obs, next_state)
            target_q_selected = target_q_values.gather(2, next_actions.unsqueeze(-1)).squeeze(-1)
        
        # Compute targets
        target_individual = rewards + self.gamma * target_q_selected * (1 - dones.unsqueeze(1).float())
        target_total = rewards.sum(dim=1) + self.gamma * target_q_total.squeeze(-1) * (1 - dones.float())
        
        # Compute losses với huber loss cho stability
        current_q_selected = current_q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        current_q_total_selected = current_q_total.squeeze(-1)
        
        individual_loss = F.smooth_l1_loss(current_q_selected, target_individual)
        total_loss = F.smooth_l1_loss(current_q_total_selected, target_total)
        
        # Combined loss với regularization
        loss = individual_loss + total_loss
        
        # Gradient clipping và optimization
        self.optimizer.zero_grad()
        loss.backward()
        
        # Adaptive gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        self.lr_scheduler.step()
        
        # Soft update target network
        self._soft_update_target_network()
        
        return {
            'loss': loss.item(),
            'individual_loss': individual_loss.item(),
            'total_loss': total_loss.item(),
            'q_values': current_q_selected.mean().item(),
            'epsilon': self.epsilon,
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    def _soft_update_target_network(self):
        """Soft update target network."""
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def save_checkpoint(self, filepath: str, episode: int, reward: float):
        """Lưu checkpoint với metadata."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_q_network_state_dict': self.target_q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'epsilon': self.epsilon,
            'episode': episode,
            'reward': reward,
            'config': self.config
        }, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Tải checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        self.epsilon = checkpoint['epsilon']
        return checkpoint['episode'], checkpoint['reward']