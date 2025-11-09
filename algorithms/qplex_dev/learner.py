"""QPLEX Learner for training and experience replay."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import random
from collections import deque
import pickle
import os

from .agent import QPLEXAgent


class ReplayBuffer:
    """Experience replay buffer for multi-agent QPLEX."""
    
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, state_dim: int, n_agents: int):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            obs_dim: Observation dimension for each agent
            action_dim: Action dimension for each agent
            state_dim: Global state dimension
            n_agents: Number of agents
        """
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.n_agents = n_agents
        
        # Initialize buffers
        self.obs_buffer = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.action_buffer = np.zeros((capacity, n_agents, action_dim), dtype=np.float32)
        self.reward_buffer = np.zeros((capacity, n_agents), dtype=np.float32)
        self.next_obs_buffer = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.done_buffer = np.zeros(capacity, dtype=np.bool_)
        self.state_buffer = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_state_buffer = np.zeros((capacity, state_dim), dtype=np.float32)
        
        self.size = 0
        self.ptr = 0
    
    def add(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
            next_obs: np.ndarray, done: bool, state: np.ndarray, next_state: np.ndarray):
        """
        Add experience to buffer.
        
        Args:
            obs: Observations of shape (n_agents, obs_dim)
            actions: Actions of shape (n_agents,)
            rewards: Rewards of shape (n_agents,)
            next_obs: Next observations of shape (n_agents, obs_dim)
            done: Whether episode is done
            state: Global state of shape (state_dim,)
            next_state: Next global state of shape (state_dim,)
        """
        self.obs_buffer[self.ptr] = obs
        self.action_buffer[self.ptr] = actions
        self.reward_buffer[self.ptr] = rewards
        self.next_obs_buffer[self.ptr] = next_obs
        self.done_buffer[self.ptr] = done
        self.state_buffer[self.ptr] = state
        self.next_state_buffer[self.ptr] = next_state
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
        
        Returns:
            batch: Dictionary containing batch data
        """
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        batch = {
            'obs': torch.FloatTensor(self.obs_buffer[indices]),
            'actions': torch.LongTensor(self.action_buffer[indices]),
            'rewards': torch.FloatTensor(self.reward_buffer[indices]),
            'next_obs': torch.FloatTensor(self.next_obs_buffer[indices]),
            'dones': torch.BoolTensor(self.done_buffer[indices]),
            'state': torch.FloatTensor(self.state_buffer[indices]),
            'next_state': torch.FloatTensor(self.next_state_buffer[indices])
        }
        
        return batch
    
    def save(self, filepath: str):
        """Save buffer to file."""
        data = {
            'obs_buffer': self.obs_buffer[:self.size],
            'action_buffer': self.action_buffer[:self.size],
            'reward_buffer': self.reward_buffer[:self.size],
            'next_obs_buffer': self.next_obs_buffer[:self.size],
            'done_buffer': self.done_buffer[:self.size],
            'state_buffer': self.state_buffer[:self.size],
            'next_state_buffer': self.next_state_buffer[:self.size],
            'size': self.size,
            'ptr': self.ptr
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load buffer from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.size = data['size']
        self.ptr = data['ptr']
        
        # Copy data to buffers
        self.obs_buffer[:self.size] = data['obs_buffer']
        self.action_buffer[:self.size] = data['action_buffer']
        self.reward_buffer[:self.size] = data['reward_buffer']
        self.next_obs_buffer[:self.size] = data['next_obs_buffer']
        self.done_buffer[:self.size] = data['done_buffer']
        self.state_buffer[:self.size] = data['state_buffer']
        self.next_state_buffer[:self.size] = data['next_state_buffer']


class QPLEXLearner:
    """QPLEX Learner for training the algorithm."""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        """
        Initialize QPLEX learner.
        
        Args:
            config: Configuration dictionary
            device: PyTorch device
        """
        self.config = config
        self.device = device
        
        # Training parameters
        self.total_timesteps = config['training']['total_timesteps']
        self.learning_starts = config['training']['learning_starts']
        self.train_freq = config['training']['train_freq']
        self.target_update_interval = config['training']['target_update_interval']
        self.gradient_steps = config['training']['gradient_steps']
        self.batch_size = config['training']['batch_size']
        self.buffer_size = config['training']['buffer_size']
        
        # Initialize agent and buffer (will be set after environment is created)
        self.agent = None
        self.buffer = None
        
        # Training statistics
        self.training_stats = {
            'episode_rewards': deque(maxlen=100),
            'episode_lengths': deque(maxlen=100),
            'losses': deque(maxlen=1000),
            'q_values': deque(maxlen=1000),
            'td_errors': deque(maxlen=1000),
            'epsilon': deque(maxlen=1000)
        }
        
        # Training state
        self.timestep = 0
        self.episode_count = 0
        self.last_log_time = 0
        self.last_save_time = 0
        
    def setup(self, obs_dim: int, action_dim: int, state_dim: int, n_agents: int):
        """
        Setup learner with environment dimensions.
        
        Args:
            obs_dim: Observation dimension for each agent
            action_dim: Action dimension for each agent
            state_dim: Global state dimension
            n_agents: Number of agents
        """
        # Initialize agent
        self.agent = QPLEXAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            state_dim=state_dim,
            n_agents=n_agents,
            config=self.config,
            device=self.device
        )
        
        # Initialize replay buffer
        self.buffer = ReplayBuffer(
            capacity=self.buffer_size,
            obs_dim=obs_dim,
            action_dim=action_dim,
            state_dim=state_dim,
            n_agents=n_agents
        )
    
    def learn(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
              next_obs: np.ndarray, done: bool, state: np.ndarray, next_state: np.ndarray,
              info: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Learn from experience.
        
        Args:
            obs: Current observations
            actions: Actions taken
            rewards: Rewards received
            next_obs: Next observations
            done: Whether episode is done
            state: Current global state
            next_state: Next global state
            info: Additional information
        
        Returns:
            learning_info: Dictionary with learning statistics
        """
        # Add experience to buffer
        self.buffer.add(obs, actions, rewards, next_obs, done, state, next_state)
        
        # Update timestep
        self.timestep += 1
        
        # Train if enough experiences and training frequency
        learning_info = {}
        if (self.timestep >= self.learning_starts and 
            self.timestep % self.train_freq == 0 and 
            self.buffer.size >= self.batch_size):
            
            # Perform gradient steps
            for _ in range(self.gradient_steps):
                batch = self.buffer.sample(self.batch_size)
                
                # Move batch to device
                for key in batch:
                    batch[key] = batch[key].to(self.device)
                
                # Train agent
                train_info = self.agent.train(batch)
                learning_info.update(train_info)
                
                # Update training statistics
                self.training_stats['losses'].append(train_info['loss'])
                self.training_stats['q_values'].append(train_info['q_values'])
                self.training_stats['td_errors'].append(train_info['td_error'])
                self.training_stats['epsilon'].append(train_info['epsilon'])
        
        # Update episode statistics
        if done:
            self.episode_count += 1
            if info is not None:
                episode_reward = info.get('episode_reward', float(np.sum(rewards)))
                episode_length = info.get('episode_length', self.timestep - self.last_log_time)
                
                self.training_stats['episode_rewards'].append(episode_reward)
                self.training_stats['episode_lengths'].append(episode_length)
                
                self.last_log_time = self.timestep
        
        return learning_info
    
    def select_action(self, obs: np.ndarray, state: np.ndarray, 
                     evaluate: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Select actions for all agents.
        
        Args:
            obs: Observations of shape (n_agents, obs_dim)
            state: Global state of shape (state_dim,)
            evaluate: Whether in evaluation mode
        
        Returns:
            actions: Selected actions
            info: Additional information
        """
        if self.agent is None:
            raise ValueError("Learner not setup. Call setup() first.")
        
        return self.agent.select_action(obs, state, evaluate)
    
    def reset_hidden_states(self):
        """Reset hidden states for RNN."""
        if self.agent is not None:
            self.agent.reset_hidden_states()
    
    def get_training_stats(self) -> Dict[str, float]:
        """Get current training statistics."""
        stats = {}
        
        # Episode statistics
        if len(self.training_stats['episode_rewards']) > 0:
            stats['mean_episode_reward'] = np.mean(self.training_stats['episode_rewards'])
            stats['std_episode_reward'] = np.std(self.training_stats['episode_rewards'])
            stats['mean_episode_length'] = np.mean(self.training_stats['episode_lengths'])
        
        # Training statistics
        if len(self.training_stats['losses']) > 0:
            stats['mean_loss'] = np.mean(self.training_stats['losses'])
            stats['mean_q_values'] = np.mean(self.training_stats['q_values'])
            stats['mean_td_error'] = np.mean(self.training_stats['td_errors'])
            stats['mean_epsilon'] = np.mean(self.training_stats['epsilon'])
        
        # General statistics
        stats['timestep'] = self.timestep
        stats['episode_count'] = self.episode_count
        stats['buffer_size'] = self.buffer.size if self.buffer else 0
        
        return stats
    
    def save(self, filepath: str):
        """Save learner state."""
        if self.agent is None:
            raise ValueError("Learner not setup. Call setup() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save agent
        agent_path = filepath.replace('.pth', '_agent.pth')
        self.agent.save(agent_path)
        
        # Save buffer
        buffer_path = filepath.replace('.pth', '_buffer.pkl')
        self.buffer.save(buffer_path)
        
        # Save learner state
        learner_state = {
            'timestep': self.timestep,
            'episode_count': self.episode_count,
            'training_stats': dict(self.training_stats),
            'config': self.config
        }
        
        torch.save(learner_state, filepath)
    
    def load(self, filepath: str):
        """Load learner state."""
        # Load learner state
        learner_state = torch.load(filepath, map_location=self.device)
        self.timestep = learner_state['timestep']
        self.episode_count = learner_state['episode_count']
        self.training_stats = learner_state['training_stats']
        
        # Load agent
        agent_path = filepath.replace('.pth', '_agent.pth')
        self.agent.load(agent_path)
        
        # Load buffer
        buffer_path = filepath.replace('.pth', '_buffer.pkl')
        if os.path.exists(buffer_path):
            self.buffer.load(buffer_path)
    
    def should_log(self, log_interval: int) -> bool:
        """Check if it's time to log."""
        return self.timestep - self.last_log_time >= log_interval
    
    def should_save(self, save_interval: int) -> bool:
        """Check if it's time to save."""
        return self.timestep - self.last_save_time >= save_interval
    
    def update_log_time(self):
        """Update last log time."""
        self.last_log_time = self.timestep
    
    def update_save_time(self):
        """Update last save time."""
        self.last_save_time = self.timestep
