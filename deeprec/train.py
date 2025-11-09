"""Training script được tối ưu cho Tensor Coverage."""
import torch
import numpy as np
import yaml
import time
import logging
from pathlib import Path
from typing import Dict, Any

from qplex.agent import OptimizedQPLEXAgent
from qplex.learner import QPLEXLearner
from environment import TensorCoverageEnvironment  # Giữ nguyên environment của bạn


class TensorCoverageTrainer:
    """Trainer được tối ưu cho tensor coverage."""
    
    def __init__(self, config_path: str = "configs/base_config.yaml"):
        self.config = self.load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize environment
        self.env = TensorCoverageEnvironment()  # Giữ nguyên environment
        
        # Get dimensions
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        state_dim = self.env.state_space.shape[0] if hasattr(self.env, 'state_space') else obs_dim * self.env.n_agents
        n_agents = self.env.n_agents
        
        # Initialize learner với optimized agent
        self.learner = QPLEXLearner(self.config, self.device)
        self.learner.setup(obs_dim, action_dim, state_dim, n_agents)
        
        # Training state
        self.start_episode = 0
        self.best_reward = -np.inf
        
        # Create directories
        Path("checkpoints").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Tải configuration file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_logging(self):
        """Setup logging system."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/training.log'),
                logging.StreamHandler()
            ]
        )
    
    def compute_enhanced_reward(self, coverage_state: Dict[str, Any]) -> np.ndarray:
        """Enhanced reward function cho tensor coverage."""
        n_tensors = coverage_state['n_tensors']
        rewards = np.zeros(n_tensors)
        
        # 1. Coverage reward với probabilistic model
        for i in range(n_tensors):
            distance = coverage_state['target_distances'][i]
            if distance <= 10:  # reliable_radius
                coverage_reward = 2.0
            elif distance <= 30:  # max_radius
                coverage_reward = 2.0 * np.exp(-0.1 * (distance - 10))
            else:
                coverage_reward = 0.0
            rewards[i] += coverage_reward
        
        # 2. Energy penalty
        rotation_costs = coverage_state.get('rotation_costs', np.zeros(n_tensors))
        rewards -= 0.05 * rotation_costs
        
        # 3. Obstacle penalty
        obstacle_distances = coverage_state.get('obstacle_distances', np.ones(n_tensors) * 100)
        obstacle_penalty = 1.0 * np.exp(-0.1 * obstacle_distances)
        rewards -= obstacle_penalty
        
        # 4. Collaboration bonus (tránh overlap)
        coverage_scores = coverage_state.get('coverage_scores', np.ones(n_tensors))
        balance_bonus = 0.5 * (1.0 - np.std(coverage_scores))
        rewards += balance_bonus
        
        return rewards
    
    def train(self, total_episodes: int = 100000):
        """Main training loop được tối ưu."""
        self.logger.info(f"Starting training for {total_episodes} episodes")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Number of agents: {self.env.n_agents}")
        
        # Resume từ checkpoint nếu có
        self._try_resume_training()
        
        for episode in range(self.start_episode, total_episodes):
            start_time = time.time()
            
            # Reset environment
            obs, state = self.env.reset()
            self.learner.reset_hidden_states()
            episode_reward = 0
            episode_steps = 0
            
            while True:
                # Select action
                actions, action_info = self.learner.select_action(obs, state)
                
                # Environment step
                next_obs, next_state, rewards, done, env_info = self.env.step(actions)
                
                # Enhanced reward shaping
                enhanced_rewards = self.compute_enhanced_reward(env_info)
                
                # Learn from experience
                learn_info = self.learner.learn(
                    obs, actions, enhanced_rewards, next_obs, done, state, next_state
                )
                
                episode_reward += np.sum(enhanced_rewards)
                episode_steps += 1
                obs, state = next_obs, next_state
                
                if done:
                    break
            
            # Episode logging
            episode_time = time.time() - start_time
            self._log_episode(episode, episode_reward, episode_steps, episode_time, learn_info)
            
            # Evaluation và checkpointing
            if episode % 100 == 0:
                eval_reward = self.evaluate()
                self._save_checkpoint(episode, eval_reward)
                
                # Early stopping nếu performance plateau
                if self._should_early_stop(episode, eval_reward):
                    self.logger.info(f"Early stopping at episode {episode}")
                    break
    
    def evaluate(self, num_episodes: int = 5) -> float:
        """Evaluation function."""
        total_reward = 0
        for _ in range(num_episodes):
            obs, state = self.env.reset()
            self.learner.reset_hidden_states()
            episode_reward = 0
            
            while True:
                actions, _ = self.learner.select_action(obs, state, evaluate=True)
                next_obs, next_state, rewards, done, _ = self.env.step(actions)
                episode_reward += np.sum(rewards)
                obs, state = next_obs, next_state
                
                if done:
                    break
            
            total_reward += episode_reward
        
        return total_reward / num_episodes
    
    def _try_resume_training(self):
        """Thử resume training từ checkpoint."""
        checkpoint_path = "checkpoints/latest_checkpoint.pth"
        if Path(checkpoint_path).exists():
            self.logger.info("Found checkpoint, resuming training...")
            self.start_episode, self.best_reward = self.learner.agent.load_checkpoint(checkpoint_path)
            self.logger.info(f"Resumed from episode {self.start_episode}")
    
    def _save_checkpoint(self, episode: int, eval_reward: float):
        """Lưu checkpoint."""
        checkpoint_path = f"checkpoints/model_{episode}.pth"
        self.learner.agent.save_checkpoint(checkpoint_path, episode, eval_reward)
        
        # Lưu best model
        if eval_reward > self.best_reward:
            self.best_reward = eval_reward
            best_path = "checkpoints/best_model.pth"
            self.learner.agent.save_checkpoint(best_path, episode, eval_reward)
            self.logger.info(f"New best model saved with reward: {eval_reward:.2f}")
        
        # Lưu latest checkpoint
        latest_path = "checkpoints/latest_checkpoint.pth"
        self.learner.agent.save_checkpoint(latest_path, episode, eval_reward)
    
    def _log_episode(self, episode: int, reward: float, steps: int, 
                    duration: float, learn_info: Dict[str, float]):
        """Log episode information."""
        if learn_info:
            loss = learn_info.get('loss', 0)
            q_values = learn_info.get('q_values', 0)
            epsilon = learn_info.get('epsilon', 0)
            lr = learn_info.get('lr', 0)
        else:
            loss = q_values = epsilon = lr = 0
        
        self.logger.info(
            f"Ep {episode:6d} | Reward: {reward:8.2f} | Steps: {steps:3d} | "
            f"Time: {duration:5.2f}s | Loss: {loss:7.4f} | Q: {q_values:7.4f} | "
            f"Eps: {epsilon:.4f} | LR: {lr:.2e}"
        )
    
    def _should_early_stop(self, episode: int, eval_reward: float) -> bool:
        """Kiểm tra xem có nên early stop không."""
        # Chỉ xem xét early stop sau 20k episodes
        if episode < 20000:
            return False
        
        # Nếu eval_reward quá thấp sau 40k episodes
        if episode > 40000 and eval_reward < 50:
            self.logger.warning(f"Early stopping: low performance {eval_reward:.2f}")
            return True
        
        return False


if __name__ == "__main__":
    # Training với config
    trainer = TensorCoverageTrainer("configs/base_config.yaml")
    trainer.train(total_episodes=100000)