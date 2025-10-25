"""Training script for QPLEX algorithm on MATE environment."""

import os
import sys
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional
import time
import logging
from pathlib import Path
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import MATE environment
import mate
from mate.environment import MultiAgentTracking
from mate.agents import GreedyTargetAgent, RandomTargetAgent

# Import QPLEX components
from algorithms.qplex.learner import QPLEXLearner
from algorithms.qplex.agent import QPLEXAgent


def setup_logging(log_dir: str, log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("QPLEX")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_environment(config: Dict[str, Any]) -> MultiAgentTracking:
    """Create MATE environment."""
    env_config = config['env']
    
    # Load environment configuration
    env = MultiAgentTracking(
        config=env_config['config_file'],
        render_mode=env_config.get('render_mode', 'human'),
        window_size=env_config.get('window_size', 800)
    )
    
    return env


def create_target_agent(agent_type: str = "greedy", seed: int = 42):
    """Create target agent for the environment."""
    if agent_type == "greedy":
        return GreedyTargetAgent(seed=seed)
    elif agent_type == "random":
        return RandomTargetAgent(seed=seed)
    else:
        raise ValueError(f"Unsupported target agent type: {agent_type}")


def evaluate_agent(learner: QPLEXLearner, env: MultiAgentTracking, 
                  n_episodes: int = 10, render: bool = False) -> Dict[str, float]:
    """Evaluate the agent."""
    episode_rewards = []
    episode_lengths = []
    coverage_rates = []
    transport_rates = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        camera_obs, target_obs = obs
        state = env.state()
        
        episode_reward = 0
        episode_length = 0
        done = False
        
        # Reset hidden states
        learner.reset_hidden_states()
        
        while not done:
            # Select actions for cameras (our agents)
            camera_actions, _ = learner.select_action(camera_obs, state, evaluate=True)
            
            # Select actions for targets (using target agent)
            target_actions = np.zeros((env.num_targets, 2))  # Default actions
            
            # Combine actions
            actions = (camera_actions, target_actions)
            
            # Step environment
            obs, rewards, terminated, truncated, info = env.step(actions)
            camera_obs, target_obs = obs
            camera_rewards, target_rewards = rewards
            state = env.state()
            
            episode_reward += np.sum(camera_rewards)
            episode_length += 1
            done = terminated or truncated
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Extract metrics from info
        if info and len(info) > 0:
            camera_infos, target_infos = info
            if camera_infos and len(camera_infos) > 0:
                coverage_rates.append(camera_infos[0].get('coverage_rate', 0.0))
                transport_rates.append(camera_infos[0].get('mean_transport_rate', 0.0))
    
    return {
        'mean_episode_reward': np.mean(episode_rewards),
        'std_episode_reward': np.std(episode_rewards),
        'mean_episode_length': np.mean(episode_lengths),
        'mean_coverage_rate': np.mean(coverage_rates) if coverage_rates else 0.0,
        'mean_transport_rate': np.mean(transport_rates) if transport_rates else 0.0
    }


def train_qplex(config: Dict[str, Any], logger: logging.Logger):
    """Main training function."""
    # Set random seeds
    seed = config.get('seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Setup device
    device = torch.device("cuda" if config['device']['use_cuda'] and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create environment
    env = create_environment(config)
    logger.info(f"Environment created: {env}")
    
    # Get environment dimensions
    obs_dim = env.camera_observation_dim
    action_dim = 2  # Camera action dimension (rotation, zoom)
    state_dim = env.state_space.shape[0]
    n_agents = env.num_cameras
    
    logger.info(f"Environment dimensions:")
    logger.info(f"  Observation dim: {obs_dim}")
    logger.info(f"  Action dim: {action_dim}")
    logger.info(f"  State dim: {state_dim}")
    logger.info(f"  Number of agents: {n_agents}")
    
    # Create learner
    learner = QPLEXLearner(config, device)
    learner.setup(obs_dim, action_dim, state_dim, n_agents)
    logger.info("QPLEX learner created and setup")
    
    # Training parameters
    training_config = config['training']
    total_timesteps = training_config['total_timesteps']
    learning_starts = training_config['learning_starts']
    train_freq = training_config['train_freq']
    target_update_interval = training_config['target_update_interval']
    batch_size = training_config['batch_size']
    
    # Logging parameters
    logging_config = config['logging']
    log_interval = logging_config['log_interval']
    eval_interval = logging_config['eval_interval']
    save_interval = logging_config['save_interval']
    log_dir = logging_config['log_dir']
    model_dir = logging_config['model_dir']
    
    # Evaluation parameters
    eval_config = config['evaluation']
    n_eval_episodes = eval_config['n_eval_episodes']
    
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Training loop
    logger.info("Starting training...")
    start_time = time.time()
    
    episode_count = 0
    episode_reward = 0
    episode_length = 0
    
    obs, info = env.reset()
    camera_obs, target_obs = obs
    state = env.state()
    
    # Reset hidden states
    learner.reset_hidden_states()
    
    for timestep in range(total_timesteps):
        # Convert to torch tensors before passing to learner
        camera_obs_tensor = torch.as_tensor(camera_obs, dtype=torch.float32, device=device)
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)

        # Add batch dimension if missing
        if camera_obs_tensor.ndim == 2:  # (n_agents, obs_dim)
            camera_obs_tensor = camera_obs_tensor.unsqueeze(0)  # -> (1, n_agents, obs_dim)
        if state_tensor.ndim == 1:  # (state_dim)
            state_tensor = state_tensor.unsqueeze(0)  # -> (1, state_dim)

        # Select actions
        camera_actions, action_info = learner.select_action(camera_obs, state)
        
        # Select actions for targets (random for now)
        target_actions = np.random.uniform(-1, 1, size=(env.num_targets, 2))
        
        # Combine actions
        actions = (camera_actions, target_actions)
        # print("actions shape:", np.array(actions).shape)

        # Step environment
        next_obs, rewards, terminated, truncated, info = env.step(actions)
        next_camera_obs, next_target_obs = next_obs
        camera_rewards, target_rewards = rewards
        next_state = env.state()
        done = terminated or truncated
        
        # Learn from experience
        learning_info = learner.learn(
            obs=camera_obs,
            actions=camera_actions,
            rewards=camera_rewards,
            next_obs=next_camera_obs,
            done=done,
            state=state,
            next_state=next_state,
            info={'episode_reward': episode_reward, 'episode_length': episode_length}
        )
        
        # Update episode statistics
        episode_reward += np.sum(camera_rewards)
        episode_length += 1
        
        # Update observations and state
        camera_obs = next_camera_obs
        state = next_state
        
        # Episode finished
        if done:
            episode_count += 1
            
            # Log episode statistics
            if episode_count % 10 == 0:
                logger.info(f"Episode {episode_count}: Reward = {episode_reward:.2f}, Length = {episode_length}")
            
            # Reset environment
            obs, info = env.reset()
            camera_obs, target_obs = obs
            state = env.state()
            
            # Reset hidden states
            learner.reset_hidden_states()
            
            # Reset episode statistics
            episode_reward = 0
            episode_length = 0
        
        # Logging
        if timestep % log_interval == 0 and timestep > 0:
            training_stats = learner.get_training_stats()
            elapsed_time = time.time() - start_time
            
            logger.info(f"Timestep {timestep}/{total_timesteps}")
            logger.info(f"  Elapsed time: {elapsed_time:.2f}s")
            logger.info(f"  Episode count: {episode_count}")
            logger.info(f"  Mean episode reward: {training_stats.get('mean_episode_reward', 0.0):.2f}")
            logger.info(f"  Mean loss: {training_stats.get('mean_loss', 0.0):.4f}")
            logger.info(f"  Mean Q-values: {training_stats.get('mean_q_values', 0.0):.4f}")
            logger.info(f"  Mean epsilon: {training_stats.get('mean_epsilon', 0.0):.4f}")
            logger.info(f"  Buffer size: {training_stats.get('buffer_size', 0)}")
            
            # Save training statistics
            stats_file = os.path.join(log_dir, 'training_stats.json')
            with open(stats_file, 'w') as f:
                json.dump(training_stats, f, indent=2)
        
        # Evaluation
        if timestep % eval_interval == 0 and timestep > 0:
            logger.info("Evaluating agent...")
            eval_results = evaluate_agent(learner, env, n_eval_episodes, render=False)
            
            logger.info("Evaluation results:")
            logger.info(f"  Mean episode reward: {eval_results['mean_episode_reward']:.2f} ± {eval_results['std_episode_reward']:.2f}")
            logger.info(f"  Mean episode length: {eval_results['mean_episode_length']:.2f}")
            logger.info(f"  Mean coverage rate: {eval_results['mean_coverage_rate']:.4f}")
            logger.info(f"  Mean transport rate: {eval_results['mean_transport_rate']:.4f}")
            
            # Save evaluation results
            eval_file = os.path.join(log_dir, f'eval_results_{timestep}.json')
            with open(eval_file, 'w') as f:
                json.dump(eval_results, f, indent=2)
        
        # Save model
        if timestep % save_interval == 0 and timestep > 0:
            model_file = os.path.join(model_dir, f'model_{timestep}.pth')
            learner.save(model_file)
            logger.info(f"Model saved to {model_file}")
    
    # Final evaluation
    logger.info("Final evaluation...")
    final_eval_results = evaluate_agent(learner, env, n_eval_episodes * 2, render=False)
    
    logger.info("Final evaluation results:")
    logger.info(f"  Mean episode reward: {final_eval_results['mean_episode_reward']:.2f} ± {final_eval_results['std_episode_reward']:.2f}")
    logger.info(f"  Mean episode length: {final_eval_results['mean_episode_length']:.2f}")
    logger.info(f"  Mean coverage rate: {final_eval_results['mean_coverage_rate']:.4f}")
    logger.info(f"  Mean transport rate: {final_eval_results['mean_transport_rate']:.4f}")
    
    # Save final model
    final_model_file = os.path.join(model_dir, 'final_model.pth')
    learner.save(final_model_file)
    logger.info(f"Final model saved to {final_model_file}")
    
    # Save final evaluation results
    final_eval_file = os.path.join(log_dir, 'final_eval_results.json')
    with open(final_eval_file, 'w') as f:
        json.dump(final_eval_results, f, indent=2)
    
    logger.info("Training completed!")
    
    # Close environment
    env.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train QPLEX on MATE environment')
    parser.add_argument('--config', type=str, default='configs/qplex_4v4_9.yaml',
                       help='Path to configuration file')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override seed if provided
    if args.seed is not None:
        config['seed'] = args.seed
    
    # Setup logging
    log_dir = config['logging']['log_dir']
    logger = setup_logging(log_dir, args.log_level)
    
    logger.info("Starting QPLEX training on MATE environment")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Seed: {config.get('seed', 42)}")
    
    try:
        train_qplex(config, logger)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
