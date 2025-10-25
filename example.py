"""Example script to run QPLEX on MATE environment."""

import os
import sys
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import MATE environment
import mate
from mate.environment import MultiAgentTracking
from mate.agents import GreedyTargetAgent, RandomTargetAgent

# Import QPLEX components
from algorithms.qplex.learner import QPLEXLearner


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_environment(config):
    """Create MATE environment."""
    env_config = config['env']
    
    env = MultiAgentTracking(
        config=env_config['config_file'],
        render_mode=env_config.get('render_mode', 'human'),
        window_size=env_config.get('window_size', 800)
    )
    
    return env


def run_episode(env, learner=None, render=True, max_steps=1000, action_dim=2):
    """Run a single episode."""
    obs, info = env.reset()
    camera_obs, target_obs = obs
    state = env.state()
    
    episode_reward = 0
    episode_length = 0
    done = False
    
    if learner:
        learner.reset_hidden_states()
    
    while not done and episode_length < max_steps:
        if learner:
            # Use trained agent
            camera_obs_batch = np.expand_dims(camera_obs, axis=0)   # (1, n_agents, obs_dim)
            state_batch = np.expand_dims(state, axis=0)   
            camera_obs_batch = np.expand_dims(camera_obs, axis=0)
            state_batch = np.expand_dims(state, axis=0)
            # print("camera_obs_batch:", camera_obs_batch.shape, "state_batch:", state_batch.shape)# (1, state_dim)
            camera_actions, _ = learner.select_action(camera_obs_batch, state_batch, evaluate=True)
            camera_actions = np.array(camera_actions)

            # Nếu đầu ra có shape (1, 4) → reshape về (4,)
            if camera_actions.ndim == 2 and camera_actions.shape[0] == 1:
                camera_actions = camera_actions[0]

            # TẠM THỜI: map mỗi action discrete thành vector 2D ngẫu nhiên (để env.step không lỗi)
            # Khi bạn train thật, bạn sẽ thay phần này bằng chính sách mapping đúng
            camera_actions = np.random.uniform(-1, 1, size=(env.num_cameras, 2))

        else:
            # Random actions
            camera_actions = np.random.uniform(-1, 1, size=(env.num_cameras, action_dim))
        
        # Random target actions
        target_actions = np.random.uniform(-1, 1, size=(env.num_targets, 2))
        
        # Step environment
        # Chuẩn hóa camera actions
        if camera_actions.ndim == 3 and camera_actions.shape[0] in [1, 2]:
            camera_actions = camera_actions[0]

        # Random target actions
        target_actions = np.random.uniform(-1, 1, size=(env.num_targets, 2))

        # Tạo tuple actions đúng định dạng cho MATE
        actions = (camera_actions, target_actions)

        # Debug in thông tin
        # print(">>> camera_actions shape:", camera_actions.shape)
        # print(">>> target_actions shape:", target_actions.shape)
        # print(">>> actions type:", type(actions))

        # Step environment

        obs, rewards, terminated, truncated, info = env.step(actions)
        camera_obs, target_obs = obs
        camera_rewards, target_rewards = rewards
        state = env.state()
        
        episode_reward += camera_rewards if isinstance(camera_rewards, (float, int)) else np.sum(camera_rewards)
        episode_length += 1
        done = terminated or truncated
        
        if render:
            env.render()
    
    return episode_reward, episode_length, info


def main():
    """Main function."""
    print("QPLEX MATE Environment Example")
    print("=" * 40)
    
    # Load configuration
    config_path = "configs/qplex_4v4_9.yaml"
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        print("Please make sure the config file exists.")
        return
    
    config = load_config(config_path)
    print(f"Loaded configuration from: {config_path}")
    
    # Create environment
    env = create_environment(config)
    print(f"Environment created: {env}")
    print(f"Number of cameras: {env.num_cameras}")
    print(f"Number of targets: {env.num_targets}")
    print(f"Number of obstacles: {env.num_obstacles}")
    
    # Get environment dimensions
    obs_dim = env.camera_observation_dim
    action_dim = 2  # Camera action dimension
    state_dim = env.state_space.shape[0]
    n_agents = env.num_cameras
    
    print(f"\nEnvironment dimensions:")
    print(f"  Observation dim: {obs_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  State dim: {state_dim}")
    print(f"  Number of agents: {n_agents}")
    
    # Test environment with random actions
    print(f"\nTesting environment with random actions...")
    episode_reward, episode_length, info = run_episode(env, render=True, max_steps=100)
    print(f"Episode reward: {episode_reward:.2f}")
    print(f"Episode length: {episode_length}")
    
    if info and len(info) > 0:
        camera_infos, target_infos = info
        if camera_infos and len(camera_infos) > 0:
            print(f"Coverage rate: {camera_infos[0].get('coverage_rate', 0.0):.4f}")
            print(f"Transport rate: {camera_infos[0].get('mean_transport_rate', 0.0):.4f}")
    
    # Test with QPLEX learner (untrained)
    print(f"\nTesting with QPLEX learner (untrained)...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    learner = QPLEXLearner(config, device)
    learner.setup(obs_dim, action_dim, state_dim, n_agents)
    print("QPLEX learner created and setup")
    
    # Run a few episodes with untrained agent
    episode_rewards = []
    for i in range(3):
        episode_reward, episode_length, info = run_episode(env, learner, render=True, max_steps=100)
        episode_rewards.append(episode_reward)
        print(f"Episode {i+1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    print(f"Average reward over 3 episodes: {np.mean(episode_rewards):.2f}")
    
    # Close environment
    env.close()
    print("\nEnvironment closed. Example completed!")


if __name__ == "__main__":
    main()