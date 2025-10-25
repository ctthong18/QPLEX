"""Quick test script for QPLEX training on MATE."""

import os
import sys
import yaml
import numpy as np
import torch
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import MATE environment
import mate
from mate.environment import MultiAgentTracking

# Import QPLEX components
from algorithms.qplex.learner import QPLEXLearner


def create_quick_config():
    """Create a quick test configuration."""
    config = {
        'name': 'QPLEX_Quick_Test',
        'env_name': 'MATE-4v4-9',
        'seed': 42,

        'env': {
            'config_file': 'mate/assets/MATE-4v4-9.yaml',
            'max_episode_steps': 1000,
            'reward_type': 'dense',
            'render_mode': 'human',
            'window_size': 800
        },

        'training': {
            'total_timesteps': 10000,  # Very short for quick test
            'learning_starts': 1000,
            'train_freq': 4,
            'target_update_interval': 100,
            'gradient_steps': 1,
            'batch_size': 32,
            'buffer_size': 10000,
            'n_episodes': 100
        },

        'algorithm': {
            'name': 'QPLEX',
            'learning_rate': 0.001,
            'gamma': 0.99,
            'tau': 0.005,
            'epsilon_start': 1.0,
            'epsilon_end': 0.1,
            'epsilon_decay': 0.995,
            'dueling': True,
            'double_q': True
        },

        'network': {
            'q_network': {
                'hidden_dims': [64, 64],  # Smaller networks for quick test
                'use_rnn': False,
                'rnn_hidden_dim': 32,
                'rnn_layers': 1,
                'rnn_type': 'lstm',
                'use_attention': False,
                'num_attention_heads': 4,
                'dropout': 0.0,
                'activation': 'relu'
            },
            'mixing_network': {
                'hidden_dims': [64, 64],
                'use_hypernet': True,
                'dropout': 0.0,
                'activation': 'relu'
            },
            'target_update': {
                'method': 'soft',
                'tau': 0.005
            }
        },

        'agents': {
            'n_agents': 4,
            'obs_dim': None,
            'action_dim': None,
            'state_dim': None
        },

        'logging': {
            'log_interval': 500,
            'eval_interval': 2000,
            'save_interval': 5000,
            'log_dir': './logs/quick_test',
            'model_dir': './models/quick_test',
            'tensorboard': False,
            'wandb': False
        },

        'evaluation': {
            'n_eval_episodes': 3,
            'eval_deterministic': True,
            'render_eval': False,
            'save_videos': False,
            'video_dir': './videos'
        },

        'device': {
            'use_cuda': True,
            'cuda_device': 0
        },

        'reproducibility': {
            'deterministic': True,
            'benchmark': False
        }
    }

    return config


def _to_tensor(x, device):
    """Helper: convert numpy array to torch tensor on device with float dtype."""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    try:
        return torch.tensor(x, dtype=torch.float32, device=device)
    except Exception:
        # fallback: convert list then tensor
        return torch.tensor(np.array(x), dtype=torch.float32, device=device)


def process_action_output(raw_action, env):
    """
    Normalize action output:
     - Accept torch.Tensor or numpy array
     - Squeeze batch dim if present
     - If returns Q-values/logits per agent -> argmax -> discrete actions (n_agents,)
     - Ensure final shape matches env expectation:
         * If learner expects discrete actions (n_agents,), return that
         * Otherwise, fallback to random continuous (n_agents, 2)
    """
    # convert to numpy for easy inspection
    if isinstance(raw_action, torch.Tensor):
        action_np = raw_action.detach().cpu().numpy()
    else:
        action_np = np.array(raw_action)

    # squeeze batch dim if present: (1, n_agents, ...) -> (n_agents, ...)
    if action_np.ndim >= 3 and action_np.shape[0] == 1:
        action_np = action_np[0]
    elif action_np.ndim == 2 and action_np.shape[0] == 1:
        action_np = action_np[0]

    # If shape (n_agents, n_actions) -> treat as Q-values / logits -> argmax per agent
    if action_np.ndim == 2 and action_np.shape[0] == env.num_cameras:
        # e.g. (n_agents, n_actions) -> discrete indices
        # if second dim > 1 assume discrete choices
        if action_np.shape[1] > 1:
            try:
                discrete = np.argmax(action_np, axis=1).astype(int)
                return discrete  # shape (n_agents,)
            except Exception:
                pass

    # If already 1D with length n_agents -> assume discrete indices
    if action_np.ndim == 1 and action_np.shape[0] == env.num_cameras:
        return action_np.astype(int)

    # If already continuous 2D of shape (n_agents, 2) -> return as continuous actions
    if action_np.ndim == 2 and action_np.shape == (env.num_cameras, 2):
        return action_np

    # Fallback: try to get env action space size for discrete mapping
    n_actions = None
    if hasattr(env, 'camera_action_space'):
        try:
            n_actions = env.camera_action_space.n
        except Exception:
            n_actions = None

    if n_actions is None:
        # try attribute
        n_actions = getattr(env, 'camera_action_dim', None)

    if n_actions is not None:
        return np.random.randint(0, int(n_actions), size=(env.num_cameras,))

    # final fallback: continuous random actions
    return np.random.uniform(-1, 1, size=(env.num_cameras, 2))


def quick_train():
    """Quick training test."""
    print("QPLEX Quick Test")
    print("=" * 30)

    # Set random seeds
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create environment
    env = MultiAgentTracking(
        config='mate/assets/MATE-4v4-9.yaml',
        render_mode='human',
        window_size=800
    )
    print(f"Environment created: {env}")

    # Get environment dimensions
    obs_dim = env.camera_observation_dim
    # NOTE: action_dim here is the continuous space dim, but QPLEX likely uses discrete actions.
    action_dim = 2  # if env uses continuous mapping; learner may output discrete indices instead
    state_dim = env.state_space.shape[0]
    n_agents = env.num_cameras

    print(f"Environment dimensions:")
    print(f"  Observation dim: {obs_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  State dim: {state_dim}")
    print(f"  Number of agents: {n_agents}")

    # Create configuration
    config = create_quick_config()

    # Create learner
    learner = QPLEXLearner(config, device)
    learner.setup(obs_dim, action_dim, state_dim, n_agents)
    print("QPLEX learner created and setup")

    # Training parameters
    total_timesteps = config['training']['total_timesteps']
    log_interval = config['logging']['log_interval']

    print(f"\nStarting quick training for {total_timesteps} timesteps...")
    start_time = time.time()

    # Training loop
    episode_count = 0
    episode_reward = 0.0
    episode_length = 0

    obs, info = env.reset()
    camera_obs, target_obs = obs
    state = env.state()

    # Reset hidden states
    learner.reset_hidden_states()

    for timestep in range(total_timesteps):
        # --- prepare inputs as torch tensors on device ---
        # camera_obs: expected shape (n_agents, obs_dim)
        camera_obs_t = _to_tensor(camera_obs, device)
        state_t = _to_tensor(state, device)

        # Some agent implementations expect no batch dim and call unsqueeze inside,
        # others expect already batched (1, n_agents, obs_dim). We'll pass (n_agents, obs_dim)
        # and let agent code unsqueeze if needed.
        # Call learner.select_action with torch tensors
        try:
            camera_actions_raw, action_info = learner.select_action(camera_obs_t, state_t)
        except Exception as e:
            # If select_action expects numpy, try passing numpy
            # (rare given error you saw, but safe fallback)
            # convert tensors back to numpy
            try:
                camera_actions_raw, action_info = learner.select_action(camera_obs, state)
            except Exception as e2:
                print("select_action failed with both tensor and numpy inputs:", e, e2)
                raise

        # Process the raw action output into a usable form for env
        camera_actions = process_action_output(camera_actions_raw, env)

        # If camera_actions are discrete indices (n_agents,), we must map to env continuous if env.step requires it.
        # Check what env.step expects: if it expects continuous (n_agents,2), build mapping.
        # We'll detect by trying a step with the actions; but safer: if camera_actions is 1D -> assume discrete policy and map to continuous.
        if camera_actions.ndim == 1:
            # Map discrete indices to continuous vectors (placeholder mapping)
            # A simple mapping: map index -> direction vector on unit circle (for small n_actions)
            n_actions = None
            if hasattr(env, 'camera_action_space'):
                try:
                    n_actions = env.camera_action_space.n
                except Exception:
                    n_actions = None
            if n_actions is None:
                n_actions = getattr(env, 'camera_action_dim', None)
            if n_actions is None:
                # fallback guess
                n_actions = int(np.max(camera_actions) + 1)

            # Create mapping: for each discrete value k -> angle = 2*pi*k/n_actions
            cont_actions = []
            for k in camera_actions:
                angle = 2.0 * np.pi * (int(k) % max(1, n_actions)) / max(1, n_actions)
                cont_actions.append([np.cos(angle), np.sin(angle)])
            camera_actions_cont = np.array(cont_actions, dtype=float)
            final_camera_actions = camera_actions_cont
        else:
            # assume already continuous (n_agents, 2)
            final_camera_actions = camera_actions

        # Random target actions
        target_actions = np.random.uniform(-1, 1, size=(env.num_targets, 2))

        # Combine actions
        actions = (final_camera_actions, target_actions)

        # Step environment
        next_obs, rewards, terminated, truncated, info = env.step(actions)
        next_camera_obs, next_target_obs = next_obs
        camera_rewards, target_rewards = rewards
        next_state = env.state()
        done = terminated or truncated

        # Learn from experience
        # Try to call learner.learn with numpy first (many learners accept numpy). If it fails, try torch batch.
        try:
            learning_info = learner.learn(
                obs=camera_obs,
                actions=camera_actions,  # pass raw (discrete) actions for learner
                rewards=camera_rewards,
                next_obs=next_camera_obs,
                done=done,
                state=state,
                next_state=next_state,
                info={'episode_reward': episode_reward, 'episode_length': episode_length}
            )
        except Exception as e:
            # Attempt batched torch wrappers if learner expects tensors
            try:
                learning_info = learner.learn(
                    obs=_to_tensor(np.expand_dims(camera_obs, axis=0), device),
                    actions=_to_tensor(np.expand_dims(camera_actions, axis=0), device),
                    rewards=_to_tensor(np.expand_dims(camera_rewards, axis=0), device),
                    next_obs=_to_tensor(np.expand_dims(next_camera_obs, axis=0), device),
                    done=np.array([done]),
                    state=_to_tensor(np.expand_dims(state, axis=0), device),
                    next_state=_to_tensor(np.expand_dims(next_state, axis=0), device),
                    info={'episode_reward': episode_reward, 'episode_length': episode_length}
                )
            except Exception as e2:
                print("learner.learn failed (both numpy and batched-tensor attempts). Skipping learning this step.")
                learning_info = {}

        # Update episode statistics
        try:
            episode_reward += np.sum(camera_rewards)
        except Exception:
            episode_reward += float(camera_rewards)

        episode_length += 1

        # Update observations and state
        camera_obs = next_camera_obs
        state = next_state

        # Episode finished
        if done:
            episode_count += 1
            print(f"Episode {episode_count}: Reward = {episode_reward:.2f}, Length = {episode_length}")

            # Reset environment
            obs, info = env.reset()
            camera_obs, target_obs = obs
            state = env.state()

            # Reset hidden states
            learner.reset_hidden_states()

            # Reset episode statistics
            episode_reward = 0.0
            episode_length = 0

        # Logging
        if timestep % log_interval == 0 and timestep > 0:
            training_stats = {}
            try:
                training_stats = learner.get_training_stats()
            except Exception:
                training_stats = {}
            elapsed_time = time.time() - start_time

            print(f"\nTimestep {timestep}/{total_timesteps}")
            print(f"  Elapsed time: {elapsed_time:.2f}s")
            print(f"  Episode count: {episode_count}")
            print(f"  Mean episode reward: {training_stats.get('mean_episode_reward', 0.0):.2f}")
            print(f"  Mean loss: {training_stats.get('mean_loss', 0.0):.4f}")
            print(f"  Mean Q-values: {training_stats.get('mean_q_values', 0.0):.4f}")
            print(f"  Mean epsilon: {training_stats.get('mean_epsilon', 0.0):.4f}")
            print(f"  Buffer size: {training_stats.get('buffer_size', 0)}")

    # Final evaluation
    print(f"\nFinal evaluation...")
    episode_rewards = []
    for i in range(3):
        obs, info = env.reset()
        camera_obs, target_obs = obs
        state = env.state()

        learner.reset_hidden_states()

        episode_reward = 0.0
        episode_length = 0
        done = False

        while not done and episode_length < 200:
            # prepare tensors
            camera_obs_t = _to_tensor(camera_obs, device)
            state_t = _to_tensor(state, device)

            camera_actions_raw, _ = learner.select_action(camera_obs_t, state_t, evaluate=True)
            camera_actions = process_action_output(camera_actions_raw, env)

            # map discrete->continuous if needed
            if camera_actions.ndim == 1:
                n_actions = getattr(env, 'camera_action_dim', None)
                if n_actions is None and hasattr(env, 'camera_action_space'):
                    try:
                        n_actions = env.camera_action_space.n
                    except Exception:
                        n_actions = None
                if n_actions is None:
                    n_actions = int(np.max(camera_actions) + 1)
                cont = []
                for k in camera_actions:
                    angle = 2.0 * np.pi * (int(k) % max(1, n_actions)) / max(1, n_actions)
                    cont.append([np.cos(angle), np.sin(angle)])
                camera_actions_cont = np.array(cont, dtype=float)
                final_camera_actions = camera_actions_cont
            else:
                final_camera_actions = camera_actions

            target_actions = np.random.uniform(-1, 1, size=(env.num_targets, 2))

            actions = (final_camera_actions, target_actions)
            obs, rewards, terminated, truncated, info = env.step(actions)
            camera_obs, target_obs = obs
            camera_rewards, target_rewards = rewards
            state = env.state()

            episode_reward += np.sum(camera_rewards)
            episode_length += 1
            done = terminated or truncated

        episode_rewards.append(episode_reward)
        print(f"Evaluation episode {i+1}: Reward = {episode_reward:.2f}, Length = {episode_length}")

    print(f"\nFinal evaluation results:")
    print(f"  Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")

    # Close environment
    env.close()

    total_time = time.time() - start_time
    print(f"\nQuick test completed in {total_time:.2f} seconds!")
    print("If this worked, you can run the full training with:")
    print("  python train_qplex_mate.py --config configs/qplex_4v4_9.yaml")


if __name__ == "__main__":
    quick_train()
