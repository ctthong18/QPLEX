# train_mappo.py
import os
import numpy as np
import sys
import yaml
import argparse
import time
import json
import logging
import numpy as np
import torch
try:
    from gymnasium import spaces
except ImportError:
    from gym import spaces

# add project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import MATE env
import mate
from mate.environment import MultiAgentTracking
from mate.agents import GreedyTargetAgent, RandomTargetAgent

from algorithms.mappo.mappo import MAPPO
from algorithms.mappo.algorithm.mappo_policy import MAPPOPolicy
from algorithms.mappo.buffer import SharedReplayBuffer


def setup_logging(log_dir: str, log_level: str = "INFO"):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("MAPPO")
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


def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate(policy: MAPPOPolicy, env: MultiAgentTracking, device, n_eval_episodes=5, render=False):
    """Evaluate policy."""
    policy.prep_rollout()
    episode_rewards = []
    episode_lengths = []
    coverage_rates = []
    transport_rates = []
    
    for ep in range(n_eval_episodes):
        obs, info = env.reset()
        camera_obs, _ = obs
        state = env.state()
        done = False
        ep_reward = 0.0
        ep_len = 0
        
        # Initialize RNN states if needed
        if policy.actor._use_recurrent_policy or policy.actor._use_naive_recurrent_policy:
            rnn_states_actor = np.zeros((1, env.num_cameras, policy.actor._recurrent_N, policy.actor.hidden_size), dtype=np.float32)
            rnn_states_critic = np.zeros((1, policy.critic._recurrent_N, policy.critic.hidden_size), dtype=np.float32)
        else:
            rnn_states_actor = None
            rnn_states_critic = None
            
        masks = np.ones((1, env.num_cameras, 1), dtype=np.float32)
        
        while not done:
            # Prepare inputs
            camera_obs_expanded = camera_obs[np.newaxis, :]  # (1, n_agents, obs_dim)
            state_expanded = state[np.newaxis, :]  # (1, state_dim)

            # Get actions
            with torch.no_grad():
                values, actions, action_log_probs, rnn_states_actor, rnn_states_critic = policy.get_actions(
                    state_expanded, camera_obs_expanded, rnn_states_actor, rnn_states_critic, 
                    masks, deterministic=True
                )
            
            # Get actions for environment - ensure correct shape
            actions_np_eval = actions.cpu().numpy()
            if actions_np_eval.ndim == 3:
                actions_eval = actions_np_eval[0]  # (n_agents, act_dim)
            elif actions_np_eval.ndim == 2:
                actions_eval = actions_np_eval  # Already (n_agents, act_dim)
            else:
                actions_eval = actions_np_eval.flatten()[:env.num_cameras * 2].reshape(env.num_cameras, 2)
            
            actions_eval = np.asarray(actions_eval, dtype=np.float64).reshape(env.num_cameras, 2)
            if rnn_states_actor is not None:
                rnn_states_actor = rnn_states_actor.cpu().numpy()
            if rnn_states_critic is not None:
                rnn_states_critic = rnn_states_critic.cpu().numpy()
            
            target_actions = np.zeros((env.num_targets, 2), dtype=np.float64)
            joint_actions = (actions_eval, target_actions)
            
            next_obs, rewards, terminated, truncated, info = env.step(joint_actions)
            next_camera_obs, _ = next_obs
            camera_rewards, _ = rewards
            next_state = env.state()
            done = terminated or truncated
            
            ep_reward += np.sum(camera_rewards)
            ep_len += 1
            camera_obs = next_camera_obs
            state = next_state
            
            if done:
                masks = np.zeros((1, env.num_cameras, 1), dtype=np.float32)
            else:
                masks = np.ones((1, env.num_cameras, 1), dtype=np.float32)
                
            if render:
                env.render()
        
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)
        
        # Extract metrics from info
        if info and len(info) > 0:
            camera_infos, target_infos = info
            if camera_infos and len(camera_infos) > 0:
                coverage_rates.append(camera_infos[0].get('coverage_rate', 0.0))
                transport_rates.append(camera_infos[0].get('mean_transport_rate', 0.0))
        
    return {
        'mean_episode_reward': float(np.mean(episode_rewards)),
        'std_episode_reward': float(np.std(episode_rewards)),
        'mean_episode_length': float(np.mean(episode_lengths)),
        'mean_coverage_rate': float(np.mean(coverage_rates)) if coverage_rates else 0.0,
        'mean_transport_rate': float(np.mean(transport_rates)) if transport_rates else 0.0
    }


def train(config_path: str, device: str = 'cpu', log_level: str = 'INFO'):
    config = load_config(config_path)
    logger = setup_logging(config['logging']['log_dir'], log_level)
    logger.info("Loaded config: %s", config_path)

    # seed
    seed = config.get('seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Using device %s", device)

    # create env
    env = MultiAgentTracking(config=config['env']['config_file'],
                             render_mode=config['env'].get('render_mode', None),
                             window_size=config['env'].get('window_size', 800))
    logger.info("Env created")

    obs_dim = env.camera_observation_dim
    act_dim = 2
    state_dim = int(env.state_space.shape[0])
    n_agents = int(env.num_cameras)

    logger.info("obs_dim=%d, act_dim=%d, state_dim=%d, n_agents=%d", obs_dim, act_dim, state_dim, n_agents)

    # Create observation and action spaces
    obs_space = env.camera_observation_space
    share_obs_space = env.state_space
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)

    # Create args object
    class Args:
        pass
    args = Args()
    
    # Algorithm config
    algo_config = config['algorithm']
    args.hidden_size = algo_config.get('hidden_size', 64)
    args.layer_N = algo_config.get('layer_N', 1)
    args.use_orthogonal = algo_config.get('use_orthogonal', True)
    args.use_ReLU = algo_config.get('use_ReLU', True)
    args.gain = algo_config.get('gain', 0.01)
    args.use_naive_recurrent_policy = algo_config.get('use_naive_recurrent_policy', False)
    args.use_recurrent_policy = algo_config.get('use_recurrent_policy', False)
    args.recurrent_N = algo_config.get('recurrent_N', 1)
    args.use_feature_normalization = algo_config.get('use_feature_normalization', False)
    args.stacked_frames = algo_config.get('stacked_frames', 1)
    args.use_popart = algo_config.get('use_popart', False)
    args.use_policy_active_masks = algo_config.get('use_policy_active_masks', False)
    
    # Training config
    args.lr = float(algo_config.get('lr', 3e-4))
    args.critic_lr = float(algo_config.get('critic_lr', 3e-4))
    args.opti_eps = float(algo_config.get('opti_eps', 1e-5))
    args.weight_decay = algo_config.get('weight_decay', 0)
    
    # MAPPO config
    args.clip_param = algo_config.get('clip_param', 0.2)
    args.ppo_epoch = algo_config.get('ppo_epoch', 10)
    args.num_mini_batch = algo_config.get('num_mini_batch', 1)
    args.data_chunk_length = algo_config.get('data_chunk_length', 10)
    args.value_loss_coef = algo_config.get('value_loss_coef', 0.5)
    args.entropy_coef = algo_config.get('entropy_coef', 0.01)
    args.max_grad_norm = algo_config.get('max_grad_norm', 10.0)
    args.huber_delta = algo_config.get('huber_delta', 10.0)
    args.use_max_grad_norm = algo_config.get('use_max_grad_norm', True)
    args.use_clipped_value_loss = algo_config.get('use_clipped_value_loss', True)
    args.use_huber_loss = algo_config.get('use_huber_loss', False)
    args.use_valuenorm = algo_config.get('use_valuenorm', False)
    args.use_value_active_masks = algo_config.get('use_value_active_masks', False)
    
    # Buffer config
    args.episode_length = algo_config.get('episode_length', 200)
    args.n_rollout_threads = algo_config.get('n_rollout_threads', 1)
    args.gamma = algo_config.get('gamma', 0.99)
    args.gae_lambda = algo_config.get('gae_lambda', 0.95)

    # Create policy
    policy = MAPPOPolicy(args, obs_space, share_obs_space, act_space, device)
    logger.info("Policy created")

    # Create trainer
    trainer = MAPPO(args, policy, device)
    logger.info("MAPPO trainer created")

    # Create buffer
    buffer = SharedReplayBuffer(args, n_agents, obs_space, share_obs_space, act_space, device)
    logger.info("Buffer created")

    total_steps = config['training']['total_timesteps']
    log_interval = config['logging'].get('log_interval', 2000)
    eval_interval = config['logging'].get('eval_interval', 2000)
    save_interval = config['logging'].get('save_interval', 10000)
    model_dir = config['logging']['model_dir']
    os.makedirs(model_dir, exist_ok=True)

    # Save config
    with open(os.path.join(config['logging']['log_dir'], 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Training loop
    logger.info("Starting training...")
    t0 = time.time()
    obs, info = env.reset()
    camera_obs, _ = obs
    state = env.state()
    
    # Initialize RNN states
    if args.use_recurrent_policy or args.use_naive_recurrent_policy:
        rnn_states_actor = np.zeros((args.n_rollout_threads, n_agents, args.recurrent_N, args.hidden_size), dtype=np.float32)
        rnn_states_critic = np.zeros((args.n_rollout_threads, args.recurrent_N, args.hidden_size), dtype=np.float32)
    else:
        rnn_states_actor = None
        rnn_states_critic = None
    
    masks = np.ones((args.n_rollout_threads, n_agents, 1), dtype=np.float32)
    active_masks = np.ones((args.n_rollout_threads, n_agents, 1), dtype=np.float32)
    
    episode_count = 0
    episode_reward = 0.0
    episode_len = 0

    for step in range(total_steps):
        # Prepare observations
        camera_obs_expanded = camera_obs[np.newaxis, :]  # (1, n_agents, obs_dim)
        state_expanded = state[np.newaxis, :]  # (1, state_dim)

        # Get actions
        policy.prep_rollout()
        with torch.no_grad():
            values, actions, action_log_probs, rnn_states_actor, rnn_states_critic = policy.get_actions(
                state_expanded, camera_obs_expanded, rnn_states_actor, rnn_states_critic,
                masks, deterministic=False
            )

        # Convert to numpy and ensure correct shapes for buffer
        # Buffer expects: (n_rollout_threads, n_agents, ...)
        actions_np = actions.cpu().numpy()
        action_log_probs_np = action_log_probs.cpu().numpy()
        values_np = values.cpu().numpy()
        
        # Handle actions: expect (batch, n_agents, act_dim) or (n_agents, act_dim)
        # Ensure actions_np has shape (1, n_agents, act_dim) for buffer
        if actions_np.ndim == 1:
            # Flattened, reshape to (1, n_agents, act_dim)
            actions_np = actions_np.reshape(-1, n_agents, act_dim)[:1]
        elif actions_np.ndim == 2:
            # (n_agents, act_dim) -> (1, n_agents, act_dim)
            if actions_np.shape[0] == n_agents and actions_np.shape[1] == act_dim:
                actions_np = actions_np[np.newaxis, :]  # (1, n_agents, act_dim)
            else:
                # Unexpected shape, try to reshape
                actions_np = actions_np.reshape(-1, n_agents, act_dim)[:1]
        elif actions_np.ndim == 3:
            # Already (batch, n_agents, act_dim), take first batch
            if actions_np.shape[1] == n_agents and actions_np.shape[2] == act_dim:
                actions_np = actions_np[0:1]  # (1, n_agents, act_dim)
            else:
                # Wrong shape, reshape
                actions_np = actions_np.reshape(-1, n_agents, act_dim)[:1]
        else:
            # Unexpected ndim, flatten and reshape
            actions_np = actions_np.flatten()[:n_agents * act_dim].reshape(1, n_agents, act_dim)
        
        # Handle action_log_probs: expect (batch, n_agents, 1) or (batch*n_agents, 1) or (n_agents, 1)
        if action_log_probs_np.ndim == 1:  # (batch*n_agents,) or (n_agents,)
            # Reshape assuming it's flattened
            if len(action_log_probs_np) == n_agents:
                action_log_probs_np = action_log_probs_np[:, np.newaxis][np.newaxis, :, :]  # (1, n_agents, 1)
            else:
                # Assume it's (batch*n_agents,), reshape to (batch, n_agents, 1)
                batch_size = len(action_log_probs_np) // n_agents
                action_log_probs_np = action_log_probs_np.reshape(batch_size, n_agents, 1)[:1]  # (1, n_agents, 1)
        elif action_log_probs_np.ndim == 2:
            if action_log_probs_np.shape[1] == 1:  # (batch*n_agents, 1) or (n_agents, 1)
                if len(action_log_probs_np) == n_agents:
                    action_log_probs_np = action_log_probs_np[np.newaxis, :, :]  # (1, n_agents, 1)
                else:
                    batch_size = len(action_log_probs_np) // n_agents
                    action_log_probs_np = action_log_probs_np.reshape(batch_size, n_agents, 1)[:1]  # (1, n_agents, 1)
            else:  # (batch, n_agents)
                action_log_probs_np = action_log_probs_np[:, :, np.newaxis][:1]  # (1, n_agents, 1)
        elif action_log_probs_np.ndim == 3:
            action_log_probs_np = action_log_probs_np[:1]  # (1, n_agents, 1)
        
        # Handle values: Critic is centralized, returns (batch, 1) for centralized value
        # Buffer expects per-agent values: (n_rollout_threads, n_agents, 1)
        if values_np.ndim == 1:  # (batch,) or (n_agents,)
            if len(values_np) == 1:
                # Centralized value (1,), expand to per-agent
                values_np = np.repeat(values_np[np.newaxis, np.newaxis, :], n_agents, axis=1)  # (1, n_agents, 1)
            elif len(values_np) == n_agents:
                # Per-agent values
                values_np = values_np[np.newaxis, :, np.newaxis]  # (1, n_agents, 1)
            else:
                # Take first and expand
                values_np = np.repeat(values_np[0:1][np.newaxis, np.newaxis, :], n_agents, axis=1)  # (1, n_agents, 1)
        elif values_np.ndim == 2:
            if values_np.shape[1] == 1:  # (batch, 1) - centralized value
                # Expand to per-agent: take first batch, repeat for n_agents
                values_np = np.repeat(values_np[0:1, np.newaxis, :], n_agents, axis=1)  # (1, n_agents, 1)
            elif values_np.shape[1] == n_agents:  # (batch, n_agents)
                values_np = values_np[0:1, :, np.newaxis]  # (1, n_agents, 1)
            else:
                # Unexpected shape, take first and expand
                values_np = np.repeat(values_np[0:1, 0:1, np.newaxis], n_agents, axis=1)  # (1, n_agents, 1)
        elif values_np.ndim == 3:
            values_np = values_np[0:1]  # (1, n_agents, 1)
        
        if rnn_states_actor is not None:
            rnn_states_actor = rnn_states_actor.cpu().numpy()
        if rnn_states_critic is not None:
            rnn_states_critic = rnn_states_critic.cpu().numpy()

        # Step environment - ensure camera_actions has correct shape (n_agents, act_dim)
        # actions_np should be (1, n_agents, act_dim) after processing
        if actions_np.ndim == 3:
            camera_actions = actions_np[0]  # (n_agents, act_dim)
        elif actions_np.ndim == 2:
            camera_actions = actions_np  # Already (n_agents, act_dim)
        else:
            # Flatten and reshape if needed
            camera_actions = actions_np.flatten()[:n_agents * act_dim].reshape(n_agents, act_dim)
        
        # Ensure shape is exactly (n_agents, act_dim)
        camera_actions = np.asarray(camera_actions, dtype=np.float64).reshape(n_agents, act_dim)
        target_actions = np.random.uniform(-1, 1, (env.num_targets, 2)).astype(np.float64)
        joint_actions = (camera_actions, target_actions)

        next_obs, rewards, terminated, truncated, info = env.step(joint_actions)
        next_camera_obs, _ = next_obs
        camera_rewards, _ = rewards
        next_state = env.state()
        done = terminated or truncated

        # Store in buffer - ensure rewards have correct shape (n_rollout_threads, n_agents, 1)
        if isinstance(camera_rewards, (float, int)):
            camera_rewards = np.array([camera_rewards] * n_agents, dtype=np.float32)
        elif not isinstance(camera_rewards, np.ndarray):
            camera_rewards = np.array(camera_rewards, dtype=np.float32)
        
        # Ensure rewards have shape (n_rollout_threads, n_agents, 1)
        if camera_rewards.ndim == 1:  # (n_agents,)
            rewards_expanded = camera_rewards[np.newaxis, :, np.newaxis]  # (1, n_agents, 1)
        elif camera_rewards.ndim == 0:  # scalar
            rewards_expanded = np.array([[[camera_rewards]]] * n_agents, dtype=np.float32).T  # (1, n_agents, 1)
        else:
            # Already has some shape, ensure it's (1, n_agents, 1)
            rewards_expanded = camera_rewards.reshape(-1)[:n_agents][np.newaxis, :, np.newaxis]  # (1, n_agents, 1)
        
        buffer.insert(
            state_expanded, camera_obs_expanded,
            rnn_states_actor, rnn_states_critic,
            actions_np,  # (1, n_agents, act_dim)
            action_log_probs_np,  # (1, n_agents, 1)
            values_np,  # (1, n_agents, 1)
            rewards_expanded,  # (1, n_agents, 1)
            masks,
            active_masks
        )

        episode_reward += camera_rewards.sum()
        episode_len += 1

        # Update masks
        if done:
            masks = np.zeros((args.n_rollout_threads, n_agents, 1), dtype=np.float32)
            active_masks = np.zeros((args.n_rollout_threads, n_agents, 1), dtype=np.float32)
        else:
            masks = np.ones((args.n_rollout_threads, n_agents, 1), dtype=np.float32)
            active_masks = np.ones((args.n_rollout_threads, n_agents, 1), dtype=np.float32)

        # Update if buffer is full
        if buffer.step == 0:
            # Compute next value for GAE
            if not done:
                with torch.no_grad():
                    next_state_expanded = next_state[np.newaxis, :]
                    if args.use_recurrent_policy or args.use_naive_recurrent_policy:
                        next_rnn_states_critic = rnn_states_critic
                    else:
                        next_rnn_states_critic = None
                    next_masks = masks
                    next_values = policy.get_values(next_state_expanded, next_rnn_states_critic, next_masks)
                    next_values = next_values.cpu().numpy()
                    # Critic returns shape (n_rollout_threads, 1) for centralized value
                    next_value = float(next_values.mean()) if next_values.size > 0 else 0.0
            else:
                next_value = 0.0
            
            # Compute returns
            buffer.compute_returns(next_value, trainer.value_normalizer)
            
            # Train
            policy.prep_training()
            train_info = trainer.train(buffer, update_actor=True)
            # logger.info("Update @ step %d: %s", step, train_info)
            
            # After update
            buffer.after_update()
            
            # Reset RNN states if needed
            if args.use_recurrent_policy or args.use_naive_recurrent_policy:
                rnn_states_actor = buffer.rnn_states[0].copy()
                rnn_states_critic = buffer.rnn_states_critic[0].copy()

        # Move to next state
        camera_obs = next_camera_obs
        state = next_state

        if done:
            episode_count += 1
            if episode_count % 10 == 0:
                logger.info("Episode %d reward=%.3f length=%d", episode_count, episode_reward, episode_len)
            
            # Reset env
            obs, info = env.reset()
            camera_obs, _ = obs
            state = env.state()
            episode_reward = 0.0
            episode_len = 0

            # Reset masks
            masks = np.ones((args.n_rollout_threads, n_agents, 1), dtype=np.float32)
            active_masks = np.ones((args.n_rollout_threads, n_agents, 1), dtype=np.float32)

        # Periodic logging / evaluation / saving
        if step > 0 and step % log_interval == 0:
            elapsed = time.time() - t0
            # Get training stats (adapt from train_info or buffer)
            training_stats = {
                'mean_episode_reward': float(np.mean([episode_reward])),  # Placeholder; adapt from buffer if needed
                'mean_loss': train_info.get('explained_variance', 0.0) if train_info else 0.0,  # Example from train_info
                'mean_q_values': 0.0,  # Placeholder for MAPPO (no Q-values)
                'mean_epsilon': 0.0,  # Placeholder
                'buffer_size': buffer.current_size if hasattr(buffer, 'current_size') else buffer.step,
                'episode_count': episode_count,
            }
            logger.info("Step %d / %d, elapsed %.1fs", step, total_steps, elapsed)
            logger.info("  Episode count: %d", episode_count)
            logger.info("  Mean episode reward: %.2f", training_stats['mean_episode_reward'])
            logger.info("  Mean loss: %.4f", training_stats['mean_loss'])
            logger.info("  Buffer size: %d", training_stats['buffer_size'])
            
            # Save training statistics
            stats_file = os.path.join(config['logging']['log_dir'], 'training_stats.json')
            with open(stats_file, 'w') as f:
                json.dump(training_stats, f, indent=2)

        if step > 0 and step % eval_interval == 0:
            eval_res = evaluate(policy, env, device, n_eval_episodes=config['evaluation'].get('n_eval_episodes', 5))
            logger.info("Evaluation @ step %d:", step)
            logger.info("  Mean episode reward: %.2f ± %.2f", eval_res['mean_episode_reward'], eval_res['std_episode_reward'])
            logger.info("  Mean episode length: %.2f", eval_res['mean_episode_length'])
            logger.info("  Mean coverage rate: %.4f", eval_res['mean_coverage_rate'])
            logger.info("  Mean transport rate: %.4f", eval_res['mean_transport_rate'])
            # save eval
            eval_file = os.path.join(config['logging']['log_dir'], f"eval_results_{step}.json")
            with open(eval_file, 'w') as f:
                json.dump(eval_res, f, indent=2)

        if step > 0 and step % save_interval == 0:
            save_path = os.path.join(model_dir, f"mappo_{step}.pt")
            torch.save({
                'actor_state_dict': policy.actor.state_dict(),
                'critic_state_dict': policy.critic.state_dict(),
                'actor_optimizer': policy.actor_optimizer.state_dict(),
                'critic_optimizer': policy.critic_optimizer.state_dict(),
            }, save_path)
            logger.info("Saved model to %s", save_path)

    # Final evaluation
    logger.info("Final evaluation...")
    n_eval_episodes = config['evaluation'].get('n_eval_episodes', 5)
    final_eval_res = evaluate(policy, env, device, n_eval_episodes * 2)
    
    logger.info("Final evaluation results:")
    logger.info("  Mean episode reward: %.2f ± %.2f", final_eval_res['mean_episode_reward'], final_eval_res['std_episode_reward'])
    logger.info("  Mean episode length: %.2f", final_eval_res['mean_episode_length'])
    logger.info("  Mean coverage rate: %.4f", final_eval_res['mean_coverage_rate'])
    logger.info("  Mean transport rate: %.4f", final_eval_res['mean_transport_rate'])
    
    # Save final evaluation results
    final_eval_file = os.path.join(config['logging']['log_dir'], 'mappo_final_eval_results.json')
    with open(final_eval_file, 'w') as f:
        json.dump(final_eval_res, f, indent=2)
    
    # Final save
    final_path = os.path.join(model_dir, "mappo_final.pt")
    torch.save({
        'actor_state_dict': policy.actor.state_dict(),
        'critic_state_dict': policy.critic.state_dict(),
        'actor_optimizer': policy.actor_optimizer.state_dict(),
        'critic_optimizer': policy.critic_optimizer.state_dict(),
    }, final_path)
    logger.info("Finished training. Saved final model to %s", final_path)
    logger.info("Training completed!")
    env.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mappo_4v4_9.yaml")
    parser.add_argument("--device", type=str, default="cpu")
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
    
    logger.info("Starting MAPPO training on MATE environment")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Seed: {config.get('seed', 42)}")
    
    try:
        train(args.config, args.device, args.log_level)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()