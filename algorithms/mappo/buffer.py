import numpy as np
import torch
from algorithms.utils.util import check


class SharedReplayBuffer:
    """
    Buffer for storing rollout data for MAPPO.
    Supports both feed-forward and recurrent policies.
    """
    def __init__(self, args, num_agents, obs_space, share_obs_space, act_space, device='cpu'):
        """
        Initialize buffer.
        :param args: (dict or argparse.Namespace) arguments
        :param num_agents: (int) number of agents
        :param obs_space: (gym.Space) observation space
        :param share_obs_space: (gym.Space) shared observation space
        :param act_space: (gym.Space) action space
        :param device: (torch.device) device
        """
        # Convert args to object if it's a dict
        if isinstance(args, dict):
            class Args:
                pass
            args_obj = Args()
            for k, v in args.items():
                setattr(args_obj, k, v)
            args = args_obj

        self.num_agents = num_agents
        self.device = device
        
        # Get shapes
        if hasattr(obs_space, 'shape'):
            obs_shape = obs_space.shape
        else:
            obs_shape = (obs_space.n,)
            
        if hasattr(share_obs_space, 'shape'):
            share_obs_shape = share_obs_space.shape
        else:
            share_obs_shape = (share_obs_space.n,)
            
        if hasattr(act_space, 'shape'):
            act_shape = act_space.shape
        else:
            act_shape = (act_space.n,)

        self.episode_length = args.episode_length if hasattr(args, 'episode_length') else 200
        self.n_rollout_threads = args.n_rollout_threads if hasattr(args, 'n_rollout_threads') else 1
        self.use_recurrent_policy = args.use_recurrent_policy if hasattr(args, 'use_recurrent_policy') else False
        self.use_naive_recurrent_policy = args.use_naive_recurrent_policy if hasattr(args, 'use_naive_recurrent_policy') else False
        self.recurrent_N = args.recurrent_N if hasattr(args, 'recurrent_N') else 1
        self.hidden_size = args.hidden_size if hasattr(args, 'hidden_size') else 64
        self.gamma = args.gamma if hasattr(args, 'gamma') else 0.99
        self.gae_lambda = args.gae_lambda if hasattr(args, 'gae_lambda') else 0.95

        # Initialize buffers
        self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *obs_shape), dtype=np.float32)
        self.share_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, *share_obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.episode_length, self.n_rollout_threads, num_agents, *act_shape), dtype=np.float32)
        
        if len(act_shape) == 0:
            # Discrete actions
            self.action_one_hot = np.zeros((self.episode_length, self.n_rollout_threads, num_agents, act_space.n), dtype=np.float32)
        else:
            self.action_one_hot = None
            
        self.rewards = np.zeros((self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.value_preds = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.returns = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.active_masks = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.action_log_probs = np.zeros((self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        
        if self.use_recurrent_policy or self.use_naive_recurrent_policy:
            self.rnn_states = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            self.rnn_states_critic = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        else:
            self.rnn_states = None
            self.rnn_states_critic = None
            
        self.available_actions = None

        self.step = 0

    def insert(self, share_obs, obs, rnn_states_actor, rnn_states_critic, actions, action_log_probs,
               value_preds, rewards, masks, active_masks=None, available_actions=None):
        """
        Insert data into buffer.
        """
        self.share_obs[self.step + 1] = share_obs.copy()
        self.obs[self.step + 1] = obs.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if self.use_recurrent_policy or self.use_naive_recurrent_policy:
            self.rnn_states[self.step + 1] = rnn_states_actor.copy()
            self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        if available_actions is not None:
            if self.available_actions is None:
                self.available_actions = np.zeros((self.episode_length + 1, self.n_rollout_threads, self.num_agents, *available_actions.shape[2:]), dtype=np.float32)
            self.available_actions[self.step + 1] = available_actions.copy()
            
        self.step = (self.step + 1) % self.episode_length

    def compute_returns(self, next_value, value_normalizer=None):
        """
        Compute returns using GAE.
        """
        if value_normalizer is not None:
            next_value = value_normalizer.denormalize(next_value)
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.episode_length)):
            if step == self.episode_length - 1:
                nextnonterminal = 1.0 - self.masks[step + 1]
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - self.masks[step + 1]
                nextvalues = self.value_preds[step + 1]
            delta = self.rewards[step] + self.gamma * nextvalues * nextnonterminal - self.value_preds[step]
            gae = delta + self.gamma * self.gae_lambda * nextnonterminal * gae
            self.returns[step] = gae + self.value_preds[step]

    def after_update(self):
        """Copy last obs to first position after update."""
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        if self.use_recurrent_policy or self.use_naive_recurrent_policy:
            self.rnn_states[0] = self.rnn_states[-1].copy()
            self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        """
        Generator for feed-forward policy training.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0], self.n_rollout_threads, self.num_agents
        batch_size = n_rollout_threads * episode_length
        
        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                f"PPO requires the number of processes ({batch_size}) "
                f"* number of steps ({episode_length}) = {batch_size} "
                f"to be greater than or equal to the number of PPO mini batches ({num_mini_batch}).")
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]
        
        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        actions = self.actions.reshape(-1, *self.actions.shape[2:])
        if self.action_one_hot is not None:
            action_one_hot = self.action_one_hot.reshape(-1, *self.action_one_hot.shape[2:])
        else:
            action_one_hot = None
        value_preds = self.value_preds[:-1].reshape(-1, *self.value_preds.shape[2:])
        returns = self.returns[:-1].reshape(-1, *self.returns.shape[2:])
        masks = self.masks[:-1].reshape(-1, *self.masks.shape[2:])
        active_masks = self.active_masks[:-1].reshape(-1, *self.active_masks.shape[2:])
        old_action_log_probs = self.action_log_probs.reshape(-1, *self.action_log_probs.shape[2:])
        advantages = advantages.reshape(-1, advantages.shape[-1])
        
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[2:]) if self.rnn_states is not None else None
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[2:]) if self.rnn_states_critic is not None else None
        available_actions = self.available_actions[:-1].reshape(-1, *self.available_actions.shape[2:]) if self.available_actions is not None else None

        for indices in sampler:
            share_obs_batch = share_obs[indices]
            obs_batch = obs[indices]
            rnn_states_batch = rnn_states[indices] if rnn_states is not None else None
            rnn_states_critic_batch = rnn_states_critic[indices] if rnn_states_critic is not None else None
            actions_batch = actions[indices]
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = old_action_log_probs[indices]
            adv_targ = advantages[indices]
            available_actions_batch = available_actions[indices] if available_actions is not None else None

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
                  value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
                  adv_targ, available_actions_batch

    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        """
        Generator for recurrent policy training.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0], self.n_rollout_threads, self.num_agents
        batch_size = n_rollout_threads * episode_length // data_chunk_length
        mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]

        # Reshape for recurrent training
        share_obs = self.share_obs[:-1].reshape(episode_length // data_chunk_length, n_rollout_threads * data_chunk_length, *self.share_obs.shape[2:])
        obs = self.obs[:-1].reshape(episode_length // data_chunk_length, n_rollout_threads * data_chunk_length, *self.obs.shape[2:])
        actions = self.actions.reshape(episode_length // data_chunk_length, n_rollout_threads * data_chunk_length, *self.actions.shape[2:])
        value_preds = self.value_preds[:-1].reshape(episode_length // data_chunk_length, n_rollout_threads * data_chunk_length, *self.value_preds.shape[2:])
        returns = self.returns[:-1].reshape(episode_length // data_chunk_length, n_rollout_threads * data_chunk_length, *self.returns.shape[2:])
        masks = self.masks[:-1].reshape(episode_length // data_chunk_length, n_rollout_threads * data_chunk_length, *self.masks.shape[2:])
        active_masks = self.active_masks[:-1].reshape(episode_length // data_chunk_length, n_rollout_threads * data_chunk_length, *self.active_masks.shape[2:])
        old_action_log_probs = self.action_log_probs.reshape(episode_length // data_chunk_length, n_rollout_threads * data_chunk_length, *self.action_log_probs.shape[2:])
        advantages = advantages.reshape(episode_length // data_chunk_length, n_rollout_threads * data_chunk_length, advantages.shape[-1])
        
        rnn_states = self.rnn_states[:-1].reshape(episode_length // data_chunk_length, n_rollout_threads * data_chunk_length, *self.rnn_states.shape[2:]) if self.rnn_states is not None else None
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(episode_length // data_chunk_length, n_rollout_threads * data_chunk_length, *self.rnn_states_critic.shape[2:]) if self.rnn_states_critic is not None else None
        available_actions = self.available_actions[:-1].reshape(episode_length // data_chunk_length, n_rollout_threads * data_chunk_length, *self.available_actions.shape[2:]) if self.available_actions is not None else None

        for indices in sampler:
            share_obs_batch = share_obs[indices]
            obs_batch = obs[indices]
            rnn_states_batch = rnn_states[indices] if rnn_states is not None else None
            rnn_states_critic_batch = rnn_states_critic[indices] if rnn_states_critic is not None else None
            actions_batch = actions[indices]
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = old_action_log_probs[indices]
            adv_targ = advantages[indices]
            available_actions_batch = available_actions[indices] if available_actions is not None else None

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
                  value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
                  adv_targ, available_actions_batch

    def naive_recurrent_generator(self, advantages, num_mini_batch):
        """Naive recurrent generator (simpler version)."""
        return self.feed_forward_generator(advantages, num_mini_batch)
