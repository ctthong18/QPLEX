import torch
from algorithms.mappo.algorithm.actor_critic import Actor, Critic


class MAPPOPolicy:
    """
    MAPPO Policy class. Wraps actor and critic networks to compute actions and value function predictions.
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = Critic(args, self.share_obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr, eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def _select_camera_group_from_tensor(self, x):
        """
        If actor/critic output has a 'groups' dimension like (B, G, N, ...),
        select group 0 (camera) and keep batch dim.
        Works for torch.Tensor and numpy arrays.
        """
        if x is None:
            return None

        # Torch tensor
        if isinstance(x, torch.Tensor):
            if x.dim() == 4 and x.size(1) == 2:
                # (B, G=2, N, ...)
                return x[:, 0].contiguous()  # result (B, N, ...)
            # if shape is (G, N, ...) without batch, try to handle too
            if x.dim() == 3 and x.size(0) == 2:
                return x[0:1].contiguous()  # make batch dim (1, N, ...)
            return x
        else:
            # numpy array
            import numpy as _np
            if isinstance(x, _np.ndarray):
                if x.ndim == 4 and x.shape[1] == 2:
                    return x[:, 0].copy()
                if x.ndim == 3 and x.shape[0] == 2:
                    return x[0:1].copy()
                return x
            return x

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        """
        if hasattr(self, '_update_linear_schedule'):
            self._update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
            self._update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)
        else:
            frac = 1.0 - (episode / episodes)
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = self.lr * frac
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = self.critic_lr * frac

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        Returns values, actions, action_log_probs, rnn_states_actor, rnn_states_critic.
        """

        actions, action_log_probs, rnn_states_actor = self.actor(obs,
                                                                 rnn_states_actor,
                                                                 masks,
                                                                 available_actions,
                                                                 deterministic)

        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)

        # --- Normalize outputs: if actor/critic returned multiple groups (e.g., camera+target),
        #     select camera group (index 0). This keeps shapes compatible with buffer which expects
        #     (batch, n_agents, ...)
        actions = self._select_camera_group_from_tensor(actions)
        action_log_probs = self._select_camera_group_from_tensor(action_log_probs)
        values = self._select_camera_group_from_tensor(values)
        rnn_states_actor = self._select_camera_group_from_tensor(rnn_states_actor)
        rnn_states_critic = self._select_camera_group_from_tensor(rnn_states_critic)

        # Debug prints (optional) - you can comment out after verifying
        # print("DEBUG get_actions shapes:", getattr(values, 'shape', None), getattr(actions, 'shape', None),
        #       getattr(action_log_probs, 'shape', None))

        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        Get value function predictions.
        """
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        values = self._select_camera_group_from_tensor(values)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                     available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        """
        # --- FIX: ensure RNN states are not None
        if rnn_states_actor is None:
            # assume no RNN → create dummy zero state
            # shape should match (batch_size, n_agents, rnn_hidden_dim)
            batch_size = obs.shape[0]
            n_agents = obs.shape[1]
            hidden_dim = getattr(self.actor, "recurrent_N", 1) * getattr(self.actor, "rnn_hidden_dim", 1)
            rnn_states_actor = torch.zeros((batch_size, n_agents, hidden_dim), device=self.device)

        if rnn_states_critic is None:
            batch_size = cent_obs.shape[0]
            n_agents = cent_obs.shape[1] if cent_obs.ndim > 2 else 1
            hidden_dim = getattr(self.critic, "recurrent_N", 1) * getattr(self.critic, "rnn_hidden_dim", 1)
            rnn_states_critic = torch.zeros((batch_size, n_agents, hidden_dim), device=self.device)
        # ---

        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            obs, rnn_states_actor, action, masks, available_actions, active_masks)

        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        action_log_probs = self._select_camera_group_from_tensor(action_log_probs)
        values = self._select_camera_group_from_tensor(values)

        return values, action_log_probs, dist_entropy


    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        """
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)

        # Only keep camera group if actor returned multiple groups
        actions = self._select_camera_group_from_tensor(actions)
        rnn_states_actor = self._select_camera_group_from_tensor(rnn_states_actor)
        return actions, rnn_states_actor

    def prep_training(self):
        """Set networks to training mode."""
        self.actor.train()
        self.critic.train()

    def prep_rollout(self):
        """Set networks to evaluation mode."""
        self.actor.eval()
        self.critic.eval()
