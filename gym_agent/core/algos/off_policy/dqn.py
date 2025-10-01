import random
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from torch import Tensor

from ...agent_base import ActType, ObsType, OffPolicyAgent
from ...polices import TargetPolicy
from ...transforms import EnvWithTransform

from ....utils import to_torch


class DQN(OffPolicyAgent):
    policy: TargetPolicy

    def __init__(
        self,
        policy: TargetPolicy,
        env_factory_fn: Callable[[], EnvWithTransform] | str,
        env_kwargs: Optional[dict[str, Any]] = None,
        num_envs: int = 1,
        n_steps: int = 1,
        gamma=0.99,
        eps_start=1.0,
        eps_decay=0.99997,
        eps_decay_steps: Optional[int]=None,
        eps_end=0.01,
        tau=1e-3,
        buffer_size=100000,
        batch_size=64,
        async_vectorization: bool = True,
        device="auto",
        seed=None,
    ):
        super().__init__(
            policy,
            env_factory_fn=env_factory_fn,
            env_kwargs=env_kwargs,
            num_envs=num_envs,
            n_steps=n_steps,
            gamma=gamma,
            buffer_size=buffer_size,
            batch_size=batch_size,
            async_vectorization=async_vectorization,
            device=device,
            seed=seed,
            supported_action_spaces=(
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        self.eps_start = eps_start
        self.eps_decay = eps_decay
        if eps_decay_steps is not None:
            self.eps_decay = (eps_end/eps_start) ** (1/eps_decay_steps)
        self.eps_end = eps_end

        # Soft update parameter
        self.tau = tau

        # Initialize epsilon for epsilon-greedy policy
        self.eps = eps_start

        self.eps_hist = [self.eps]

        self.add_save_kwargs("eps")

    def reset(self):
        self.eps = max(self.eps_end, self.eps_decay * self.eps)
        self.eps_hist.append(self.eps)
        return super().reset()

    @torch.no_grad()
    def predict(self, state: ObsType, deterministic=True) -> ActType:
        # Determine epsilon value based on evaluation mode
        if deterministic:
            eps = 0
        else:
            eps = self.eps

        # Epsilon-greedy action selection
        if random.random() >= eps:
            # Convert state to tensor and move to the appropriate device
            state = to_torch(state, self.device).float()

            # Set local model to evaluation mode
            self.policy.eval()
            # Get action values from the local model
            action_value: Tensor = self.policy.forward(state)
            # Set local model back to training mode
            self.policy.train()

            # Return the action with the highest value
            return np.argmax(action_value.cpu().data.numpy(), axis=1)
        else:
            # Return a random action from the action space
            return np.array(
                [self.envs.single_action_space.sample() for _ in range(state.shape[0])]
            )

    def learn(self):
        buffers = self.memory.sample(self.batch_size)

        observations = buffers.observations
        actions = buffers.actions
        rewards = buffers.rewards
        next_observations = buffers.next_observations
        terminals = buffers.terminals

        # Get the maximum predicted Q values for the next states from the target model
        q_targets_next: Tensor = (
            self.policy.target_forward(next_observations).detach().max(1)[0]
        )
        # Compute the Q targets for the current states
        q_targets: Tensor = rewards + (self.gamma * q_targets_next * (~terminals))

        # Get the expected Q values from the local model
        q_expected: Tensor = self.policy.forward(observations)

        # Deep Q-Learning always expects actions is Discrete
        actions = actions.long()
        if actions.dim() == q_expected.dim() - 1:
            actions = actions.unsqueeze(-1)

        q_expected = q_expected.gather(-1, actions).squeeze(-1)

        # Compute the loss
        loss: Tensor = F.mse_loss(q_expected, q_targets)

        # Minimize the loss
        self.policy.zero_grad()
        loss.backward()
        self.policy.optimizers_step()
        self.policy.lr_schedulers_step()

        # Update the target network
        self.policy.soft_update(self.tau)
