from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from ...agent_base import ActType, ObsType, ActorCriticPolicyAgent
from ...distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    make_proba_distribution,
)
from ...polices import ActorCriticPolicy
from ...transforms import EnvWithTransform

from ....utils import to_torch


class A2C(ActorCriticPolicyAgent):
    policy: ActorCriticPolicy

    def __init__(
        self,
        policy: ActorCriticPolicy,
        env_factory_fn: Callable[[], EnvWithTransform] | str,
        env_kwargs: Optional[dict[str, Any]] = None,
        n_steps: int = 5,
        num_envs: int = 1,
        max_grad_norm: Optional[float] = 0.5,
        gamma=0.99,
        gae_lambda=1.0,
        vf_coef=0.5,
        entropy_coef=0.0,
        normalize_advantage: bool = False,
        device="auto",
        seed=None,
    ):
        super().__init__(
            policy,
            env_factory_fn=env_factory_fn,
            env_kwargs=env_kwargs,
            n_steps=n_steps,
            num_envs=num_envs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            device=device,
            seed=seed,
            supported_action_spaces=(
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
                # spaces.Box, # Continuous action space, not implemented yet
            ),
        )

        self.max_grad_norm = max_grad_norm

        self.normalize_advantage = normalize_advantage
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef
        self.action_dist = make_proba_distribution(self.action_space)

        # If the action space is continuous, we need to learn the standard deviation
        if isinstance(self.action_dist, DiagGaussianDistribution):
            log_std_init = 0
            self.log_std = nn.Parameter(
                torch.ones(self.action_dist.action_dim) * log_std_init,
                requires_grad=True,
            ).to(self.device)

    def distribution(self, action_logits: torch.Tensor) -> Distribution:
        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(action_logits, self.log_std)
        elif isinstance(
            self.action_dist,
            (
                CategoricalDistribution,
                MultiCategoricalDistribution,
                BernoulliDistribution,
            ),
        ):
            # Here action_logits are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=action_logits)
        else:
            raise ValueError("Invalid action distribution")

    def predict(self, state: ObsType, deterministic: bool = True) -> ActType:
        action_logits = self.policy.forward_actor(
            to_torch(state, self.device).float()
        )  # Shape or dict of Shape: (num_envs, action_dim) or (num_envs, n_actions)

        dist = self.distribution(action_logits)
        actions = dist.get_actions(deterministic=deterministic)
        return actions.cpu().numpy()

    def learn(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.

        Parameters:
            memory (RolloutBuffer): Buffer containing the rollouts.
        """
        # Sample from the buffer

        for rollout_data in self.memory.get(self.batch_size):
            actions = (
                rollout_data.actions
            )  # Shape: (batch_size, 1) or (batch_size, action_dim)

            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = actions.long().flatten()  # Shape: (batch_size,)

            log_probs, values, entropy = self.evaluate_actions(
                rollout_data.observations, actions
            )

            advantages = rollout_data.advantages

            if self.normalize_advantage and advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

            # Actor loss
            policy_loss = -(advantages * log_probs).mean()

            # Critic loss
            value_loss = F.mse_loss(rollout_data.returns, values)

            if entropy is None:
                # If the distribution does not have an entropy method, approximate it
                entropy_loss = -torch.mean(-log_probs)
            else:
                entropy_loss = -torch.mean(entropy)

            loss = (
                policy_loss
                + self.vf_coef * value_loss
                + self.entropy_coef * entropy_loss
            )


            self.policy.zero_grad()
            loss.backward()

            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

            self.policy.optimizers_step()
            self.policy.lr_schedulers_step()
