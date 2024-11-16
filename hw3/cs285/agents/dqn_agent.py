from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn

import numpy as np

import cs285.infrastructure.pytorch_util as ptu


class DQNAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: int,
        use_double_q: bool = False,
        clip_grad_norm: Optional[float] = None,
    ):
        super().__init__()

        self.critic = make_critic(observation_shape, num_actions)
        self.target_critic = make_critic(observation_shape, num_actions)
        self.critic_optimizer = make_optimizer(self.critic.parameters())
        self.lr_scheduler = make_lr_schedule(self.critic_optimizer)

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.discount = discount
        self.target_update_period = target_update_period
        self.clip_grad_norm = clip_grad_norm
        self.use_double_q = use_double_q

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

    def get_action(self, observation: np.ndarray, epsilon: float = 0.02) -> int:
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]

        q_values = self.critic(observation).detach()
        argmx = torch.argmax(q_values, dim=1)
        if len(q_values[0]) > 1:
            q_values.fill_(epsilon / (len(q_values[0])-1))
        q_values[:,argmx] = 1 - epsilon

        action_index = torch.multinomial(q_values, 1).item()

        # TODO(student): get the action from the critic using an epsilon-greedy strategy
        action = action_index

        #return ptu.to_numpy(action).squeeze(0).item()
        return action

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> dict:
        """Update the DQN critic, and return stats for logging."""
        (batch_size,) = reward.shape

        # Compute target values
        with torch.no_grad():
            # TODO(student): compute target values
            next_qa_values = self.target_critic(next_obs)

            if self.use_double_q:
                raise NotImplementedError
            else:
                next_action = torch.argmax(next_qa_values, dim=1)
            # print(next_action.shape, next_qa_values.shape)
            next_q_values = next_qa_values[torch.arange(next_qa_values.size(0)),next_action]
            target_values = reward + (1-done.float()) * self.discount * next_q_values

        # TODO(student): train the critic with the target values
        qa_values = self.critic(obs)
        q_values = qa_values[torch.arange(qa_values.size(0)), action] # Compute from the data actions; see torch.gather
        loss = self.critic_loss(target_values, q_values)


        self.critic_optimizer.zero_grad()
        loss.backward()
        if self.clip_grad_norm:
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.clip_grad_norm)
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.critic_optimizer.step()

        self.lr_scheduler.step()

        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
            "grad_norm": grad_norm.item(),
        }

    def update_target_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def update(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        step: int,
    ) -> dict:
        """
        Update the DQN agent, including both the critic and target.
        """
        # TODO(student): update the critic, and the target if needed

        critic_stats = self.update_critic(obs, action, reward, next_obs, done)

        if step % self.target_update_period == 0:
            self.update_target_critic()


        return critic_stats
