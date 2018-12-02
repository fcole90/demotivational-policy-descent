import logging
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import torch

from demotivational_policy_descent.utils.utils import prod
from demotivational_policy_descent.agents.agent_interface import AgentInterface
from demotivational_policy_descent.agents.policy_gradient import PolicyGradient, PolicyNormal
from demotivational_policy_descent.utils.utils import discount_rewards, softmax_sample

class ActorCriticPolicy(PolicyNormal):
    def __init__(self, state_shape, action_shape, depth=50):
        super().__init__(state_shape, action_shape, depth)
        self.fc_value = torch.nn.Linear(50, action_shape)
        # Initialize neural network weights
        self.init_weights()

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        mean = self.fc_mean(x)
        std = self.fc_std(x)
        std = torch.sigmoid(std)
        value = self.fc_value(x)
        return mean, std, value


class ActorCritic(PolicyGradient):
    def __init__(self, env, state_shape: tuple, action_shape, player_id: int = 1, policy=None, cuda=False):

        # Automatic default policy
        self.__default_policy__ = ActorCritic
        if policy is None:
            policy = self.__default_policy__(state_shape, action_shape)

        super().__init__(env, state_shape, action_shape, player_id, policy, cuda)

        self.reset()  # Call reset here to avoid code duplication!

    def reset(self):
        super().reset()
        self.dones_list = list()
        self.observations_list = list()
        self.next_observations_list = list()

    def get_action_value(self, frame: np.array,
                   evaluation=False,
                   combine=False,
                   store_prev_mode=False) -> tuple:
        return super().get_action(frame, evaluation, combine, store_prev_mode, get_value=True)

    def fix_negative_strides(self, observation):
        fixed_observation = observation.copy()
        del observation
        return fixed_observation

    def store_outcome(self, log_action_prob, reward, observation, next_state, done):
        try:
            self.observations_list.append(torch.Tensor(observation))
        except ValueError:
            self.observations_list.append(torch.Tensor(self.fix_negative_strides(observation)))

        try:
            self.next_observations_list.append(torch.Tensor(next_state))
        except ValueError:
            self.next_observations_list.append(torch.Tensor(self.fix_negative_strides(next_state)))

        self.log_action_prob_list.append(-log_action_prob)
        self.tensor_rewards_list.append(torch.Tensor([reward]))
        self.dones_list.append([done])

    def optimise_policy(self, elapsed_episodes):
        # Stack gathered data for torch processing
        log_action_prob_stack = torch.stack(self.log_action_prob_list, dim=0).to(self.train_device).squeeze(-1)
        rewards_stack = torch.stack(self.tensor_rewards_list, dim=0).to(self.train_device).squeeze(-1)
        observations_stack = torch.stack([o for o in self.observations_list], dim=0).to(self.train_device).squeeze(-1)
        next_observations_stack = torch.stack([o for o in self.next_observations_list], dim=0).to(self.train_device).squeeze(-1)
        dones_stack = torch.stack([torch.Tensor(o) for o in self.dones_list], dim=0).to(self.train_device).squeeze(-1)

        # Reset storage variables for following learning
        self.reset()

        # Recompute state value estimates
        _, _, v_old = self.policy(observations_stack)
        _, _, v_next_state = self.policy(next_observations_stack)

        # Transform terminal states into zeroes
        v_next_state = v_next_state.squeeze(-1) * (1 - dones_stack)

        # Detach variables from torch
        v_next_state = v_next_state.detach()
        v_old = v_old.squeeze(-1)

        # Discount rewards with gamma parameter, center and normalise data
        discounted_rewards = discount_rewards(rewards_stack, self.gamma)
        discounted_rewards -= torch.mean(discounted_rewards)
        discounted_rewards /= torch.std(discounted_rewards)

        # estimate of state value and critic loss
        updated_state_values = rewards_stack + self.gamma * v_next_state

        # delta and normalised
        delta = updated_state_values - v_old

        # critic_loss = F.mse_loss(delta)
        critic_loss = torch.sum(delta ** 2)

        delta -= torch.mean(delta)
        delta /= torch.std(delta)
        delta = delta.detach()

        # actor update
        weighted_probs = log_action_prob_stack * delta
        actor_loss = torch.sum(weighted_probs)

        loss = actor_loss + 0.5 * critic_loss
        loss.backward()

        if (elapsed_episodes + 1) % self.batch_size == 0:
            self.update_policy()
