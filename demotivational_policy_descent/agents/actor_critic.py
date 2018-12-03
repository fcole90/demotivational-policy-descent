import logging
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import torch

from demotivational_policy_descent.utils.utils import prod
from demotivational_policy_descent.agents.agent_interface import AgentInterface
from demotivational_policy_descent.agents.policy_gradient import PolicyGradient, PolicyNormal, PolicyCategorical
from demotivational_policy_descent.utils.utils import discount_rewards, softmax_sample


class ActorCriticPolicyNormal(PolicyNormal):
    def __init__(self, state_shape, action_shape, depth=50):
        super().__init__(state_shape, action_shape, depth)
        self.fc_value = torch.nn.Linear(50, 1)
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


class ActorCriticPolicyCategorical(PolicyCategorical):
    def __init__(self, state_shape, action_shape, depth=50):
        super().__init__(state_shape, action_shape, depth_last=depth)
        self.fc_value = torch.nn.Linear(50, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        value = self.fc_value(x)
        x = self.fc3(x)
        return F.softmax(x, dim=-1), value


class ActorCritic(PolicyGradient):
    def __init__(self, env, state_shape, action_shape, player_id: int = 1, policy=None, cuda=False):

        # Automatic default policy
        self.__default_policy__ = ActorCriticPolicyCategorical
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
        ret_val = super().get_action(frame, evaluation, combine, store_prev_mode, get_value=True)
        try:
            self.observations_list.append(torch.Tensor(self.finalised_observation))
        except ValueError:
            self.observations_list.append(torch.Tensor(self.fix_negative_strides(self.finalised_observation)))
        return ret_val

    def fix_negative_strides(self, observation):
        fixed_observation = observation.copy()
        del observation
        return fixed_observation

    def store_outcome(self, log_action_prob, reward, done):
        self.log_action_prob_list.append(-log_action_prob)
        # if issubclass(self.current_policy_class, PolicyCategorical):
        #     self.tensor_rewards_list.append(torch.Tensor([reward] * self.action_shape))
        #     self.dones_list.append([done] * self.action_shape)
        # else:
        self.tensor_rewards_list.append(torch.Tensor([reward]))
        self.dones_list.append([done])

    def optimise_policy(self, elapsed_episodes):
        # Stack gathered data for torch processing
        log_action_prob_stack = torch.stack(self.log_action_prob_list, dim=0).to(self.train_device).squeeze(-1)
        rewards_stack = torch.stack(self.tensor_rewards_list, dim=0).to(self.train_device).squeeze(-1)
        observations_stack = torch.stack(self.observations_list[:-1], dim=0).to(self.train_device).squeeze(-1)
        next_observations_stack = torch.stack(self.observations_list[1:], dim=0).to(self.train_device).squeeze(-1)
        dones_stack = torch.stack([torch.Tensor(o) for o in self.dones_list], dim=0).to(self.train_device).squeeze(-1)

        # Reset storage variables for following learning
        self.reset()

        # Recompute state value estimates
        print("Observations stack shape:", observations_stack.shape)
        print("Current policy:", self.policy.__class__.__name__)

        # terminal_states_mask = (1 - dones_stack)

        if issubclass(self.current_policy_class, PolicyCategorical):
            _, current_observation_value = self.policy(observations_stack)
            _, next_observation_value = self.policy(next_observations_stack)
        else:
            _, _, current_observation_value = self.policy(observations_stack)
            _, _, next_observation_value = self.policy(next_observations_stack)

        # Transform terminal states into zeroes
        logging.debug("next_observation_value shape:", next_observation_value.shape)
        logging.debug("dones_stack shape:", dones_stack.shape)
        logging.debug("rewards_stack shape:", rewards_stack.shape)
        logging.debug("log_action_prob_stack:", log_action_prob_stack.shape)

        next_observation_value = (1 - dones_stack) * next_observation_value.squeeze(-1)

        # Detach variables from torch
        next_observation_value = next_observation_value.detach()
        current_observation_value = current_observation_value.squeeze(-1)

        # Discount rewards with gamma parameter, center and normalise data
        discounted_rewards = discount_rewards(rewards_stack, self.gamma)
        discounted_rewards -= torch.mean(discounted_rewards)
        discounted_rewards /= torch.std(discounted_rewards)

        # estimate of state value and critic loss
        updated_state_values = rewards_stack + self.gamma * next_observation_value

        # delta and normalised
        delta = updated_state_values - current_observation_value

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
