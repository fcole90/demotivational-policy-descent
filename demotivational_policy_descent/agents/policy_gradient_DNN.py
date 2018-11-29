import logging
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import torch

from demotivational_policy_descent.agents.agent_interface import AgentInterface
from demotivational_policy_descent.agents.policy_gradient import PolicyGradient
from demotivational_policy_descent.utils.utils import discount_rewards, softmax_sample

class PolicyDNN(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        # Create layers etc
        self.state_space = state_space
        self.action_space = action_space
        self.fc1 = torch.nn.Linear(state_space, 1024)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.fc3 = torch.nn.Linear(512, 512)
        self.fc4 = torch.nn.Linear(512, 256)
        self.fc5 = torch.nn.Linear(1024, action_space)

        # Initialize neural network weights
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)
        x = F.relu(x)

        x = self.fc5(x)

        x = F.softmax(x, dim=-1)

        return x


class PolicyGradientDNN(PolicyGradient):
    def __init__(self, env, state_shape: tuple, action_shape, player_id: int = 1, policy=None, cuda=False):

        # Automatic default policy
        if policy is None:
            policy = PolicyDNN(state_shape, action_shape)

        super().__init__(env, state_shape, action_shape, player_id, policy, cuda)

        self.reset()  # Call reset here to avoid code duplication!
