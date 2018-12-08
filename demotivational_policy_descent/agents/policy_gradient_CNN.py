import logging
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import torch
import torchvision.transforms as transforms

from demotivational_policy_descent.agents.agent_interface import AgentInterface
from demotivational_policy_descent.agents.policy_gradient import PolicyGradient, PolicyNormal, PolicyCategorical
from demotivational_policy_descent.utils.utils import discount_rewards, softmax_sample, prod



class PolicyCNNNormal(PolicyNormal):

    def output_size(in_size, kernel_size, stride, padding):
        output = int((in_size - kernel_size + 2 * (padding)) / stride) + 1
        return (output)

    def __init__(self, state_space, action_space):
        super().__init__()
        # Create layers etc
        self.state_space = state_space
        self.action_space = action_space

        self.conv1 = torch.nn.Conv2d(3, 20, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        if type(state_space) in [tuple, list]:
            state_space_p = prod(state_space)

        self.fc1 = torch.nn.Linear(20 * state_space[0] * state_space[1], 50)
        self.fc_mean = torch.nn.Linear(50, action_space)
        self.fc_s = torch.nn.Linear(50, action_space)

        # Initialize neural network weights
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.fc1(x)
        x = F.relu(x)
        mean = self.fc_mean(x)
        s = self.fc_s(x)
        s = torch.sigmoid(s)
        return mean, s


class PolicyCNNCategorical(PolicyCategorical):
    def __init__(self, state_shape, action_shape, depth_last=128):

        if type(state_shape) in [tuple, list]:
            state_shape_p = prod(state_shape)

        super().__init__(state_shape_p, action_shape)

        if type(action_shape) is tuple:
            if len(action_shape) == 1:
                action_shape = action_shape[0]
            else:
                raise ValueError("Expected int or tuple of len=1 for action_shape, found {}".format(action_shape))


        # Create layers etc
        self.state_shape = state_shape
        self.action_shape = action_shape

        self.conv1 = torch.nn.Conv2d(in_channels=state_shape[2], out_channels=20, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = torch.nn.Linear(20 * state_shape[0] * state_shape[1], 256)
        self.fc2 = torch.nn.Linear(256, depth_last)
        self.fc3 = torch.nn.Linear(depth_last, action_shape)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return F.softmax(x, dim=-1)


class PolicyGradientCNN(PolicyGradient):
    def __init__(self, env, state_shape: tuple, action_shape, player_id: int = 1, policy=None, cuda=False):

        self.current_policy_class = PolicyCNNCategorical.__class__

        if type(state_shape) is list:
            state_shape = tuple(state_shape)

        # Automatic default policy
        if policy is None:
            policy = PolicyCNNCategorical(state_shape, action_shape)

        super().__init__(env, state_shape, action_shape, player_id, policy, cuda)


        self.reset()  # Call reset here to avoid code duplication!

    def get_action(self,
                   frame: np.array,
                   evaluation=False,
                   combine=False,
                   store_prev_mode=False,
                   get_value=False,
                   cnn_mode=False) -> tuple:

        return super().get_action(frame=frame,
                                  evaluation=evaluation,
                                  combine=combine,
                                  store_prev_mode=store_prev_mode,
                                  get_value=get_value,
                                  cnn_mode=True)

