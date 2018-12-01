import logging
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import torch

from demotivational_policy_descent.agents.agent_interface import AgentInterface
from demotivational_policy_descent.agents.policy_gradient import PolicyGradient
from demotivational_policy_descent.utils.utils import discount_rewards, softmax_sample



class PolicyCNN(torch.nn.Module):

    def outputSize(in_size, kernel_size, stride, padding):
        output = int((in_size - kernel_size + 2 * (padding)) / stride) + 1
        return (output)

    def __init__(self, state_space, action_space):
        super().__init__()
        # Create layers etc
        self.state_space = state_space
        self.action_space = action_space

        self.conv1 = torch.nn.Conv2d(3, 20, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)


        self.fc1 = torch.nn.Linear(200, 50)
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


class PolicyGradientCNN(PolicyGradient):
    def __init__(self, env, state_shape: tuple, action_shape, player_id: int = 1, policy=None, cuda=False):

        # Automatic default policy
        if policy is None:
            policy = PolicyCNN(state_shape, action_shape)

        super().__init__(env, state_shape, action_shape, player_id, policy, cuda)

        self.reset()  # Call reset here to avoid code duplication!


    def get_action(self, observation, evaluation=False, frame: np.array=None) -> tuple:
        # TODO: give the full fram to the convolutional

        x = torch.from_numpy(observation).float().to(self.train_device)
        mean, s = self.policy.forward(x)
        if evaluation:
            action = np.argmax(mean)
        else:
            action = Normal(loc=mean, scale=s).sample()

        log_prob = Normal(loc=mean, scale=s).log_prob(action)
        chosen_action = softmax_sample(torch.exp(log_prob))
        return chosen_action, log_prob[chosen_action]

