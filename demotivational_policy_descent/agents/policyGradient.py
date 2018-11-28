import logging
import torch.nn.functional as F
import numpy as np
import torch

from demotivational_policy_descent.agents.agent_interface import AgentInterface

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        # Create layers etc
        self.state_space = state_space
        self.action_space = action_space
        self.fc1 = torch.nn.Linear(state_space, 20)
        self.fc_mean = torch.nn.Linear(20, action_space)
        self.fc_s = torch.nn.Linear(20, action_space)

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
        mean = self.fc_mean(x)
        s = self.fc_s(x)
        s = torch.sigmoid(s)
        return mean, s


class PolicyGradient(AgentInterface):
    def __init__(self, env, player_id:int=1):
        super().__init__(env=env, player_id=player_id)

        self.reset()  # Call reset here to avoid code duplication!

    def reset(self):
        logging.debug("Resetting parameters...")
        self.observations = []
        self.actions = []
        self.rewards = []
        logging.debug("Reset!")

    def get_action(self, frame: np.array=None) -> int:
        logging.debug("Returning a random action sampled from the frame..")
        return np.random.choice([0, 1, 2])


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    dummy = PolicyGradient(env=None, player_id=1)
    dummy.test_attribute = 100
    name = "pg_test_model.mdl"
    dummy.save_model(name)
    dummy.test_attribute = 200
    dummy.load_model(name)
    assert dummy.test_attribute == 100
    dummy.reset()
    assert dummy.test_attribute == 5
    print("Dummy action", dummy.get_action(np.zeros(1)))
