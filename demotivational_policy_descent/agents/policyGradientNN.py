import logging
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import torch

from demotivational_policy_descent.agents.agent_interface import AgentInterface
from demotivational_policy_descent.utils.pg import discount_rewards, softmax_sample

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        # Create layers etc
        self.state_space = state_space
        self.action_space = action_space
        self.fc1 = torch.nn.Linear(state_space, 1024)
        #self.fc2 = torch.nn.Linear(1024, 512)
        #self.fc3 = torch.nn.Linear(512, 512)
        #self.fc4 = torch.nn.Linear(512, 256)
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

        #x = self.fc2(x)
        #x = F.relu(x)

        #x = self.fc3(x)
        #x = F.relu(x)

        #x = self.fc4(x)
        #x = F.relu(x)

        x = self.fc5(x)
        x = F.softmax(x, dim=-1)

        return x


class PolicyGradientNN(AgentInterface):
    def __init__(self, env, state_space, action_space, policy, player_id:int=1):
        super().__init__(env=env, player_id=player_id)

        self.train_device = "cpu" #if torch.cuda.is_available() else "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=5e-3)
        self.batch_size = 1
        self.gamma = 0.98

        self.reset()  # Call reset here to avoid code duplication!

    def reset(self):
        logging.debug("Resetting parameters...")
        #self.observations = []
        self.actions = []
        self.rewards = []
        logging.debug("Reset!")

    def episode_finished(self, episode_number):
        #all_actions = torch.stack(self.actions, dim=0).to(self.train_device).squeeze(-1)
        print(self.actions)
        all_actions = torch.stack(torch.Tensor(self.actions), dim=0).to(self.train_device).squeeze(-1)
        all_rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        self.reset()

        discounted_rewards = discount_rewards(all_rewards, self.gamma)
        discounted_rewards -= torch.mean(discounted_rewards)
        discounted_rewards /= torch.std(discounted_rewards)

        weighted_probs = all_actions * discounted_rewards
        loss = torch.sum(weighted_probs)
        loss.backward()

        if (episode_number + 1) % self.batch_size == 0:
            self.update_policy()

    def update_policy(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_action(self, observation, evaluation=False, frame: np.array=None) -> int:
        observation = observation.flatten()
        x = torch.from_numpy(observation).float().to(self.train_device)#float().to(self.train_device)
        prob = self.policy.forward(x)#.detach().numpy()
        chosen_action = softmax_sample(prob)
        prob = prob.detach().numpy()

        return chosen_action, np.log(prob[chosen_action])

    def store_outcome(self, log_action_prob, action_taken, reward):#observation, log_action_prob, action_taken, reward):
        # dist = torch.distributions.Categorical(action_output)
        action_taken = torch.Tensor([action_taken]).to(self.train_device)
        # log_action_prob = -dist.log_prob(action_taken)

        #self.observations.append(observation)
        self.actions.append(-log_action_prob)
        self.rewards.append(torch.Tensor([reward]))


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    dummy = PolicyGradientNN(env=None, player_id=1)
    dummy.test_attribute = 100
    name = "pg_test_model.mdl"
    dummy.save_model(name)
    dummy.test_attribute = 200
    dummy.load_model(name)
    assert dummy.test_attribute == 100
    dummy.reset()
    assert dummy.test_attribute == 5
    print("Dummy action", dummy.get_action(np.zeros(1)))
