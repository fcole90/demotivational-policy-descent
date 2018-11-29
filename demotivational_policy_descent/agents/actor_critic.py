import logging
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import torch

from demotivational_policy_descent.agents.agent_interface import AgentInterface
from demotivational_policy_descent.utils.utils import discount_rewards, softmax_sample

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        # Create layers etc
        self.state_space = state_space
        self.action_space = action_space
        self.fc1 = torch.nn.Linear(state_space, 50)
        self.fc_mean = torch.nn.Linear(50, action_space)
        self.fc_s = torch.nn.Linear(50, action_space)
        self.fc_v = torch.nn.Linear(50, action_space)

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
        v = self.fc_v(x)
        return mean, s, v


class ActorCritic(AgentInterface):
    def __init__(self, env, state_space, action_space, policy, player_id:int=1):
        super().__init__(env=env, player_id=player_id)

        self.train_device = "cpu" #if torch.cuda.is_available() else "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.RMSprop(policy.parameters(), lr=5e-3)
        self.batch_size = 1
        self.gamma = 0.98

        self.reset()  # Call reset here to avoid code duplication!

    def reset(self):
        logging.debug("Resetting parameters...")
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_states = []
        logging.debug("Reset!")

    def episode_finished(self, episode_number):
        # Stack data
        all_actions = torch.stack(self.actions, dim=0).to(self.train_device).squeeze(-1)
        all_states = torch.stack([o for o in self.observations], dim=0).to(self.train_device).squeeze(-1)
        all_next_states = torch.stack([o for o in self.next_states], dim=0).to(self.train_device).squeeze(-1)
        all_rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        all_done = torch.stack([torch.Tensor(o) for o in self.dones], dim=0).to(self.train_device).squeeze(-1)


        self.reset()

        # compute state value estimates
        _, _, v_old = self.policy(all_states)
        _, _, v_next_state = self.policy(all_next_states)

        # zero out for terminal states
        v_next_state = v_next_state.squeeze(-1) * (1 - all_done)

        # detach
        v_next_state = v_next_state.detach()
        v_old = v_old.squeeze(-1)

        # normalise rewards
        discounted_rewards = discount_rewards(all_rewards, self.gamma)
        discounted_rewards -= torch.mean(discounted_rewards)
        discounted_rewards /= torch.std(discounted_rewards)

        # estimate of state value and critic loss
        updated_state_values = all_rewards + self.gamma * v_next_state

        # delta and normalised
        delta = updated_state_values - v_old

        # critic_loss = F.mse_loss(delta)
        critic_loss = torch.sum(delta ** 2)

        delta -= torch.mean(delta)
        delta /= torch.std(delta)
        delta = delta.detach()

        # actor update
        weighted_probs = all_actions * delta
        actor_loss = torch.sum(weighted_probs)

        loss = actor_loss + 0.5 * critic_loss
        loss.backward()

        if (episode_number + 1) % self.batch_size == 0:
            self.update_policy()

    def update_policy(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_action(self, observation, evaluation=False, frame: np.array=None) -> int:
        observation = observation.flatten()
        x = torch.from_numpy(observation).float().to(self.train_device)#float().to(self.train_device)
        mean, s, v = self.policy.forward(x)
        if evaluation:
            action = np.argmax(mean)
        else:
            action = Normal(loc=mean, scale=s).sample()

        log_prob = Normal(loc=mean, scale=s).log_prob(action)
        chosen_action = softmax_sample(torch.exp(log_prob))
        return chosen_action, log_prob[chosen_action], v

    def store_outcome(self, observation, next_state, action_output, reward, done):
        try:
            self.observations.append(torch.Tensor(observation))
        except ValueError:
            self.observations.append(torch.Tensor(self.fix_negative_strides(observation)))

        try:
            self.next_states.append(torch.Tensor(next_state))
        except ValueError:
            self.next_states.append(torch.Tensor(self.fix_negative_strides(next_state)))

        self.actions.append(-action_output)
        self.rewards.append(torch.Tensor([reward]))
        self.dones.append([done])


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    dummy = ActorCritic(env=None, player_id=1)
    dummy.test_attribute = 100
    name = "actor_critic.mdl"
    dummy.save_model(name)
    dummy.test_attribute = 200
    dummy.load_model(name)
    assert dummy.test_attribute == 100
    dummy.reset()
    assert dummy.test_attribute == 5
    print("Dummy action", dummy.get_action(np.zeros(1)))
