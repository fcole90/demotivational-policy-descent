import logging
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import torch

from demotivational_policy_descent.utils.utils import prod
from demotivational_policy_descent.agents.agent_interface import AgentInterface
from demotivational_policy_descent.utils.utils import discount_rewards, softmax_sample

__FRAME_SIZE__ = (200, 210, 3)

class StateMode:
    standard = prod(__FRAME_SIZE__)
    average = prod(__FRAME_SIZE__[0:2])
    preprocessed = 100*100

class ActionMode:
    standard = 3
    reduced = 2

class Policy(torch.nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()

        if type(state_shape) is tuple:
            if len(state_shape) == 1:
                state_shape = state_shape[0]
            else:
                raise ValueError("Expected int or tuple of len=1 for state_shape, found {}".format(state_shape))

        if type(action_shape) is tuple:
            if len(action_shape) == 1:
                action_shape = action_shape[0]
            else:
                raise ValueError("Expected int or tuple of len=1 for action_shape, found {}".format(action_shape))


        # Create layers etc
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.fc1 = torch.nn.Linear(state_shape, 200)
        self.fc_mean = torch.nn.Linear(200, action_shape)
        self.fc_std = torch.nn.Linear(200, action_shape)

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
        std = self.fc_std(x)
        std = torch.sigmoid(std)
        return mean, std


class PolicyGradient(AgentInterface):
    def __init__(self, env, state_shape, action_shape, player_id:int=1, policy=None, cuda=False):
        """Agent implementing Policy Gradient.

        Parameters
        ----------
        env: object
            Environment of the experiment. Not used in this setting as we
            are assuming partial observability (where an observation is
            a frame of the game).
        state_shape: int
            shape of the state, expressed as a multiplication int
            e.g. (200 * 210 * 3)
        action_shape: int
            shape of the action, often equal to the number of dimension
            of the action space.
        player_id: int
            id of the player for labelling purposes
        policy: Policy (Optional)
            the policy of the agent, if a Policy is not
            given, then a default one is built.
        """
        super().__init__(env=env, player_id=player_id)

        # Automatic default policy
        self.__default_policy__ = Policy
        if policy is None:
            policy = self.__default_policy__(state_shape, action_shape)

        self.train_device = "cuda" if cuda is True else "cpu"
        self.state_shape = state_shape
        self.action_shape = action_shape

        # Policy of the agent
        self.policy = policy.to(self.train_device)

        # NN Optimiser
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

        # Batch size for the training
        self.batch_size = 1

        # Discount for previous rewards
        self.gamma = 0.98

        # Call reset here to avoid code duplication!
        self.reset()

    def reset(self):
        """Reset some attributes.
        """
        logging.debug("Resetting parameters...")
        self.observations = list()
        self.log_action_prob_list = list()
        self.tensor_rewards_list = list()
        self.log_action_prob_list = list()
        logging.debug("Reset!")

    def shape_check_and_adapt(self, observation_shape):
        # Conformance check for policy and frame to have the right shape
        actual_state_shape = observation_shape
        if type(actual_state_shape) is tuple and len(actual_state_shape) == 1:
            actual_state_shape = actual_state_shape[0]

        if actual_state_shape != self.state_shape:
            logging.warning("Expected frame size {} but found {}."
                            " Using default policy with correct size.".format(self.state_shape, actual_state_shape))
            logging.warning("If you want a different policy rerun the"
                            " program with 'state_shape={}'".format(actual_state_shape))
            self.state_shape = actual_state_shape
            self.policy = self.__default_policy__(self.state_shape, self.action_shape).to(self.train_device)

    @staticmethod
    def preprocess(frame: np.array, down_x=2, down_y=2):
        frame = PolicyGradient.average_black_white(frame)
        frame = frame[:, 5:-5]
        frame = frame[::down_y, ::down_x]  # downsample by factor of 2 and
        frame[frame < 5] = 0  # Make the background full black
        frame[frame > 0] = 255  # Make everything else full white
        return frame

    @staticmethod
    def average_black_white(frame: np.array):
        return np.sum(frame, axis=2, dtype=float) / 3

    def get_action(self, frame: np.array, evaluation=False) -> tuple:
        """Observe the environment and take an appropriate action.

        Parameters
        ----------
        frame: np.array
            A game frame as numpy array
        evaluation: bool (Optional, default: False)
            Instead of sampling from a distribution, always exploit
            to get the best value.

        Returns
        -------
        tuple (inst, float)
            An action to take and its associated log probabilities of success.
        """
        given_shape = frame.shape

        if self.state_shape == StateMode.average:
            # Average values transforming in greyscale
            frame = PolicyGradient.average_black_white(frame)
            if prod(frame.shape) != StateMode.average:
                raise ValueError("Expected shape {}, found {}, original was {}".format(StateMode.average,
                                                                                       prod(frame.shape),
                                                                                       prod(given_shape)))
        observation = frame.ravel() / 255

        self.shape_check_and_adapt(observation.shape)

        x = torch.from_numpy(observation).float().to(self.train_device)
        mean, s = self.policy.forward(x)

        if evaluation is True:
            action = np.argmax(mean)
        else:
            action = Normal(loc=mean, scale=s).sample()

        log_prob = Normal(loc=mean, scale=s).log_prob(action)
        chosen_action = softmax_sample(torch.exp(log_prob))

        # If using only UP and DOWN add one to skip "STAY"
        if self.action_shape == ActionMode.reduced:
            return chosen_action + 1, log_prob[chosen_action]

        return chosen_action, log_prob[chosen_action]

    def store_outcome(self, log_action_prob, reward):
        """Store the outcome of the last action.

        Parameters
        ----------
        log_action_prob: np.array
            log action probabilities of the action(s)
        reward: float
            reward for the last action taken

        """
        self.log_action_prob_list.append(log_action_prob)
        self.tensor_rewards_list.append(torch.Tensor([reward]))

    def optimise_policy(self, elapsed_episodes):
        """Use gathered data to optimise the policy.

        Parameters
        ----------
        elapsed_episodes: int
            number of elapsed episodes
        """

        # Stack gathered data for torch processing
        log_action_prob_stack = torch.stack(self.log_action_prob_list, dim=0).to(self.train_device).squeeze(-1)
        rewards_stack = torch.stack(self.tensor_rewards_list, dim=0).to(self.train_device).squeeze(-1)

        # Reset storage variables for following learning
        self.reset()

        # Discount rewards with gamma parameter, center and normalise data
        discounted_rewards = discount_rewards(rewards_stack, self.gamma)
        discounted_rewards -= torch.mean(discounted_rewards)
        discounted_rewards /= torch.std(discounted_rewards)

        # Weight the log_probabilities by the discounted rewards
        # to give more value to the more rewarding actions.
        # We take the negative of the probabilities to compute the value as a loss.
        weighted_probs = (-log_action_prob_stack) * discounted_rewards

        # Actual backpropagation, minimising loss
        loss = torch.sum(weighted_probs)
        loss.backward()

        if (elapsed_episodes + 1) % self.batch_size == 0:
            self.update_policy()

    def update_policy(self):
        """Update the policy towards gradient minimisation.
        """
        self.optimizer.step()
        self.optimizer.zero_grad()




if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    pg = PolicyGradient(env=None, action_shape=1, state_shape=2*3)
    name = "test_pg_model.mdl"
    pg.save_model(name)