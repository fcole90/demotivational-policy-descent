import logging
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np
import torch
import torchvision

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

class PolicyNormal(torch.nn.Module):
    def __init__(self, state_shape, action_shape, depth=200):
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
        self.fc1 = torch.nn.Linear(state_shape, depth)
        self.fc_mean = torch.nn.Linear(depth, action_shape)
        self.fc_std = torch.nn.Linear(depth, action_shape)

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


class PolicyCategorical(torch.nn.Module):
    def __init__(self, state_shape, action_shape, depth_last=128):
        super().__init__()

        if type(state_shape) in [tuple, list]:
            if len(state_shape) == 1:
                state_shape = state_shape[0]
            else:
                raise ValueError("Expected int or tuple of len=1 for state_shape, found {}".format(state_shape))

        if type(action_shape) in [tuple, list]:
            if len(action_shape) == 1:
                action_shape = action_shape[0]
            else:
                raise ValueError("Expected int or tuple of len=1 for action_shape, found {}".format(action_shape))


        # Create layers etc
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.fc1 = torch.nn.Linear(state_shape, 256)
        self.fc2 = torch.nn.Linear(256, depth_last)
        self.fc3 = torch.nn.Linear(depth_last, action_shape)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return F.softmax(x, dim=-1)


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
        self.__default_policy__ = PolicyCategorical
        if policy is None:
            policy = self.__default_policy__(state_shape, action_shape)

        self.train_device = "cuda" if cuda is True else "cpu"
        self.state_shape = state_shape
        self.action_shape = action_shape

        # Policy of the agent
        self.current_policy_class = policy.__class__
        self.policy = policy.to(self.train_device)

        # NN Optimiser
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=0.0001)

        # Batch size for the training
        self.batch_size = 50

        # Discount for previous rewards
        self.gamma = 0.98

        # Call reset here to avoid code duplication!
        self.reset()

    def debug_attr(self):
        attribute_list = vars(self).keys()
        for attribute in attribute_list:
            print(attribute, ": ", type(getattr(self, attribute)), sep="")

    def reset(self):
        """Reset some attributes.
        """
        logging.debug("Resetting parameters...")
        self.log_action_prob_list = list()
        self.tensor_rewards_list = list()
        self.log_action_prob_list = list()
        logging.debug("Reset!")

    def shape_check_and_adapt(self, observation_shape):
        # Conformance check for policy and frame to have the right shape
        actual_state_shape = observation_shape
        if type(actual_state_shape) in [tuple, list] and len(actual_state_shape) == 1:
            actual_state_shape = actual_state_shape[0]

        if actual_state_shape != self.state_shape:
            logging.warning("Expected frame size {} but found {}."
                            " Using default policy ({}) with correct size.".format(self.state_shape,
                                                                                   actual_state_shape,
                                                                                   self.__default_policy__.__name__))
            logging.warning("If you want a different policy rerun the"
                            " program with 'state_shape={}'".format(actual_state_shape))
            self.state_shape = actual_state_shape
            self.policy = self.__default_policy__(self.state_shape, self.action_shape).to(self.train_device)

    @staticmethod
    def preprocess(frame: np.array, down_x=2, down_y=2):
        frame = PolicyGradient.average_black_white(frame)
        frame = frame[:, 5:-5]  # Make it as large as high
        frame = frame[::down_y, ::down_x]  # Downsample by the specified amount
        frame[frame < 5] = 0  # Make the background full black
        frame[frame > 0] = 255  # Make everything else full white
        return frame

    @staticmethod
    def average_black_white(frame: np.array):
        return np.sum(frame, axis=2, dtype=float) / 3

    def set_prev_observation(self, frame: np.array, combine=False):
        self.get_action(frame, combine=combine, store_prev_mode=True)

    def get_action(self, frame: np.array,
                   evaluation=False,
                   combine=False,
                   store_prev_mode=False,
                   get_value=False,
                   cnn_mode=False) -> tuple:
        """Observe the environment and take an appropriate action.

        Parameters
        ----------
        frame: np.array
            A game frame as numpy array
        evaluation: bool (Optional, default: False)
            Instead of sampling from a distribution, always exploit
            to get the best value.
        combine: bool (Optional, default: False)
            If true combine the frames by stacking them on axis 1,
            otherwise subtract the previous one from the current one
        store_prev_mode: bool (Optional, default: False)
            Doesn't provide any action, but only runs the part pertaining
            to frame preprocessing.
        get_value: bool (Optional, default: False)
            Used in an actor critic context, allows to also get a
            value for the current frame.

        Returns
        -------
        tuple (inst, float)
            An action to take and its associated log probabilities of success.
        """
        initial_frame_shape = frame.shape
        combine_mul = 2 if combine is True else 1  # multiplicator to make the lathes count when checking shapes
        observation = frame

        if self.state_shape == StateMode.average * combine_mul:
            # Average values transforming in greyscale
            observation = PolicyGradient.average_black_white(frame)
            if prod(observation.shape) != StateMode.average:
                logging.warning("[Avg] Expected shape {}, found {}, original was {}".format(StateMode.average,
                                                                                       prod(observation.shape),
                                                                                       prod(initial_frame_shape)))
        elif self.state_shape == StateMode.preprocessed * combine_mul:
            # Preprocess the frame in greyscale, and downsample it.
            observation = PolicyGradient.preprocess(frame)
            if prod(observation.shape) != StateMode.preprocessed:
                logging.warning("[Prp] Expected shape {}, found {}, original was {}".format(StateMode.average,
                                                                                  prod(observation.shape),
                                                                                  prod(initial_frame_shape)))

        # Always store the previous observation, already processed
        if store_prev_mode is True:
            # If only storing the prev_observation (iteration 0), then exit
            self.prev_observation = observation
            return (None, None)
        else:
            try:
                prev_observation = self.prev_observation
                self.prev_observation = observation
            except AttributeError:
                raise ValueError("You called 'get_action' but you did not 'set_prev_observation'.")


        # How do we combine the frames?
        if combine is True:
            observation = np.concatenate((observation, prev_observation), axis=1)
        else:
            observation = observation - prev_observation

        if cnn_mode is not True:
            # Make the observation flat
            observation = observation.ravel() / 255.0
        else:
            transform = torchvision.transforms.ToTensor()


        # Can be used by children implementations
        self.finalised_observation = observation

        # Sanity check to have the right NN
        self.shape_check_and_adapt(observation.shape)

        if cnn_mode is True:
            x = transform(observation).unsqueeze(0)
        else:
            x = torch.from_numpy(observation).float().to(self.train_device)

        if issubclass(self.current_policy_class, PolicyCategorical):

            # Categorical Version
            if get_value is True:
                prob, v = self.policy.forward(x)
            else:
                prob = self.policy.forward(x)

            if evaluation:
                action = torch.argmax(prob).item()
            else:
                action = softmax_sample(prob)

            chosen_action = action
            chosen_log_prob = torch.log(prob[action])

        else:
            # Gaussian version
            if get_value is True:
                mean, s, v = self.policy.forward(x)
            else:
                mean, s = self.policy.forward(x)

            if evaluation is True:
                action = np.argmax(mean)
            else:
                action = Normal(loc=mean, scale=s).sample()

            log_prob = Normal(loc=mean, scale=s).log_prob(action)

            chosen_action = softmax_sample(log_prob)
            chosen_log_prob = log_prob[chosen_action]

        # --- Valid for all versions ---

        # If using only UP and DOWN add one to skip "STAY"
        if self.action_shape == ActionMode.reduced:
            chosen_action += 1


        if get_value is True:
            return chosen_action, chosen_log_prob, v
        else:
            return chosen_action, chosen_log_prob

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