import logging

import numpy as np

from demotivational_policy_descent.agents.agent_interface import AgentInterface


class SimpleRL(AgentInterface):
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
    dummy = SimpleRL(env=None, player_id=1)
    dummy.test_attribute = 100
    name = "dummy_test_model.mdl"
    dummy.save_model(name)
    dummy.test_attribute = 200
    dummy.load_model(name)
    assert dummy.test_attribute == 100
    dummy.reset()
    assert dummy.test_attribute == 5
    print("Dummy action", dummy.get_action(np.zeros(1)))
