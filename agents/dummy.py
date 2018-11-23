import logging

import numpy as np

from agents.agent_interface import AgentInterface


class DummyAgent(AgentInterface):
    def __init__(self):
        super().__init__()
        self.reset()  # Call reset here to avoid code duplication!

    def reset(self):
        logging.debug("Resetting parameters...")
        self.test = 1
        self.attributes = ["test"]  # List all the attributes of the class
        logging.debug("Reset!")

    def get_action(self, frame: np.array) -> int:
        logging.debug("Returning a random action sampled from the frame..")
        return np.random.choice(frame)


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logging.info("Testing \"{}\"".format(__file__))
    dummy = DummyAgent()
    dummy.test = 6
    name = "dummy_test_model.mdl"
    dummy.save_model(name)
    dummy.test = 5
    dummy.load_model(name)
    assert dummy.test == 6
    dummy.reset()
    assert dummy.test == 1
    print("Dummy action", dummy.get_action(np.array([1, 2, 3])))
