import abc
import logging
import os
import pickle

import numpy as np

from demotivational_policy_descent.envs.pong import Pong
from demotivational_policy_descent.utils import io


class AgentInterface(abc.ABC):
    def __init__(self, env, player_id=1):
        if env is None:
            logging.debug("Running with empty env for debugging purposes")
        elif type(env) is not Pong:
            raise TypeError("Expected type(env) == 'Pong', found '{}' instead.".format(type(env)))
        self.env = env
        self.player_id = player_id

    def load_model(self, filename: str):
        """Loads a model from file.

        A method that takes a model file name (string) as input and loads the saved model into the agent.

        filename: str
            a path to the model to load
        """
        file_path = os.path.join(io.MODELS_PATH, filename)

        attribute_list = vars(self).keys()

        if os.sep in filename:
            raise ValueError("A dir separator is contained in the filename."
                             " Just give your file a name, it will be loaded from {}".format(io.MODELS_PATH))

        logging.debug("Loading model...")

        with open(file_path, "rb") as model_file:
            model = pickle.load(model_file)
        logging.debug("Loaded!")

        for attribute in attribute_list:
            if hasattr(model, attribute):
                val = getattr(model, attribute)
                setattr(self, attribute, val)
            else:
                raise AttributeError("You tried to load {}.{} from you model, but the model in your file"
                                     " does not have such attribute.".format(model.get_name(), attribute))

    def save_model(self, filename: str):
        """Saves the model to file."""
        if os.sep in filename:
            raise ValueError("A dir separator is contained in the filename."
                             " Just give your file a name, it will be loaded from {}".format(io.MODELS_PATH))

        file_path = os.path.join(io.MODELS_PATH, filename)

        logging.debug("Saving model...")

        with open(file_path, "wb") as model_file:
            pickle.dump(self, model_file)

        logging.debug("Saved!")

    @abc.abstractmethod
    def reset(self):
        """Resets the agent.

        Useful to reset the agent after an episode finishes.

        """
        pass

    @abc.abstractmethod
    def get_action(self, frame: np.array=None) -> int:
        pass

    def get_name(self) -> str:
        return self.__class__.__name__
