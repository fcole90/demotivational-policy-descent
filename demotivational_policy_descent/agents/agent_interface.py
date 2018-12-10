import abc
import logging
import os
import pickle

import numpy as np
import torch

from demotivational_policy_descent.environment.pong import Pong
from demotivational_policy_descent.utils import io


class AgentInterface(abc.ABC):
    def __init__(self, env, player_id=1):
        if env is None:
            logging.debug("Running with empty env for debugging purposes")
        elif type(env) is not Pong:
            raise TypeError("Expected type(env) == 'Pong', found '{}' instead.".format(type(env)))
        self.player_id = player_id
        self.agent_name = self.__class__.__name__

    def load_model(self, filename: str):
        """Loads a model from file.

        A method that takes a model file name (string) as input and loads the saved model into the agent.

        filename: str
            a path to the model to load
        """
        file_path = os.path.join(io.MODELS_PATH, filename) + ".mdl"

        attribute_list = vars(self).keys()

        if os.sep in filename:
            raise ValueError("A dir separator is contained in the filename."
                             " Just give your file a name, it will be loaded from {}".format(io.MODELS_PATH))

        logging.debug("Loading model...")

        with open(file_path, "rb") as model_file:
            device = torch.device('cpu')
            model = torch.load(model_file, map_location=device)
        logging.debug("Loaded!")

        for attribute in attribute_list:
            if hasattr(model, attribute):
                val = getattr(model, attribute)
                setattr(self, attribute, val)
            else:
                raise AttributeError("You tried to load {}.{} from you model, but the model in your file"
                                     " does not have such attribute.".format(model.get_name(), attribute))


    def save_model(self, filename: str):
        """Saves the model to file.

        Returns
        -------
        str:
            final filename of the file
        """
        if os.sep in filename:
            raise ValueError("A dir separator is contained in the filename."
                             " Just give your file a name, it will be loaded from {}".format(io.MODELS_PATH))

        ext = ".mdl"
        file_path = os.path.join(io.MODELS_PATH, filename)

        if os.path.exists(file_path + ext):
            i = 1
            file_path_edit = file_path + "_" + str(i) + ext
            while os.path.exists(file_path_edit):
                i += 1
                file_path_edit = file_path + "_" + str(i) + ext
            file_path = file_path_edit
        else:
            file_path += ext

        logging.debug("Saving model...")

        with open(file_path, "wb") as model_file:
            pickle.dump(self, model_file)
            if hasattr(self, "policy"):
                try:
                    torch.save(self.policy.state_dict(), file_path + ".pt")
                except Exception as e:
                    logging.error(e)



        logging.info("Model saved as {}".format(file_path))

        # Rebuild filename if edited
        filename = "".join(file_path.split(os.sep)[-1].split(".")[:-1])
        return filename

    @abc.abstractmethod
    def reset(self):
        """Resets the agent.

        Useful to reset the agent after an episode finishes.

        """
        pass

    @abc.abstractmethod
    def get_action(self, frame: np.array) -> tuple:
        pass

    def get_name(self) -> str:
        return self.agent_name
