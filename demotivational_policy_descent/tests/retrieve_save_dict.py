import argparse
import logging
import sys
import socket
import os
import pickle
import torch

import numpy as np

from demotivational_policy_descent.environment.pong import Pong
from demotivational_policy_descent.agents.simple_ai import PongAi
from demotivational_policy_descent.agents.policy_gradient import PolicyGradient, StateMode, ActionMode, PolicyNormal
from demotivational_policy_descent.agents.policy_gradient_CNN import PolicyGradientCNN
from demotivational_policy_descent.agents.policy_gradient_DNN import PolicyGradientDNN
from demotivational_policy_descent.utils.utils import load_logger, alert_on_cuda, get_commit_hash, save_tmp_safe
from demotivational_policy_descent.utils import io

__FRAME_SIZE__ = (200, 210, 3)


def load_model(filename: str):
    """Loads a model from file.
    A method that takes a model file name (string) as input and loads the saved model into the agent.
    filename: str
        a path to the model to load
    """
    file_path = os.path.join(io.MODELS_PATH, filename) + ".mdl"

    if os.sep in filename:
        raise ValueError("A dir separator is contained in the filename."
                         " Just give your file a name, it will be loaded from {}".format(io.MODELS_PATH))

    logging.debug("Loading model...")

    with open(file_path, "rb") as model_file:
        model = pickle.load(model_file)
    logging.debug("Loaded!")
    return model




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Run with rendering")
    parser.add_argument("--normal", action="store_true", help="Run with a normal distribution")
    parser.add_argument("--reduced", action="store_true", help="Run with only up and down")
    parser.add_argument("--average", action="store_true", help="Run in averaged greyscale")
    parser.add_argument("--preprocess", action="store_true", help="Run preprocess image")
    parser.add_argument("--combine", action="store_true", help="Run on stacked images")
    parser.add_argument("--cnn", action="store_true", help="Use a CNN instead than simple NN")
    parser.add_argument("--dnn", action="store_true", help="Use a DNN instead than simple NN")
    parser.add_argument("--cuda", action="store_true", help="Run in cuda device")
    parser.add_argument("--load", action="store_true", help="Load a pre-trained model")
    args = parser.parse_args()

    # Default values
    filename = "policy_gradient"
    state_shape = StateMode.standard
    cnn_state_shape = list(__FRAME_SIZE__)
    action_shape = ActionMode.standard

    # Check incompatible combinations
    if args.cnn is True and args.dnn is True:
        raise ValueError("Only one can be selected among cnn and dnn.")

    if any([args.cnn, args.dnn]) is True and args.normal is True:
        raise ValueError("Selecting the distribution is not currently supported for CNN and DNN")

    if args.reduced is True:
        action_shape = ActionMode.reduced
        filename += "_red_actions"

    if args.average is True:
        state_shape = StateMode.average
        cnn_state_shape[2] = 1
        filename += "_average_grayscale"

    if args.preprocess is True:
        state_shape = StateMode.preprocessed
        cnn_state_shape = [100, 100, 1]
        filename += "_preprocessed_grayscale"

    if args.combine is True:
        state_shape *= 2
        cnn_state_shape[0] *= 2
        filename += "_combine"

    if args.cnn is True:
        filename += "_CNN"

    if args.dnn is True:
        filename += "_DNN"

    if args.normal is True:
        filename += "_normal"

    if args.load is True:
        load_filename = filename
        filename += "_loaded"

    load_logger(filename=filename)

    logging.info("*** New run ----------------------------------------------------------------------------------- ***")
    logging.info("Hostname: {}".format(socket.gethostname()))
    logging.info("Commit hash: {}".format(get_commit_hash()))
    logging.info("$" + " ".join(sys.argv))  # print script name and arguments


    if args.cuda is True:
        alert_on_cuda()

    if args.load is True:
        logging.info("Loading pre-trained model as both opponent and player.")

    env = Pong(headless=not args.render)
    episodes = 500000

    logging.info("Action shape: {}, State shape: {}".format(action_shape, state_shape))

    player_id = 1

    # Select the Agent method to use
    player = None
    if args.cnn is True:
        # Todo: use the tuple to set the shape of the first layer of convolution
        player = PolicyGradientCNN(env, cnn_state_shape, action_shape, player_id, cuda=args.cuda)
    elif args.dnn is True:
        player = PolicyGradientDNN(env, state_shape, action_shape, player_id, cuda=args.cuda)
    elif args.normal is True:
        player = PolicyGradient(env, state_shape, action_shape, player_id, cuda=args.cuda,
                                policy=PolicyNormal(state_shape, action_shape))
    else:
        player = PolicyGradient(env, state_shape, action_shape, player_id, cuda=args.cuda)

    opponent_id = 3 - player_id
    if args.load is False:
        opponent = PongAi(env, opponent_id)
    else:
        if args.cnn is True:
            # Todo: use the tuple to set the shape of the first layer of convolution
            opponent = PolicyGradientCNN(env, cnn_state_shape, action_shape, player_id)
        elif args.dnn is True:
            opponent = PolicyGradientDNN(env, state_shape, action_shape, player_id)
        elif args.normal is True:
            opponent = PolicyGradient(env, state_shape, action_shape, player_id,
                                    policy=PolicyNormal(state_shape, action_shape))
        else:
            opponent = PolicyGradient(env, state_shape, action_shape, player_id)

    model = load_model(filename)
    policy = model.policy
    torch.save(policy.state_dict(), os.path.join(io.MODELS_PATH, filename + ".pt"))



if __name__ == "__main__":
    main()