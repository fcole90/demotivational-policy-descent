import argparse
import logging
import sys
import socket


import numpy as np

from demotivational_policy_descent.environment.pong import Pong
from demotivational_policy_descent.agents.simple_ai import PongAi
from demotivational_policy_descent.agents.policy_gradient import PolicyGradient, StateMode, ActionMode, PolicyNormal
from demotivational_policy_descent.agents.policy_gradient_CNN import PolicyGradientCNN
from demotivational_policy_descent.agents.policy_gradient_DNN import PolicyGradientDNN
from demotivational_policy_descent.utils.utils import load_logger, alert_on_cuda, get_commit_hash, save_tmp_safe

__FRAME_SIZE__ = (200, 210, 3)

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



    load_logger(filename=filename)

    logging.info("*** New run ----------------------------------------------------------------------------------- ***")
    logging.info("Hostname: {}".format(socket.gethostname()))
    logging.info("Commit hash: {}".format(get_commit_hash()))
    logging.info("$" + " ".join(sys.argv))  # print script name and arguments


    if args.cuda is True:
        alert_on_cuda()

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
    opponent = PongAi(env, opponent_id)

    env.set_names(player.get_name(), opponent.get_name())

    # Initialisation
    (ob1, ob2) = env.reset()
    player.set_prev_observation(ob1, combine=args.combine)

    reward = 0
    logging.info("Beginning training..")
    for episode in range(episodes):

        # Reset accumulator variables
        rewards_sum = 0
        timesteps = 0

        # Run until done
        done = False
        while done is False:
            action1, log_prob = player.get_action(ob1, combine=args.combine)
            action2 = opponent.get_action()

            (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))

            player.store_outcome(log_prob, rew1)

            rewards_sum += rew1
            reward += rew1
            timesteps += 1

            if args.render is True:
                env.render()

        # When done..
        (ob1, ob2) = env.reset()
        # plot(ob1) # plot the reset observation
        if ((episode +1 ) % 20) == 0:
            logging.info("{}: [{:8}/{}] reward={}".format(filename, episode + 1, episodes, reward))
            reward = 0

        if ((episode + 1) % 5) == 0:
            player.optimise_policy(episode)

        if ((episode + 1) > 10000) and ((episode + 1) % 5000) == 0:
            save_tmp_safe(player, filename)

    # Needs to be called in the end to shut down pygame
    env.end()
    player.save_model(filename)

if __name__ == "__main__":
    main()