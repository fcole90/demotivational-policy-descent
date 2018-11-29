import argparse
import logging

from demotivational_policy_descent.environment.pong import Pong
from demotivational_policy_descent.agents.simple_ai import PongAi
from demotivational_policy_descent.agents.policy_gradient import PolicyGradient, StateMode, ActionMode
from demotivational_policy_descent.utils.utils import load_logger, alert_on_cuda

__FRAME_SIZE__ = (200, 210, 3)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Run with rendering")
    parser.add_argument("--reduced", action="store_true", help="Run with only up and down")
    parser.add_argument("--average", action="store_true", help="Run in averaged greyscale")
    parser.add_argument("--cuda", action="store_true", help="Run in cuda device")
    args = parser.parse_args()

    # Default values
    filename = "policy_gradient"
    state_shape = StateMode.standard
    action_shape = ActionMode.standard

    print("--reduced:", args.reduced)
    print("--average:", args.average)
    print("--render:", args.render)
    print("--cuda:", args.cuda)

    if args.reduced is True:
        action_shape = ActionMode.reduced
        filename += "_red_actions"

    if args.average is True:
        state_shape = StateMode.average
        filename += "_average_grayscale"

    load_logger(filename=filename)

    logging.info("*** New run ----------------------------------------------------------------------------------- ***")

    if args.cuda is True:
        alert_on_cuda()


    env = Pong(headless=not args.render)
    episodes = 100000


    logging.info("Action shape: {}, State shape: {}".format(action_shape, state_shape))

    player_id = 1
    player = PolicyGradient(env, state_shape, action_shape, player_id, cuda=args.cuda)

    opponent_id = 3 - player_id
    opponent = PongAi(env, opponent_id)

    env.set_names(player.get_name(), opponent.get_name())

    # Initialisation
    (ob1, ob2) = env.reset()
    prev_ob1 = ob1

    reward = 0
    logging.info("Beginning training..")
    for episode in range(episodes):

        # Reset accumulator variables
        rewards_sum = 0
        timesteps = 0

        # Run until done
        done = False
        while done is False:
            action1, log_prob = player.get_action(ob1 - prev_ob1)
            prev_ob1 = ob1
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

        if ((episode + 1) % 10) == 0:
            player.optimise_policy(episode)

    # Needs to be called in the end to shut down pygame
    env.end()

if __name__ == "__main__":
    main()