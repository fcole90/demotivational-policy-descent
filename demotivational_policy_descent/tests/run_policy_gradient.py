import argparse
import logging

from demotivational_policy_descent.environment.pong import Pong
from demotivational_policy_descent.agents.simple_ai import PongAi
from demotivational_policy_descent.agents.policy_gradient import PolicyGradient
from demotivational_policy_descent.utils.utils import prod, load_logger

__FRAME_SIZE__ = (200, 210, 3)

class StateMode:
    standard = prod(__FRAME_SIZE__)
    average = prod(__FRAME_SIZE__[0:1])

class ActionMode:
    standard = 3
    reduced = 2

__ACTION_MODE__ = ActionMode.standard

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    args = parser.parse_args()

    load_logger("policy_gradient")

    env = Pong(headless=args.headless)
    episodes = 100000

    state_shape = StateMode.standard
    action_shape = ActionMode.standard

    player_id = 1
    player = PolicyGradient(env, state_shape, action_shape, player_id)

    opponent_id = 3 - player_id
    opponent = PongAi(env, opponent_id)

    env.set_names(player.get_name(), opponent.get_name())

    # Initialisation
    (ob1, ob2) = env.reset()
    prev_ob1 = ob1

    reward = 0
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

            if not args.headless:
                env.render()

        # When done..
        (ob1, ob2) = env.reset()
        # plot(ob1) # plot the reset observation
        if ((episode +1 ) % 20) == 0:
            logging.info("episode {} over - reward = {}".format(episode + 1, reward))
            reward = 0

        if ((episode + 1) % 10) == 0:
            player.optimise_policy(episode)

    # Needs to be called in the end to shut down pygame
    env.end()

if __name__ == "__main__":
    main()