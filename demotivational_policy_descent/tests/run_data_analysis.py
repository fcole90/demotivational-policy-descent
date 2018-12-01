import argparse

import matplotlib.pyplot as plt
import numpy as np

from demotivational_policy_descent.environment.pong import Pong
from demotivational_policy_descent.agents.simple_ai import PongAi

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
args = parser.parse_args()

def reduce_size(observation):
    return observation[:, 10:-10]


def plot(observation, title=None, bn=False):
    if title is not None:
        plt.title(title)
    if bn is True:
        observation = np.sum(observation, axis=2, dtype=float) / 3
    plt.imshow(observation / 255)
    plt.show()

def main():
    env = Pong(headless=args.headless)
    episodes = 10

    player_id = 1
    opponent_id = 3 - player_id
    opponent = PongAi(env, opponent_id)
    player = PongAi(env, player_id)

    env.set_names(player.get_name(), opponent.get_name())
    ob1, ob2 = env.reset()

    plot(reduce_size(ob1), "State 0, PL1")
    plot(reduce_size(ob1), "State 0, PL1", bn=True)
    exit()
    plot(ob2, "State 0, PL2")

    for i in range(5):
        action1 = player.get_action()
        action2 = opponent.get_action()
        (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))

    plot(ob1, "State 5, PL1")
    plot(ob2, "State 5, PL2")
    exit()


    for i in range(0, episodes):
        done = False
        while not done:
            action1 = player.get_action()
            action2 = opponent.get_action()
            (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))


            if not args.headless:
                env.render()
            if done:
                observation = env.reset()
                # plot(ob1) # plot the reset observation
                print("episode {} over".format(i+1))

    # Needs to be called in the end to shut down pygame
    env.end()

if __name__ == "__main__":
    main()