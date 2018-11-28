import argparse

import matplotlib.pyplot as plt

from demotivational_policy_descent.envs.pong import Pong
from demotivational_policy_descent.agents.simple_ai import PongAi
from demotivational_policy_descent.agents.policyGradient import PolicyGradient, Policy

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
args = parser.parse_args()


def plot(observation):
    plt.imshow(observation / 255)
    plt.show()

def main():
    env = Pong(headless=args.headless)
    episodes = 10

    player_id = 1
    opponent_id = 3 - player_id
    opponent = PongAi(env, opponent_id)

    state_space = 3
    action_space = 200 * 210 * 3

    policy = Policy(state_space, action_space)
    player = PolicyGradient(env, state_space, action_space, policy, player_id)

    env.set_names(player.get_name(), opponent.get_name())
    (ob1, ob2) = env.reset()

    for i in range(0, episodes):
        done = False
        while not done:
            action1 = player.get_action(ob1)
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