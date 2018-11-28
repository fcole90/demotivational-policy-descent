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
    episodes = 100000

    player_id = 1
    opponent_id = 3 - player_id
    opponent = PongAi(env, opponent_id)

    state_space = 200 * 210 * 3
    action_space = 3

    policy = Policy(state_space, action_space)
    player = PolicyGradient(env, state_space, action_space, policy, player_id)

    env.set_names(player.get_name(), opponent.get_name())
    (ob1, ob2) = env.reset()

    reward_n = 0
    for episode_no in range(episodes):
        reward_sum, timesteps = 0, 0
        done = False
        while not done:
            action1, log_prob = player.get_action(ob1)
            prev_ob1 = ob1
            action2 = opponent.get_action()

            (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))

            player.store_outcome(prev_ob1, log_prob, action1, rew1)

            reward_sum += rew1
            reward_n += rew1
            timesteps += 1

            if not args.headless:
                env.render()

        # When done..
        (ob1, ob2) = env.reset()
        # plot(ob1) # plot the reset observation
        if ((episode_no +1 ) % 20) == 0:
            print("episode {} over - reward = {}".format(episode_no + 1, reward_n / 10))
            reward_n = 0

        if ((episode_no + 1) % 10) == 0:
            player.episode_finished(episode_no)

    # Needs to be called in the end to shut down pygame
    env.end()

if __name__ == "__main__":
    main()