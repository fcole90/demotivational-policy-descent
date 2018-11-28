import random

from demotivational_policy_descent.environment.pong import Pong


class PongAi(object):
    def __init__(self, env, player_id=1):
        if type(env) is not Pong:
            raise TypeError("I'm not a very smart AI. All I can play is Pong.")
        self.env = env
        self.player_id = player_id
        self.bpe = 4
        self.name = "SimpleAI"

    def get_name(self):
        return self.name

    def get_action(self, ob=None):
        player = self.env.player1 if self.player_id == 1 else self.env.player2
        my_y = player.y
        ball_y = self.env.ball.y + (random.random() * self.bpe - self.bpe / 2)
        y_diff = my_y - ball_y

        if abs(y_diff) < 2:
            action = 0  # Stay
        else:
            if y_diff > 0:
                action = self.env.MOVE_UP  # Up
            else:
                action = self.env.MOVE_DOWN  # Down

        return action

    def reset(self):
        # Nothing to done for now...
        return
